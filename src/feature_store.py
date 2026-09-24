"""Task 5 — SageMaker Feature Store ingestion.

Feature group `diabetes130-encounter-features`, offline store only (this is
a batch scoring system — there is no online serving requirement).

Uses boto3 directly rather than the SageMaker SDK's high-level Feature Store
helpers: the SDK's Feature Store module has been substantially restructured
across versions, while the boto3 control-plane (`create_feature_group`,
`describe_feature_group`) and data-plane (`put_record`) APIs are stable.

Handles the five gotchas from the brief:
  1. Casts every object column to str and every bool to int (Feature Store
     only accepts String, Integral, Fractional).
  2. Validates feature names against `[a-zA-Z0-9_-]+`, no leading digit.
  3. Fills nulls in string columns with an explicit sentinel before
     ingesting (nulls in numeric columns are left as-is — permitted).
  4. Polls Athena for the expected row count after ingestion instead of
     assuming the (asynchronous) offline store landed immediately.
  5. Reads back the offline store's auto-created Glue table location from
     `describe_feature_group` rather than guessing its name.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from config import load_config

# Error codes worth retrying (throttling / transient); anything else (bad
# input, auth) fails fast instead of retrying uselessly.
RETRYABLE_ERROR_CODES = {
    "ThrottlingException", "ServiceUnavailable", "InternalFailure",
    "RequestLimitExceeded", "TooManyRequestsException",
}

FEATURE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")
RECORD_IDENTIFIER = "encounter_id"
EVENT_TIME_FEATURE = "event_time"
NULL_SENTINEL = "Missing"


def _execution_role_arn() -> str:
    metadata_path = Path("/opt/ml/metadata/resource-metadata.json")
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if "ExecutionRoleArn" in meta:
            return meta["ExecutionRoleArn"]
    raise RuntimeError(
        "Could not determine the SageMaker execution role automatically; "
        "pass role_arn explicitly."
    )


def validate_feature_names(columns: list[str]) -> None:
    bad = [c for c in columns if not FEATURE_NAME_RE.match(c)]
    if bad:
        raise ValueError(f"invalid Feature Store feature names (must match "
                          f"[a-zA-Z0-9_-]+ and not start with a digit): {bad}")


def prepare_for_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    """Cast dtypes and fill string nulls per the Feature Store gotchas."""
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str).where(df[col].notna(), None)
            df[col] = df[col].fillna(NULL_SENTINEL)
            # category/object -> plain python str; "?"/nan already handled
            # upstream in transform.py, this is a defensive backstop.
            df[col] = df[col].apply(lambda v: NULL_SENTINEL if v is None else str(v))

    df[EVENT_TIME_FEATURE] = time.time()

    validate_feature_names(list(df.columns))
    return df


def _feature_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "Integral"
    if pd.api.types.is_float_dtype(dtype):
        return "Fractional"
    return "String"


def create_or_reuse_feature_group(
    sm_client,
    cfg,
    df: pd.DataFrame,
    role_arn: str,
) -> dict:
    """Create the feature group if it doesn't exist; return its description."""
    name = cfg.feature_group

    try:
        return sm_client.describe_feature_group(FeatureGroupName=name)
    except sm_client.exceptions.ResourceNotFound:
        pass

    feature_definitions = [
        {"FeatureName": col, "FeatureType": _feature_type(df[col].dtype)}
        for col in df.columns
    ]

    sm_client.create_feature_group(
        FeatureGroupName=name,
        RecordIdentifierFeatureName=RECORD_IDENTIFIER,
        EventTimeFeatureName=EVENT_TIME_FEATURE,
        FeatureDefinitions=feature_definitions,
        OfflineStoreConfig={
            "S3StorageConfig": {"S3Uri": cfg.s3_uri("features", "offline-store")},
        },
        RoleArn=role_arn,
        Description=f"Encounter-level features for {cfg.project} (offline store only).",
    )

    while True:
        desc = sm_client.describe_feature_group(FeatureGroupName=name)
        status = desc["FeatureGroupStatus"]
        if status == "Created":
            return desc
        if status == "CreateFailed":
            raise RuntimeError(f"feature group creation failed: {desc.get('FailureReason')}")
        time.sleep(5)


def _put_record(runtime_client, feature_group_name: str, record: dict, max_attempts: int = 5) -> None:
    fs_record = [
        {"FeatureName": k, "ValueAsString": str(v)}
        for k, v in record.items()
        if v is not None and not (isinstance(v, float) and pd.isna(v))
    ]
    for attempt in range(1, max_attempts + 1):
        try:
            runtime_client.put_record(FeatureGroupName=feature_group_name, Record=fs_record)
            return
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in RETRYABLE_ERROR_CODES or attempt == max_attempts:
                raise
            time.sleep(0.2 * (2 ** attempt))


def ingest_dataframe(runtime_client, feature_group_name: str, df: pd.DataFrame, max_workers: int = 20) -> None:
    records = df.to_dict(orient="records")
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_put_record, runtime_client, feature_group_name, rec): i
            for i, rec in enumerate(records)
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                future.result()
            except Exception as e:  # noqa: BLE001
                errors.append((futures[future], str(e)))
            if done % 10000 == 0:
                print(f"  ingested {done:,} / {len(records):,} records")

    if errors:
        raise RuntimeError(f"{len(errors)} records failed to ingest, e.g. row {errors[0][0]}: {errors[0][1]}")
    print(f"ingested {len(records):,} records into {feature_group_name}")


def poll_offline_store_row_count(cfg, offline_data_catalog: dict, expected_min: int,
                                  timeout_seconds: int = 1200, interval_seconds: int = 30) -> int:
    """Poll Athena against the auto-created offline-store Glue table.

    Offline store ingestion is asynchronous and can take up to ~15 minutes
    to appear, so this polls rather than assuming it has already landed.
    """
    import awswrangler as wr
    wr.engine.set("python")

    database = offline_data_catalog["Database"]
    table = offline_data_catalog["TableName"]
    s3_output = cfg.s3_uri("athena_results")

    deadline = time.time() + timeout_seconds
    last_count = 0
    while time.time() < deadline:
        try:
            df = wr.athena.read_sql_query(
                f'SELECT COUNT(*) AS n FROM "{database}"."{table}"',
                database=database, s3_output=s3_output,
            )
            last_count = int(df["n"].iloc[0])
            print(f"  offline store row count so far: {last_count:,} (target >= {expected_min:,})")
            if last_count >= expected_min:
                return last_count
        except Exception as e:  # noqa: BLE001
            print(f"  query not ready yet ({e}); retrying...")
        time.sleep(interval_seconds)

    raise TimeoutError(
        f"offline store only reached {last_count:,} rows after {timeout_seconds}s, "
        f"expected >= {expected_min:,}"
    )


def run(df: pd.DataFrame, role_arn: str | None = None) -> dict:
    """End-to-end: prepare, create/reuse the feature group, ingest, verify."""
    cfg = load_config()
    role_arn = role_arn or _execution_role_arn()

    prepared = prepare_for_feature_store(df)

    sm_client = boto3.client("sagemaker", region_name=cfg.region)
    runtime_client = boto3.client("sagemaker-featurestore-runtime", region_name=cfg.region)

    print(f"creating/reusing feature group {cfg.feature_group}...")
    fg_desc = create_or_reuse_feature_group(sm_client, cfg, prepared, role_arn)

    print(f"ingesting {len(prepared):,} records...")
    ingest_dataframe(runtime_client, cfg.feature_group, prepared)

    offline_catalog = fg_desc["OfflineStoreConfig"]["DataCatalogConfig"]
    print(f"polling offline store table {offline_catalog['Database']}.{offline_catalog['TableName']}...")
    row_count = poll_offline_store_row_count(cfg, offline_catalog, expected_min=len(prepared))

    return {
        "feature_group": cfg.feature_group,
        "records_ingested": len(prepared),
        "offline_store_row_count": row_count,
        "offline_database": offline_catalog["Database"],
        "offline_table": offline_catalog["TableName"],
    }


def main() -> None:
    import json as _json
    from pathlib import Path as _Path

    from catalog import split_ids_mapping
    from transform import clean_and_transform

    repo_root = _Path(__file__).resolve().parent.parent
    datasets_dir = repo_root / "Datasets"

    raw = pd.read_csv(datasets_dir / "diabetic_data.csv", dtype=str)
    id_maps = split_ids_mapping(datasets_dir / "IDS_mapping.csv")
    clean_df = clean_and_transform(raw, id_maps)

    result = run(clean_df)
    print(result)

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "task5_feature_store.json").write_text(_json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
