"""Task 4 — deployment.

Batch Transform on an EventBridge daily schedule, not a real-time endpoint:
discharge cohorts finalize on a daily cycle and the care team works the
list the next morning, so nothing needs sub-second inference, and a
persistent endpoint would bill continuously for zero operational benefit.

The deployable unit is the whole artifact bundle (preprocessor + model +
calibrator + threshold + feature schema), not the model file alone — ship
these separately and they drift apart.
"""
from __future__ import annotations

import json
import shutil
import tarfile
import time
from datetime import date
from pathlib import Path

import boto3
import pandas as pd

from config import load_config

MODELS_XGBOOST_DIR = Path(__file__).resolve().parent.parent / "models" / "xgboost"
BUNDLE_FILES = ["preprocessor.joblib", "model.ubj", "calibrator.joblib", "feature_schema.json"]
CODE_FILES = ["inference.py", "preprocess.py", "calibration.py"]


# ---------------------------------------------------------------------------
# Artifact bundle
# ---------------------------------------------------------------------------

def write_threshold_file(model_dir: Path, threshold: float, target_alert_fraction: float) -> Path:
    path = model_dir / "threshold.json"
    path.write_text(json.dumps({
        "threshold": threshold,
        "target_alert_fraction": target_alert_fraction,
        "selected_on": "validation",
        "frozen_at": pd.Timestamp.utcnow().isoformat(),
    }, indent=2))
    return path


def bundle_model(model_dir: str, output_tarball: str) -> str:
    """Assemble model.tar.gz: the 5 artifact files plus code/inference.py
    and code/preprocess.py (needed for the SageMaker container to serve
    predictions and to unpickle the preprocessor)."""
    model_dir_path = Path(model_dir)
    for required in BUNDLE_FILES + ["threshold.json"]:
        if not (model_dir_path / required).exists():
            raise FileNotFoundError(
                f"{required} missing from {model_dir} — run train.py + calibrate.py "
                f"+ write_threshold_file first"
            )

    staging = model_dir_path / "_bundle_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    (staging / "code").mkdir()

    for f in BUNDLE_FILES + ["threshold.json"]:
        shutil.copy(model_dir_path / f, staging / f)
    for f in CODE_FILES:
        shutil.copy(MODELS_XGBOOST_DIR / f, staging / "code" / f)

    with tarfile.open(output_tarball, "w:gz") as tar:
        for item in staging.iterdir():
            tar.add(item, arcname=item.name)

    shutil.rmtree(staging)
    return output_tarball


# ---------------------------------------------------------------------------
# S3 / Model Registry / Batch Transform (boto3 — not yet invoked)
# ---------------------------------------------------------------------------

def upload_bundle(cfg, tarball_path: str, version_tag: str) -> str:
    s3 = boto3.client("s3", region_name=cfg.region)
    key = f"{cfg.prefixes['models']}{version_tag}/model.tar.gz"
    s3.upload_file(tarball_path, cfg.bucket, key)
    return f"s3://{cfg.bucket}/{key}"


def ensure_model_package_group(sm_client, group_name: str, description: str) -> None:
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group_name)
    except sm_client.exceptions.ClientError:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=description,
        )


def register_model_package(sm_client, cfg, group_name: str, model_data_url: str,
                            image_uri: str, role_arn: str, metrics: dict) -> str:
    """Register into a ModelPackageGroup with PendingManualApproval.
    Attaches validation/test metrics as a JSON report in S3, referenced via
    ModelMetrics, so the registry shows performance alongside the version."""
    ensure_model_package_group(sm_client, group_name, f"Models for {cfg.project}")

    s3 = boto3.client("s3", region_name=cfg.region)
    metrics_key = f"models/{group_name}/metrics/{int(time.time())}.json"
    s3.put_object(
        Bucket=cfg.bucket, Key=metrics_key,
        Body=json.dumps(metrics, indent=2).encode(), ContentType="application/json",
    )
    metrics_s3_uri = f"s3://{cfg.bucket}/{metrics_key}"

    response = sm_client.create_model_package(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{
                "Image": image_uri,
                "ModelDataUrl": model_data_url,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                },
            }],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {"ContentType": "application/json", "S3Uri": metrics_s3_uri},
            },
        },
    )
    return response["ModelPackageArn"]


def create_sagemaker_model(sm_client, model_name: str, model_data_url: str,
                            image_uri: str, role_arn: str) -> str:
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        },
        ExecutionRoleArn=role_arn,
    )
    return model_name


def launch_batch_transform(sm_client, cfg, model_name: str, job_name: str,
                            input_s3_uri: str, output_s3_uri: str) -> str:
    """The join-back problem: input has [identifier, ...features], the
    model must not see the identifier, and the output must still be
    traceable to a patient."""
    sm_client.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_s3_uri}},
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": output_s3_uri,
            "Accept": "text/csv",
            "AssembleWith": "Line",
        },
        TransformResources={"InstanceType": "ml.m5.xlarge", "InstanceCount": 1},
        DataProcessing={
            "InputFilter": "$[1:]",   # drop the identifier column from what the model sees
            "JoinSource": "Input",    # attach input to output
            "OutputFilter": "$[0,-1]",  # keep identifier + prediction
        },
    )
    return job_name


def wait_for_transform_job(sm_client, job_name: str, poll_seconds: int = 15) -> str:
    while True:
        desc = sm_client.describe_transform_job(TransformJobName=job_name)
        status = desc["TransformJobStatus"]
        if status in ("Completed", "Failed", "Stopped"):
            return status
        time.sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Post-processing: rank, threshold, write the worklist
# ---------------------------------------------------------------------------

def write_ranked_worklist(cfg, raw_output_s3_prefix: str, threshold: float,
                           as_of_date: str | None = None) -> str:
    """Read the raw [identifier, score] Batch Transform output, sort
    descending by score, add the flag at the frozen threshold, and write
    the ranked worklist to s3://<bucket>/predictions/dt=<date>/."""
    import awswrangler as wr
    wr.engine.set("python")

    as_of_date = as_of_date or date.today().isoformat()

    raw = wr.s3.read_csv(raw_output_s3_prefix, header=None, names=["encounter_id", "score"])
    raw["score"] = raw["score"].astype(float)
    raw["flag"] = (raw["score"] >= threshold).astype(int)
    ranked = raw.sort_values("score", ascending=False).reset_index(drop=True)

    dest = cfg.s3_uri("predictions", f"dt={as_of_date}", "worklist.csv")
    wr.s3.to_csv(ranked, dest, index=False)
    return dest


# ---------------------------------------------------------------------------
# Smoke test — the training/serving skew regression test
# ---------------------------------------------------------------------------

def run_smoke_test(local_model_dir: str, canned_input_csv: str, transform_output_csv: str) -> dict:
    """Assert: output row count == input row count, all scores in [0, 1],
    and a known row's Batch Transform score matches what the same local
    artifact produces directly. This is the training/serving skew
    regression test and belongs in CI.
    """
    import sys

    sys.path.insert(0, str(MODELS_XGBOOST_DIR))
    import inference

    canned = pd.read_csv(canned_input_csv, header=None)
    identifiers = canned.iloc[:, 0]
    features_csv = canned.iloc[:, 1:].to_csv(header=False, index=False)

    bundle = inference.model_fn(local_model_dir)
    local_scores = inference.predict_fn(features_csv, bundle)

    transform_out = pd.read_csv(transform_output_csv, header=None, names=["encounter_id", "score"])

    checks = {
        "row_count_matches": len(transform_out) == len(canned),
        "scores_in_range": bool(((transform_out["score"] >= 0) & (transform_out["score"] <= 1)).all()),
    }

    # Skew check: first row's transform-job score vs. the same row scored
    # directly through the local artifact.
    first_id = identifiers.iloc[0]
    transform_score = float(transform_out.loc[transform_out["encounter_id"] == first_id, "score"].iloc[0])
    local_score = float(local_scores[0])
    checks["no_training_serving_skew"] = abs(transform_score - local_score) < 1e-4
    checks["transform_score_sample"] = transform_score
    checks["local_score_sample"] = local_score

    checks["passed"] = all([
        checks["row_count_matches"], checks["scores_in_range"], checks["no_training_serving_skew"],
    ])
    return checks
