"""Task 2 — Glue catalog and Athena tables over the raw data.

Creates the `diabetes130` Glue database and external tables for:
  - raw_encounters   over raw/encounters/   (as uploaded by ingest.py)
  - raw_icd9         over raw/icd9/         (as uploaded by ingest.py)
  - admission_type, discharge_disposition, admission_source
    over processed/id_mapping/<name>/ — IDS_mapping.csv is three stacked
    tables separated by blank lines, not one table, so it cannot be
    cataloged directly. This script parses it into three clean CSVs first
    and writes those to `processed/` (the raw zone stays untouched), then
    catalogs each as its own external table.

All tables use OpenCSVSerde (handles quoted fields with embedded commas in
`medical_specialty` and the diagnosis descriptions) and declare every column
as `string` — OpenCSVSerde reads everything as string regardless of the
declared type, so casting happens downstream instead of fighting the SerDe.

Also runs the three validation queries from the brief and writes their
results to reports/task2_validation.json.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import awswrangler as wr
import pandas as pd

from config import load_config

# Force the plain-python engine — a distributed (Ray/Modin) engine may be
# auto-detected in this environment and doesn't support all kwargs used here.
wr.engine.set("python")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "Datasets"
REPORTS_DIR = REPO_ROOT / "reports"

OPEN_CSV_SERDE = "org.apache.hadoop.hive.serde2.OpenCSVSerde"
OPEN_CSV_SERDE_PARAMS = {"separatorChar": ",", "quoteChar": '"', "escapeChar": "\\"}

# All 50 encounter columns, declared as string per the brief's guidance.
ENCOUNTER_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed", "readmitted",
]

ICD9_COLUMNS = ["dgns_cd", "longdesc", "shortdesc", "version", "fyear"]

ID_MAPPING_TABLES = {
    "admission_type": ["admission_type_id", "description"],
    "discharge_disposition": ["discharge_disposition_id", "description"],
    "admission_source": ["admission_source_id", "description"],
}


def split_ids_mapping(path: Path) -> dict[str, pd.DataFrame]:
    """Parse IDS_mapping.csv's three stacked tables into separate frames.

    Blocks are separated by a line with no non-comma content (a truly blank
    line, or a line that is just a bare comma). Each block's first row is
    its own header.
    """
    text = path.read_text()
    lines = text.splitlines()

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip().strip(",") == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    if len(blocks) != 3:
        raise ValueError(f"expected 3 stacked tables in IDS_mapping.csv, found {len(blocks)}")

    names = list(ID_MAPPING_TABLES.keys())
    frames = {}
    for name, block_lines in zip(names, blocks):
        # keep_default_na=False: several descriptions are the literal text
        # "NULL" (e.g. admission_type_id 6) — pandas' default NA-string list
        # would otherwise silently turn that real value into a null.
        df = pd.read_csv(
            io.StringIO("\n".join(block_lines)), dtype=str,
            keep_default_na=False, na_values=[""],
        )
        expected_cols = ID_MAPPING_TABLES[name]
        assert list(df.columns) == expected_cols, (
            f"{name}: expected columns {expected_cols}, got {list(df.columns)}"
        )
        frames[name] = df
    return frames


def write_id_mapping_tables(cfg, frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        dest = cfg.s3_uri("processed", "id_mapping", name, f"{name}.csv")
        wr.s3.to_csv(df, dest, index=False)
        print(f"wrote {name} ({len(df)} rows) -> {dest}")


def create_tables(cfg) -> None:
    wr.catalog.create_database(cfg.glue_database, exist_ok=True)

    wr.catalog.create_csv_table(
        database=cfg.glue_database,
        table="raw_encounters",
        path=cfg.s3_uri("raw", "encounters/"),
        columns_types={c: "string" for c in ENCOUNTER_COLUMNS},
        serde_library=OPEN_CSV_SERDE,
        serde_parameters=OPEN_CSV_SERDE_PARAMS,
        skip_header_line_count=1,
        mode="overwrite",
    )

    wr.catalog.create_csv_table(
        database=cfg.glue_database,
        table="raw_icd9",
        path=cfg.s3_uri("raw", "icd9/"),
        columns_types={c: "string" for c in ICD9_COLUMNS},
        serde_library=OPEN_CSV_SERDE,
        serde_parameters=OPEN_CSV_SERDE_PARAMS,
        skip_header_line_count=1,
        mode="overwrite",
    )

    for name, cols in ID_MAPPING_TABLES.items():
        wr.catalog.create_csv_table(
            database=cfg.glue_database,
            table=name,
            path=cfg.s3_uri("processed", "id_mapping", name) + "/",
            columns_types={c: "string" for c in cols},
            serde_library=OPEN_CSV_SERDE,
            serde_parameters=OPEN_CSV_SERDE_PARAMS,
            skip_header_line_count=1,
            mode="overwrite",
        )

    print(f"catalog ready: {cfg.glue_database}.raw_encounters, raw_icd9, "
          f"{', '.join(ID_MAPPING_TABLES)}")


def run_validation_queries(cfg) -> dict:
    s3_output = cfg.s3_uri("athena_results")

    results = {}

    df = wr.athena.read_sql_query(
        f"SELECT COUNT(*) AS n FROM {cfg.glue_database}.raw_encounters",
        database=cfg.glue_database, s3_output=s3_output,
    )
    results["count_raw_encounters"] = {"actual": int(df["n"].iloc[0]), "expected": 101766}

    df = wr.athena.read_sql_query(
        f"SELECT COUNT(DISTINCT patient_nbr) AS n FROM {cfg.glue_database}.raw_encounters",
        database=cfg.glue_database, s3_output=s3_output,
    )
    results["count_distinct_patients"] = {"actual": int(df["n"].iloc[0]), "expected": 71518}

    df = wr.athena.read_sql_query(
        f"SELECT readmitted, COUNT(*) AS n FROM {cfg.glue_database}.raw_encounters "
        f"GROUP BY readmitted",
        database=cfg.glue_database, s3_output=s3_output,
    )
    breakdown = dict(zip(df["readmitted"], df["n"].astype(int)))
    results["readmitted_breakdown"] = {
        "actual": breakdown,
        "expected": {"<30": 11357, ">30": 35545, "NO": 54864},
    }

    return results


def main() -> None:
    cfg = load_config()

    frames = split_ids_mapping(DATASETS_DIR / "IDS_mapping.csv")
    write_id_mapping_tables(cfg, frames)

    create_tables(cfg)

    print("running validation queries...")
    results = run_validation_queries(cfg)
    for key, val in results.items():
        print(f"  {key}: {val}")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "task2_validation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"validation results written to {out_path}")

    count_ok = results["count_raw_encounters"]["actual"] == 101766
    patients_ok = results["count_distinct_patients"]["actual"] == 71518
    if not (count_ok and patients_ok):
        raise SystemExit("validation counts do not match expected values — fix SerDe/header config")


if __name__ == "__main__":
    main()
