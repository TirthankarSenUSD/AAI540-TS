"""Task 4 — cleaning and feature engineering.

Pure functions only: no S3/Athena calls in here, so the whole pipeline is
unit-testable against a small in-memory fixture (see tests/test_transform.py).
Callers are responsible for loading the raw encounters frame and the three
ID-mapping frames (e.g. via Athena) and passing them in.

This stage produces a clean, typed, human-readable feature table. It does
not scale, encode, or impute — those are model-stage operations fit on
training data only.
"""
from __future__ import annotations

import pandas as pd

# Discharge dispositions meaning the patient died or entered hospice — they
# cannot be readmitted, so these encounters are removed before anything else.
#
# NOTE: the Week 3 brief's Task 4 literally lists {11, 19, 20, 21} (expired
# variants only) but separately states the decision as "expired AND hospice
# discharges are removed" and expects ~101,766 -> ~99,340 rows. The literal
# 4-ID set only drops 1,652 rows (-> 100,114), not enough to reach ~99,340.
# Adding 13 ("Hospice / home") and 14 ("Hospice / medical facility") drops
# 2,423 rows -> 99,343, matching both the stated row count and the plain-
# language rule. Treating {11, 19, 20, 21} as an incomplete transcription;
# see RESULTS.md.
EXPIRED_HOSPICE_DISPOSITION_IDS = {11, 13, 14, 19, 20, 21}

# All 23 individual drug/dosage-change columns in the raw data.
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# High-prevalence drugs kept as individual categoricals rather than folded
# into the aggregate counts only.
KEPT_INDIVIDUAL_DRUGS = ["insulin", "metformin"]

# Columns dropped outright: weight (~97% null), citoglipton/examide (zero
# variance).
LOW_VALUE_COLUMNS = ["weight", "citoglipton", "examide"]

AGE_BANDS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
AGE_ORDINAL_MAP = {band: i for i, band in enumerate(AGE_BANDS)}

# ICD-9 group ranges, from Strack et al. (2014) Table 2.
DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]

# Columns that are always fully populated, clearly-numeric IDs/counts in the
# raw data (verified against the full dataset — zero non-numeric/missing
# values). OpenCSVSerde forces every Athena column to read as string
# (Task 2); this is where that "cast downstream" happens for the columns
# where it's unambiguous.
NUMERIC_COLUMNS = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
# encounter_id / patient_nbr are identifiers, not modeling features — kept
# as strings so they're never accidentally treated as a numeric feature.

ID_MAPPING_MERGE_KEYS = {
    "admission_type": ("admission_type_id", "admission_type"),
    "discharge_disposition": ("discharge_disposition_id", "discharge_disposition"),
    "admission_source": ("admission_source_id", "admission_source"),
}


def icd9_group(code) -> str:
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code.startswith(("V", "E")):
        return "Other"
    try:
        v = float(code)
    except ValueError:
        return "Other"
    if 250 <= v < 251:
        return "Diabetes"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: replace the literal string "?" with null across all columns."""
    return df.replace("?", pd.NA)


def drop_expired_hospice(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: drop rows whose discharge means the patient died or entered hospice."""
    disposition = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    return df[~disposition.isin(EXPIRED_HOSPICE_DISPOSITION_IDS)].copy()


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: binary target readmit_30 = (readmitted == "<30"); drop readmitted."""
    df = df.copy()
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns=["readmitted"])


def drop_low_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: drop weight (~97% null) and the zero-variance drug columns."""
    return df.drop(columns=LOW_VALUE_COLUMNS, errors="ignore")


def add_icd9_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DIAG_COLUMNS:
        df[f"{col}_group"] = df[col].map(icd9_group)
    return df


def add_derived_features(df: pd.DataFrame, id_maps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = df.copy()

    df["age_ordinal"] = df["age"].map(AGE_ORDINAL_MAP)

    # citoglipton/examide were already dropped as zero-variance columns
    # (always "No"), so they never contribute to these counts regardless of
    # whether they're still present in the frame.
    present_drug_cols = [c for c in DRUG_COLUMNS if c in df.columns]
    df["n_med_changes"] = (df[present_drug_cols].isin(["Up", "Down"])).sum(axis=1)
    df["n_diabetes_meds"] = (df[present_drug_cols] != "No").sum(axis=1)

    df["a1c_tested"] = df["A1Cresult"].notna().astype(int)
    df["a1c_level"] = df["A1Cresult"].fillna("NotTested")

    df["glu_tested"] = df["max_glu_serum"].notna().astype(int)
    df["glu_level"] = df["max_glu_serum"].fillna("NotTested")

    df["specialty_missing"] = df["medical_specialty"].isna().astype(int)
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")

    df["payer_missing"] = df["payer_code"].isna().astype(int)
    df["payer_code"] = df["payer_code"].fillna("Unknown")

    for map_name, (id_col, out_col) in ID_MAPPING_MERGE_KEYS.items():
        mapping = id_maps[map_name]
        id_to_desc = dict(zip(
            mapping[mapping.columns[0]].astype(str),
            mapping[mapping.columns[1]],
        ))
        df[out_col] = df[id_col].astype(str).map(id_to_desc)

    df["prior_visits_total"] = (
        pd.to_numeric(df["number_inpatient"], errors="coerce").fillna(0)
        + pd.to_numeric(df["number_emergency"], errors="coerce").fillna(0)
        + pd.to_numeric(df["number_outpatient"], errors="coerce").fillna(0)
    )

    drop_drug_cols = [c for c in DRUG_COLUMNS if c not in KEPT_INDIVIDUAL_DRUGS and c in df.columns]
    df = df.drop(columns=drop_drug_cols)

    return df


def cast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast clearly-numeric, always-populated columns from string to int64.

    This is the "cast downstream" step referenced in Task 2 — OpenCSVSerde
    forces every Athena column to read as string, so real typing happens
    here instead of fighting the SerDe.
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col]).astype("int64")
    return df


def clean_and_transform(df: pd.DataFrame, id_maps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run the full Task 4 pipeline, in the order specified in the brief."""
    df = replace_missing_markers(df)
    df = drop_expired_hospice(df)
    df = build_target(df)
    df = drop_low_value_columns(df)
    df = add_icd9_groups(df)
    df = add_derived_features(df, id_maps)
    df = cast_numeric_columns(df)
    return df
