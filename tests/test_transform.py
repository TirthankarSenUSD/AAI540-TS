"""Unit tests for src/transform.py — no AWS credentials required.

Runs against a small in-memory fixture plus the real diagnosis-distribution
acceptance check against the full local dataset (skipped if that file isn't
present, e.g. in CI without the raw data mounted).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from catalog import split_ids_mapping  # noqa: E402
from split import (  # noqa: E402
    add_split_column,
    assert_no_patient_leakage,
    assert_prevalence_stable,
)
from transform import (  # noqa: E402
    DRUG_COLUMNS,
    build_target,
    clean_and_transform,
    drop_expired_hospice,
    icd9_group,
    replace_missing_markers,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "Datasets"


# ---------------------------------------------------------------------------
# icd9_group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected", [
    ("250.83", "Diabetes"),
    ("428.0", "Circulatory"),
    ("V58.49", "Other"),
    ("E885.9", "Other"),
    (float("nan"), "Missing"),
    ("abc", "Other"),
])
def test_icd9_group_known_codes(code, expected):
    assert icd9_group(code) == expected


# ---------------------------------------------------------------------------
# Fixture-based pipeline tests
# ---------------------------------------------------------------------------

def make_fixture() -> pd.DataFrame:
    """A tiny hand-built frame covering the cases each cleaning step must handle."""
    drug_cols = {c: "No" for c in DRUG_COLUMNS}

    def row(**overrides):
        base = dict(
            encounter_id="1", patient_nbr="100", race="Caucasian", gender="Female",
            age="[40-50)", weight="?", admission_type_id="1",
            discharge_disposition_id="1", admission_source_id="7",
            time_in_hospital="3", payer_code="?", medical_specialty="?",
            num_lab_procedures="40", num_procedures="0", num_medications="10",
            number_outpatient="0", number_emergency="0", number_inpatient="0",
            diag_1="250.83", diag_2="428.0", diag_3="?", number_diagnoses="3",
            max_glu_serum="?", A1Cresult="?",
            change="No", diabetesMed="Yes", readmitted="NO",
            **drug_cols,
        )
        base.update(overrides)
        return base

    rows = [
        row(encounter_id="1", patient_nbr="100", readmitted="<30"),
        row(encounter_id="2", patient_nbr="100", readmitted=">30"),
        row(encounter_id="3", patient_nbr="101", discharge_disposition_id="11", readmitted="NO"),
        row(encounter_id="4", patient_nbr="102", discharge_disposition_id="13", readmitted="NO"),
        row(encounter_id="5", patient_nbr="103", discharge_disposition_id="14", readmitted="NO"),
        row(encounter_id="6", patient_nbr="104", discharge_disposition_id="19", readmitted="NO"),
        row(encounter_id="7", patient_nbr="105", discharge_disposition_id="20", readmitted="NO"),
        row(encounter_id="8", patient_nbr="106", insulin="Up", metformin="Down", readmitted="NO"),
    ]
    return pd.DataFrame(rows)


def make_id_maps() -> dict[str, pd.DataFrame]:
    return {
        "admission_type": pd.DataFrame(
            {"admission_type_id": ["1"], "description": ["Emergency"]}
        ),
        "discharge_disposition": pd.DataFrame(
            {"discharge_disposition_id": ["1"], "description": ["Discharged to home"]}
        ),
        "admission_source": pd.DataFrame(
            {"admission_source_id": ["7"], "description": [" Emergency Room"]}
        ),
    }


def test_replace_missing_markers_removes_question_marks():
    df = make_fixture()
    out = replace_missing_markers(df)
    assert not (out.astype(str) == "?").any().any()


def test_drop_expired_hospice_removes_all_six_ids():
    df = replace_missing_markers(make_fixture())
    out = drop_expired_hospice(df)
    remaining_ids = out["discharge_disposition_id"].astype(int)
    assert not remaining_ids.isin([11, 13, 14, 19, 20, 21]).any()
    # only encounters 1 and 2 (patient 100) and 8 (patient 106) survive
    assert set(out["encounter_id"]) == {"1", "2", "8"}


def test_build_target_maps_readmitted_correctly():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
    out = build_target(df)
    assert list(out["readmit_30"]) == [1, 0, 0]
    assert "readmitted" not in out.columns


def test_full_pipeline_on_fixture():
    df = make_fixture()
    id_maps = make_id_maps()
    out = clean_and_transform(df, id_maps)

    # expired/hospice rows gone
    assert set(out["encounter_id"]) == {"1", "2", "8"}

    # target constructed
    assert out.set_index("encounter_id").loc["1", "readmit_30"] == 1
    assert out.set_index("encounter_id").loc["2", "readmit_30"] == 0

    # low-value columns dropped
    for col in ["weight", "citoglipton", "examide", "readmitted"]:
        assert col not in out.columns

    # record identifiers survive
    assert "encounter_id" in out.columns
    assert "patient_nbr" in out.columns

    # icd9 groups
    assert out.set_index("encounter_id").loc["1", "diag_1_group"] == "Diabetes"
    assert out.set_index("encounter_id").loc["1", "diag_2_group"] == "Circulatory"
    assert out.set_index("encounter_id").loc["1", "diag_3_group"] == "Missing"

    # derived med-change features for the row with Up/Down values
    r8 = out.set_index("encounter_id").loc["8"]
    assert r8["n_med_changes"] == 2  # insulin=Up, metformin=Down
    assert r8["n_diabetes_meds"] == 2
    assert r8["insulin"] == "Up"
    assert r8["metformin"] == "Down"

    # 21 non-kept drug columns dropped, insulin/metformin retained
    kept_drug_cols = {c for c in DRUG_COLUMNS if c in out.columns}
    assert kept_drug_cols == {"insulin", "metformin"}

    # decoded ID mapping columns
    assert out.set_index("encounter_id").loc["1", "admission_type"] == "Emergency"
    assert out.set_index("encounter_id").loc["1", "discharge_disposition"] == "Discharged to home"

    # missing categorical fills
    assert (out["medical_specialty"] == "Unknown").all()
    assert (out["payer_code"] == "Unknown").all()
    assert (out["specialty_missing"] == 1).all()


# ---------------------------------------------------------------------------
# Acceptance criterion against the full real dataset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (DATASETS_DIR / "diabetic_data.csv").exists(),
    reason="raw dataset not present locally",
)
def test_diag_1_group_distribution_within_one_point():
    df = pd.read_csv(DATASETS_DIR / "diabetic_data.csv", dtype=str)
    id_maps = split_ids_mapping(DATASETS_DIR / "IDS_mapping.csv")
    out = clean_and_transform(df, id_maps)

    expected = {
        "Circulatory": 30.6, "Respiratory": 13.6, "Digestive": 9.3,
        "Diabetes": 8.2, "Injury": 6.7, "Musculoskeletal": 5.8,
        "Genitourinary": 4.9, "Neoplasms": 3.6, "Other": 17.3,
    }
    actual = (out["diag_1_group"].value_counts(normalize=True) * 100)

    for group, expected_pct in expected.items():
        actual_pct = actual.get(group, 0.0)
        assert abs(actual_pct - expected_pct) <= 1.0, (
            f"{group}: expected {expected_pct}pp, got {actual_pct:.2f}pp"
        )


@pytest.mark.skipif(
    not (DATASETS_DIR / "diabetic_data.csv").exists(),
    reason="raw dataset not present locally",
)
def test_row_count_after_cleaning_matches_expected():
    df = pd.read_csv(DATASETS_DIR / "diabetic_data.csv", dtype=str)
    id_maps = split_ids_mapping(DATASETS_DIR / "IDS_mapping.csv")
    out = clean_and_transform(df, id_maps)
    assert abs(len(out) - 99_340) < 100


# ---------------------------------------------------------------------------
# Task 6 split assertions (1 and 4), run as tests on a fixture.
# ---------------------------------------------------------------------------

def make_split_fixture(n_patients: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Patients with varying encounter counts and a ~11% positive rate,
    similar in shape to the real post-cleaning dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    encounter_id = 0
    for patient_id in range(n_patients):
        n_encounters = rng.choice([1, 1, 1, 2, 2, 3], size=1)[0]
        for _ in range(n_encounters):
            encounter_id += 1
            rows.append({
                "encounter_id": encounter_id,
                "patient_nbr": patient_id,
                "readmit_30": int(rng.random() < 0.11),
            })
    return pd.DataFrame(rows)


SPLITS_CONFIG = {"train": 0.40, "validation": 0.10, "test": 0.10, "production": 0.40}


def test_split_assertion_1_no_patient_leakage():
    df = make_split_fixture()
    split_df = add_split_column(df, seed=42, splits=SPLITS_CONFIG)
    assert_no_patient_leakage(split_df)  # must not raise

    # Sanity check the assertion actually catches leakage if introduced.
    tampered = split_df.copy()
    leak_patient = tampered["patient_nbr"].iloc[0]
    tampered.loc[tampered["patient_nbr"] == leak_patient, "split"] = (
        ["train", "production"] * len(tampered[tampered["patient_nbr"] == leak_patient])
    )[: (tampered["patient_nbr"] == leak_patient).sum()]
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_patient_leakage(tampered)


def test_split_assertion_4_prevalence_stable():
    df = make_split_fixture()
    split_df = add_split_column(df, seed=42, splits=SPLITS_CONFIG)
    assert_prevalence_stable(split_df)  # must not raise, real data is well within 2pp

    # Sanity check the assertion actually catches drift if introduced.
    drifted = split_df.copy()
    drifted.loc[drifted["split"] == "train", "readmit_30"] = 1
    with pytest.raises(AssertionError, match="prevalence drift"):
        assert_prevalence_stable(drifted)
