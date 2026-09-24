"""Task 6 — the four-way patient-level split.

Splits on patients, not encounters, into train/validation/test/production
partitions and writes each to s3://<bucket>/features/split=<name>/ as
Parquet. The production holdout simulates incoming production traffic for
later monitoring/drift work — it never touches training or evaluation.

Read the brief's "Read this carefully" note before changing the class-floor
assertion: the 10,000-per-class requirement applies to the FULL dataset
before splitting (which satisfies it), not to the 60% modelling subset
(train+val+test), which will legitimately fall short (~6,600 positives).
That's an expected, reported statistic — not a gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import awswrangler as wr
import numpy as np
import pandas as pd

from config import load_config

CLASS_FLOOR = 10_000
PROPORTION_TOLERANCE = 0.02
PREVALENCE_TOLERANCE = 0.02


def assign_splits(patient_nbr: pd.Series, seed: int, splits: dict) -> pd.Series:
    """Partition the shuffled, unique patient array into four groups.

    Deliberately not nested GroupShuffleSplit calls — partitioning a single
    shuffled array keeps the four proportions easy to reason about and
    review, per the brief.
    """
    rng = np.random.default_rng(seed)
    patients = patient_nbr.unique()
    rng.shuffle(patients)

    names = ["train", "validation", "test", "production"]
    fractions = [splits[n] for n in names]
    assert abs(sum(fractions) - 1.0) < 1e-9, f"splits must sum to 1.0, got {fractions}"

    n = len(patients)
    bounds = np.cumsum(fractions)[:-1] * n
    groups = np.split(patients, bounds.astype(int))

    assignment = {}
    for name, group in zip(names, groups):
        assignment.update({p: name for p in group})

    return patient_nbr.map(assignment)


def add_split_column(df: pd.DataFrame, seed: int, splits: dict) -> pd.DataFrame:
    df = df.copy()
    df["split"] = assign_splits(df["patient_nbr"], seed, splits)
    return df


# ---------------------------------------------------------------------------
# Assertions — all must fail the build, not warn.
# ---------------------------------------------------------------------------

def assert_no_patient_leakage(df: pd.DataFrame) -> None:
    counts = df.groupby("patient_nbr")["split"].nunique()
    assert (counts == 1).all(), "patient leakage across splits"


def assert_split_proportions(df: pd.DataFrame, splits: dict, tol: float = PROPORTION_TOLERANCE) -> None:
    for name, target in splits.items():
        actual = (df["split"] == name).mean()
        assert abs(actual - target) < tol, f"{name}: {actual:.3f} vs {target}"


def assert_full_dataset_class_floor(full_df: pd.DataFrame, floor: int = CLASS_FLOOR) -> None:
    """The 10,000-per-class floor applies to the FULL dataset before
    splitting — see the module docstring. Do not run this on the modelling
    subset; it will legitimately fail there."""
    pos = int(full_df["readmit_30"].sum())
    neg = len(full_df) - pos
    assert pos >= floor, f"positive class {pos} below project floor on full dataset"
    assert neg >= floor, f"negative class {neg} below project floor on full dataset"


def assert_prevalence_stable(df: pd.DataFrame, tol: float = PREVALENCE_TOLERANCE) -> None:
    prev = df.groupby("split")["readmit_30"].mean()
    assert prev.max() - prev.min() < tol, f"prevalence drift:\n{prev}"


def run_all_assertions(full_df: pd.DataFrame, split_df: pd.DataFrame, splits: dict) -> dict:
    """Run every gating assertion and return the statistics for the report."""
    assert_no_patient_leakage(split_df)
    assert_split_proportions(split_df, splits)
    assert_full_dataset_class_floor(full_df)
    assert_prevalence_stable(split_df)

    model_df = split_df[split_df["split"] != "production"]
    model_pos = int(model_df["readmit_30"].sum())
    model_neg = len(model_df) - model_pos

    stats = {
        "full_dataset": {"n": len(full_df), "positives": int(full_df["readmit_30"].sum())},
        "split_proportions": {
            name: float((split_df["split"] == name).mean()) for name in splits
        },
        "split_counts": {
            name: int((split_df["split"] == name).sum()) for name in splits
        },
        "prevalence_by_split": split_df.groupby("split")["readmit_30"].mean().to_dict(),
        "modelling_portion": {
            "n": len(model_df),
            "positives": model_pos,
            "negatives": model_neg,
            "below_10k_floor": model_pos < CLASS_FLOOR,
            "note": (
                "Expected: the 60% modelling subset (train+val+test) legitimately "
                "falls below the 10,000-per-class floor even though the full "
                "dataset satisfies it. This is a reported statistic, not a gate — "
                "see the brief's 'Read this carefully' note on Task 6. Flagged for "
                "the team to consider whether the 40% production reservation "
                "should be revisited."
            ),
        },
    }
    return stats


# ---------------------------------------------------------------------------
# Write to S3
# ---------------------------------------------------------------------------

def write_splits(df: pd.DataFrame, cfg) -> str:
    """Write to s3://<bucket>/features/split=<name>/ as Parquet, partitioned
    so Athena can read each split independently, and register/refresh the
    Glue table in the same call."""
    dest = cfg.s3_uri("features")
    wr.s3.to_parquet(
        df=df,
        path=dest,
        dataset=True,
        partition_cols=["split"],
        mode="overwrite",
        database=cfg.glue_database,
        table="model_features",
    )
    return dest


def main() -> None:
    from catalog import split_ids_mapping
    from transform import clean_and_transform

    repo_root = Path(__file__).resolve().parent.parent
    datasets_dir = repo_root / "Datasets"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    cfg = load_config()

    raw = pd.read_csv(datasets_dir / "diabetic_data.csv", dtype=str)
    id_maps = split_ids_mapping(datasets_dir / "IDS_mapping.csv")
    full_df = clean_and_transform(raw, id_maps)

    split_df = add_split_column(full_df, seed=cfg.seed, splits=cfg.splits)

    print("running assertions...")
    stats = run_all_assertions(full_df, split_df, cfg.splits)
    print(json.dumps(stats, indent=2, default=str))

    print(f"writing splits to {cfg.s3_uri('features')} (and cataloging model_features)...")
    write_splits(split_df, cfg)

    (reports_dir / "task6_split.json").write_text(json.dumps(stats, indent=2, default=str))
    print("done")


if __name__ == "__main__":
    main()
