"""Shared helpers for loading the Week 3 splits for modeling.

Centralizes the feature-exclusion list so every training/evaluation script
enforces it the same way — an identifier leaking into training is the
classic way to get a suspiciously good model.
"""
from __future__ import annotations

import awswrangler as wr
import pandas as pd

from config import load_config

TARGET_COLUMN = "readmit_30"

# Never used as a model feature: identifiers, the split label, and the
# Feature Store event-time / system columns (present if loading from the
# offline store rather than the Task 6 split Parquet directly).
EXCLUDED_COLUMNS = [
    "encounter_id", "patient_nbr", "split",
    "event_time", "write_time", "api_invocation_time", "is_deleted",
]


def load_split(split_name: str) -> pd.DataFrame:
    """Read one split's Parquet partition from s3://<bucket>/features/split=<name>/."""
    wr.engine.set("python")
    cfg = load_config()
    return wr.s3.read_parquet(cfg.s3_uri("features", f"split={split_name}/"))


def assert_no_excluded_columns(feature_columns: list[str]) -> None:
    leaked = [c for c in feature_columns if c in EXCLUDED_COLUMNS or c == TARGET_COLUMN]
    assert not leaked, f"identifier/target columns leaked into feature list: {leaked}"


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded split frame into (X, y), dropping excluded columns."""
    drop_cols = [c for c in EXCLUDED_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols + [TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    assert_no_excluded_columns(list(X.columns))
    return X, y
