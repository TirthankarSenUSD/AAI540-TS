"""Preprocessing for the XGBoost model — fit on training data only.

`transform.py` from brief 1 produced a clean, typed, human-readable table.
It deliberately did not encode or scale. This module does that, at model
stage, fit on train only, then persisted alongside the model so serving
never silently drifts from what training saw.

Uses XGBoost's native categorical handling (`enable_categorical=True`)
rather than one-hot encoding — several columns (`diag_1`, `diag_2`,
`diag_3`, `medical_specialty`) are high cardinality (hundreds of levels),
and one-hot would blow those up into hundreds of sparse columns for no
benefit over native categorical splits.

Lives inside `models/xgboost/` (not `src/`) so the SageMaker training
container's source bundle is self-contained: `train.py` imports this
module directly from the same uploaded source directory. Any script that
later loads a fitted preprocessor via joblib (calibration, evaluation) must
put this directory on `sys.path` first so the pickle resolves.
"""
from __future__ import annotations

import pandas as pd


class XGBoostCategoricalPreprocessor:
    """Casts object/string columns to pandas `category` dtype (categories
    fixed from training data) and numeric columns to float64. Both XGBoost
    and pandas represent missing values natively (NaN / unknown category),
    so nulls are left as-is rather than imputed here.
    """

    def __init__(self):
        self.categorical_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categories_: dict[str, list] = {}

    def fit(self, X: pd.DataFrame) -> "XGBoostCategoricalPreprocessor":
        self.categorical_columns_ = [
            c for c in X.columns if X[c].dtype in ("object",) or str(X[c].dtype) == "string"
        ]
        self.numeric_columns_ = [c for c in X.columns if c not in self.categorical_columns_]

        for col in self.categorical_columns_:
            # Categories from training data only — unseen categories at
            # inference become NaN (XGBoost treats as missing), never a
            # silent new split.
            self.categories_[col] = sorted(X[col].dropna().unique().tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.categorical_columns_:
            X[col] = pd.Categorical(X[col], categories=self.categories_[col])
        for col in self.numeric_columns_:
            X[col] = X[col].astype("float64")
        return X[self.categorical_columns_ + self.numeric_columns_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    @property
    def feature_names_(self) -> list[str]:
        return self.categorical_columns_ + self.numeric_columns_
