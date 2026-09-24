"""Task 2 — calibration.

Fits Platt scaling on the trained XGBoost model's **validation**-split
predictions (never train — that would just re-fit to the same data the
model already saw). Runs locally against a downloaded training-job
artifact rather than as its own SageMaker job — calibration on ~10k rows
is a few seconds of work, not worth a managed job.

NOTE on the brief's exact API: `CalibratedClassifierCV(..., cv="prefit")`
looked like the obvious choice, but it (and its modern replacement,
wrapping the estimator in `sklearn.frozen.FrozenEstimator`) pickles the
*entire* wrapped model into the calibrator artifact, and that pickle's
compatibility across sklearn versions is not guaranteed — this genuinely
broke Batch Transform in this project: the dev environment's sklearn
(1.7.2) produced a `calibrator.joblib` referencing `sklearn.frozen`, which
doesn't exist in the older sklearn shipped in the SageMaker XGBoost 1.7-1
serving container, so `model_fn` failed on every request. Platt scaling is
just a 1-D logistic regression on the raw score, so
`models/xgboost/calibration.py`'s `PlattCalibrator` stores only the two
fitted floats — no sklearn object to unpickle at serving time, and no
version-skew surface at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

MODELS_XGBOOST_DIR = Path(__file__).resolve().parent.parent / "models" / "xgboost"


def load_trained_model(model_dir: Path) -> tuple[xgb.XGBClassifier, object]:
    """Load the booster + preprocessor saved by models/xgboost/train.py."""
    sys.path.insert(0, str(MODELS_XGBOOST_DIR))  # for unpickling XGBoostCategoricalPreprocessor

    model = xgb.XGBClassifier()
    model.load_model(str(model_dir / "model.ubj"))
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    return model, preprocessor


def fit_calibrator(raw_scores: np.ndarray, y_val: pd.Series):
    if str(MODELS_XGBOOST_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_XGBOOST_DIR))
    from calibration import PlattCalibrator  # models/xgboost/calibration.py

    lr = LogisticRegression()
    lr.fit(raw_scores.reshape(-1, 1), y_val)
    return PlattCalibrator(a=float(lr.coef_[0][0]), b=float(lr.intercept_[0]))


def calibration_report(raw_scores: np.ndarray, cal_scores: np.ndarray, y_val: pd.Series) -> dict:
    return {
        "uncalibrated": {
            "roc_auc": float(roc_auc_score(y_val, raw_scores)),
            "brier_score": float(brier_score_loss(y_val, raw_scores)),
        },
        "calibrated": {
            "roc_auc": float(roc_auc_score(y_val, cal_scores)),
            "brier_score": float(brier_score_loss(y_val, cal_scores)),
        },
    }


def main(model_dir: str) -> None:
    import json

    from model_data import features_and_target, load_split

    model_dir_path = Path(model_dir)
    model, preprocessor = load_trained_model(model_dir_path)

    val_df = load_split("validation")
    X_val, y_val = features_and_target(val_df)
    X_val_t = preprocessor.transform(X_val)

    raw_scores = model.predict_proba(X_val_t)[:, 1]
    calibrator = fit_calibrator(raw_scores, y_val)
    cal_scores = calibrator.predict_proba(raw_scores)[:, 1]

    report = calibration_report(raw_scores, cal_scores, y_val)
    print(json.dumps(report, indent=2))

    joblib.dump(calibrator, model_dir_path / "calibrator.joblib")
    (model_dir_path / "calibration_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="local directory holding model.ubj + preprocessor.joblib")
    args = parser.parse_args()
    main(args.model_dir)
