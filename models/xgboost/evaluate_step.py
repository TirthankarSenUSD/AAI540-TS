"""Pipeline evaluation step -- scores the just-trained model against the
test split and writes evaluation.json, then enforces the quality gate
itself (AUC >= $AUC_THRESHOLD) by exiting non-zero on failure.

The gate is enforced here rather than via a native Pipelines Condition
step + PropertyFile reference: a hand-written `{"Get":
"Steps.X.PropertyFiles.Y.z"}` expression hit "Unknown property reference"
against the real service, and burning more cycles guessing at the exact
undocumented-to-us JSON syntax wasn't worth it when this achieves the
identical demonstrable outcome (the pipeline reaches RegisterModel on a
pass, stops with a failed, red step on a miss) through a mechanism (a
step's own exit code) that has no such syntax ambiguity.

Deliberately dependency-light: reads test data as plain CSV (no pyarrow/
awswrangler needed, avoiding the numpy/pandas ABI break hit earlier in
src/fairness_report.py), and only needs the model + preprocessor, not the
calibrator (this is a CI/CD DAG demonstration -- the pipeline's own
registered model is separate from the hand-calibrated, human-approved
production model already deployed via src/deploy.py).
"""
import json
import os
import sys
import tarfile

sys.path.insert(0, "/opt/ml/processing/input/code")

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from preprocess import XGBoostCategoricalPreprocessor  # noqa: F401  needed to unpickle

MODEL_INPUT_DIR = "/opt/ml/processing/input/model"
TEST_INPUT_DIR = "/opt/ml/processing/input/test"
OUTPUT_DIR = "/opt/ml/processing/output"

TARGET = "readmit_30"
EXCLUDED_COLUMNS = [
    "encounter_id", "patient_nbr", "split",
    "event_time", "write_time", "api_invocation_time", "is_deleted",
]


def main():
    import joblib

    model_tarball = [f for f in os.listdir(MODEL_INPUT_DIR) if f.endswith(".tar.gz")][0]
    extract_dir = "/opt/ml/processing/input/model_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(os.path.join(MODEL_INPUT_DIR, model_tarball)) as tar:
        tar.extractall(extract_dir)

    preprocessor = joblib.load(os.path.join(extract_dir, "preprocessor.joblib"))
    booster = xgb.XGBClassifier()
    booster.load_model(os.path.join(extract_dir, "model.ubj"))

    test_files = [f for f in os.listdir(TEST_INPUT_DIR) if f.endswith(".csv")]
    test_df = pd.concat([pd.read_csv(os.path.join(TEST_INPUT_DIR, f)) for f in test_files], ignore_index=True)

    drop_cols = [c for c in EXCLUDED_COLUMNS if c in test_df.columns]
    X_test = test_df.drop(columns=drop_cols + [TARGET])
    y_test = test_df[TARGET].astype(int)

    X_t = preprocessor.transform(X_test)
    scores = booster.predict_proba(X_t)[:, 1]

    report = {
        "auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "n_rows": len(test_df),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    threshold = float(os.environ.get("AUC_THRESHOLD", "0.0"))
    if report["auc"] < threshold:
        print(f"QUALITY GATE FAILED: auc {report['auc']:.4f} < threshold {threshold:.4f}")
        sys.exit(1)
    print(f"quality gate passed: auc {report['auc']:.4f} >= threshold {threshold:.4f}")


if __name__ == "__main__":
    main()
