"""Model monitor -- SageMaker Model Monitor, Model Quality type.

Baseline: classification-quality metrics (accuracy, precision, recall,
AUC, etc.) computed from the **validation** split's predictions vs. true
labels -- what "acceptable model quality" looks like. Monitoring: the
**production** split -- scored through the real deployed artifact
(preprocessor -> booster -> PlattCalibrator -> frozen threshold) and
compared against the baseline. In real production the labels wouldn't be
known yet at scoring time; this project's production split carries known
outcomes specifically so this monitoring pipeline can be built and
validated before it ever needs to run against genuinely-unlabeled data --
exactly the "monitoring and drift work" it was reserved for back in
Week 3's decision #5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb

from aws_jobs import execution_role_arn
from config import load_config
from monitor_common import (
    MODEL_MONITOR_ANALYZER_IMAGE,
    run_processing_job,
    s3_input,
    s3_output,
    sagemaker_client,
)

DEPLOYED_MODEL_DIR = Path(__file__).resolve().parent.parent / ".deployed_model"
MODELS_XGBOOST_DIR = Path(__file__).resolve().parent.parent / "models" / "xgboost"


def score_split(split_name: str, model_dir: Path = DEPLOYED_MODEL_DIR) -> pd.DataFrame:
    """Score a split through the exact deployed artifact (same
    preprocessor/model/calibrator/threshold as production), returning
    columns: probability, prediction, label -- the schema the Model
    Monitor model-quality analyzer expects for BinaryClassification."""
    sys.path.insert(0, str(MODELS_XGBOOST_DIR))
    import joblib

    from model_data import features_and_target, load_split

    df = load_split(split_name)
    X, y = features_and_target(df)

    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    booster = xgb.XGBClassifier()
    booster.load_model(str(model_dir / "model.ubj"))
    threshold = json.loads((model_dir / "threshold.json").read_text())["threshold"]

    X_t = preprocessor.transform(X)
    raw_scores = booster.predict_proba(X_t)[:, 1]
    calibrated = calibrator.predict_proba(raw_scores)[:, 1]
    prediction = (calibrated >= threshold).astype(int)

    return pd.DataFrame({
        "probability": calibrated,
        "prediction": prediction,
        "label": y.to_numpy(),
    })


def upload_scored_csv(cfg, scored_df: pd.DataFrame, key: str) -> str:
    local_path = Path("/tmp") / Path(key).name
    scored_df.to_csv(local_path, index=False)
    s3 = boto3.client("s3", region_name=cfg.region)
    s3.upload_file(str(local_path), cfg.bucket, key)
    return f"s3://{cfg.bucket}/{key}"


def _frozen_threshold(model_dir: Path = DEPLOYED_MODEL_DIR) -> float:
    return json.loads((model_dir / "threshold.json").read_text())["threshold"]


def model_quality_env() -> dict:
    """NOTE: this must use our actual frozen operating threshold, not the
    analyzer's 0.5 default. At ~11% prevalence, calibrated probabilities
    cluster well under 0.5 — a first run with the default left the
    confusion matrix all-negative (0 true positives), so recall/precision/
    F1 were degenerately 0 even though the model works (AUC 0.656,
    accuracy 0.89 both came through fine — only the threshold-dependent
    metrics were meaningless)."""
    return {
        "dataset_format": json.dumps({"csv": {"header": True}}),
        "analysis_type": "MODEL_QUALITY",
        "problem_type": "BinaryClassification",
        "probability_attribute": "probability",
        "probability_threshold_attribute": str(_frozen_threshold()),
        "ground_truth_attribute": "label",
    }


def run_model_quality_baseline(cfg, baseline_csv_uri: str, job_name: str) -> dict:
    role_arn = execution_role_arn()
    sm = sagemaker_client()
    output_s3 = f"s3://{cfg.bucket}/{cfg.prefixes['monitoring']}model-quality/baseline-output/{job_name}/"

    desc = run_processing_job(
        sm, job_name, MODEL_MONITOR_ANALYZER_IMAGE, role_arn,
        inputs=[s3_input("baseline_dataset_input", baseline_csv_uri.rsplit("/", 1)[0] + "/",
                          "/opt/ml/processing/input/baseline_dataset")],
        outputs=[s3_output("monitoring_output", output_s3, "/opt/ml/processing/output")],
        env={
            **model_quality_env(),
            "dataset_source": "/opt/ml/processing/input/baseline_dataset",
            "output_path": "/opt/ml/processing/output",
            "publish_cloudwatch_metrics": "Disabled",
        },
    )
    if desc["ProcessingJobStatus"] != "Completed":
        raise RuntimeError(f"model-quality baseline {job_name} failed: "
                            f"{desc.get('ExitMessage') or desc.get('FailureReason')}")
    return {
        "statistics_uri": f"{output_s3}statistics.json",
        "constraints_uri": f"{output_s3}constraints.json",
        "output_prefix": output_s3,
    }


def run_model_quality_monitoring_execution(cfg, production_csv_uri: str, baseline: dict, job_name: str) -> dict:
    role_arn = execution_role_arn()
    sm = sagemaker_client()
    output_s3 = f"s3://{cfg.bucket}/{cfg.prefixes['monitoring']}model-quality/monitoring-output/{job_name}/"

    desc = run_processing_job(
        sm, job_name, MODEL_MONITOR_ANALYZER_IMAGE, role_arn,
        inputs=[
            s3_input("dataset_input", production_csv_uri.rsplit("/", 1)[0] + "/",
                      "/opt/ml/processing/input/dataset"),
            s3_input("baseline_constraints_input", baseline["constraints_uri"].rsplit("/", 1)[0] + "/",
                      "/opt/ml/processing/baseline/constraints"),
        ],
        outputs=[s3_output("monitoring_output", output_s3, "/opt/ml/processing/output")],
        env={
            **model_quality_env(),
            "dataset_source": "/opt/ml/processing/input/dataset",
            "baseline_constraints": "/opt/ml/processing/baseline/constraints/constraints.json",
            "output_path": "/opt/ml/processing/output",
            "publish_cloudwatch_metrics": "Disabled",
        },
    )
    if desc["ProcessingJobStatus"] != "Completed":
        raise RuntimeError(f"model-quality monitoring {job_name} failed: "
                            f"{desc.get('ExitMessage') or desc.get('FailureReason')}")
    return {
        "constraint_violations_uri": f"{output_s3}constraint_violations.json",
        "statistics_uri": f"{output_s3}statistics.json",
        "output_prefix": output_s3,
    }


def main() -> None:
    import time

    cfg = load_config()

    val_scored = score_split("validation")
    baseline_csv_uri = upload_scored_csv(
        cfg, val_scored, f"{cfg.prefixes['monitoring']}model-quality/baseline-dataset/validation_scored.csv"
    )
    print("baseline (validation) scored dataset:", baseline_csv_uri, val_scored.shape)

    baseline_job_name = f"diabetes130-mq-baseline-{int(time.time())}"
    baseline = run_model_quality_baseline(cfg, baseline_csv_uri, baseline_job_name)
    print(json.dumps(baseline, indent=2))

    prod_scored = score_split("production")
    production_csv_uri = upload_scored_csv(
        cfg, prod_scored, f"{cfg.prefixes['monitoring']}model-quality/production-dataset/production_scored.csv"
    )
    print("production scored dataset:", production_csv_uri, prod_scored.shape)

    monitor_job_name = f"diabetes130-mq-monitor-{int(time.time())}"
    result = run_model_quality_monitoring_execution(cfg, production_csv_uri, baseline, monitor_job_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
