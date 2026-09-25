"""Model and data reports -- SageMaker Clarify.

NOT FUNCTIONAL in this AWS account -- kept as a documented record of what
was attempted. Launching a real Clarify processing job here fails
immediately with:

    ValidationException: SageMaker Clarify processing is in maintenance
    mode and is not available to new customers. Existing customers are
    unaffected.

This is an account-level platform restriction, not a bug in this code or
this project's setup. `src/fairness_report.py` +
`models/fairness_report/generate_report.py` produce the same substance
(pre/post-training bias metrics on race and gender, global feature
importance via XGBoost's native SHAP contributions) as a real SageMaker
Processing Job, without depending on the blocked Clarify container --
that is the module actually used; see RESULTS.md's monitoring section.

This module would otherwise use "bring your own predictions" mode -- the
scored production split already has a `prediction` column from the real
deployed artifact (monitor_model_quality.score_split), so Clarify
wouldn't need to invoke a live endpoint to compute post-training bias
metrics -- with facet race == "AfricanAmerican" vs. every other race
value, the largest minority group in this dataset.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import boto3
import pandas as pd

from aws_jobs import execution_role_arn
from config import load_config
from monitor_common import run_processing_job, s3_input, s3_output, sagemaker_client

CLARIFY_IMAGE = "205585389593.dkr.ecr.us-east-1.amazonaws.com/sagemaker-clarify-processing:1.0"


def prepare_bias_dataset(cfg) -> str:
    """race, true label, and the deployed model's prediction for every
    production-split row -- everything the bias analysis needs, all
    already computed."""
    import sys

    MODELS_XGBOOST_DIR = Path(__file__).resolve().parent.parent / "models" / "xgboost"
    sys.path.insert(0, str(MODELS_XGBOOST_DIR))

    from model_data import features_and_target, load_split
    from monitor_model_quality import score_split

    prod_df = load_split("production")
    X_prod, _ = features_and_target(prod_df)

    scored = score_split("production")
    combined = pd.DataFrame({
        "race": X_prod["race"].astype(object).where(X_prod["race"].notna(), "Missing").reset_index(drop=True),
        "label": scored["label"].reset_index(drop=True),
        "prediction": scored["prediction"].reset_index(drop=True),
    })

    local_path = Path("/tmp/clarify_bias_dataset.csv")
    combined.to_csv(local_path, index=False)

    s3 = boto3.client("s3", region_name=cfg.region)
    key = f"{cfg.prefixes['monitoring']}clarify/bias-dataset/production_bias.csv"
    s3.upload_file(str(local_path), cfg.bucket, key)
    return f"s3://{cfg.bucket}/{key}"


def build_bias_analysis_config() -> dict:
    return {
        "version": "1.0",
        "dataset_type": "text/csv",
        "dataset_uri": "/opt/ml/processing/input/data/production_bias.csv",
        "headers": ["race", "label", "prediction"],
        "label": "label",
        "label_values_or_threshold": [1],
        "facet": [{"name_or_index": "race", "value_or_threshold": ["AfricanAmerican"]}],
        "predicted_label": "prediction",
        "methods": {
            "pre_training_bias": {"methods": "all"},
            "post_training_bias": {"methods": "all"},
            "report": {"name": "diabetes130_bias_report", "title": "Diabetes 130 Readmission -- Race Bias Report"},
        },
    }


def run_bias_report(cfg, dataset_uri: str, job_name: str) -> dict:
    role_arn = execution_role_arn()
    sm = sagemaker_client()

    config_local = Path("/tmp/analysis_config.json")
    config_local.write_text(json.dumps(build_bias_analysis_config(), indent=2))

    s3 = boto3.client("s3", region_name=cfg.region)
    config_key = f"{cfg.prefixes['monitoring']}clarify/bias-config/{job_name}/analysis_config.json"
    s3.upload_file(str(config_local), cfg.bucket, config_key)
    config_uri = f"s3://{cfg.bucket}/{config_key}"

    output_s3 = f"s3://{cfg.bucket}/{cfg.prefixes['monitoring']}clarify/bias-output/{job_name}/"

    desc = run_processing_job(
        sm, job_name, CLARIFY_IMAGE, role_arn,
        inputs=[
            s3_input("analysis_config", config_uri.rsplit("/", 1)[0] + "/", "/opt/ml/processing/input/config"),
            s3_input("dataset", dataset_uri.rsplit("/", 1)[0] + "/", "/opt/ml/processing/input/data"),
        ],
        outputs=[s3_output("analysis_result", output_s3, "/opt/ml/processing/output")],
        env={},
        max_runtime_seconds=1800,
    )
    if desc["ProcessingJobStatus"] != "Completed":
        raise RuntimeError(f"bias report {job_name} failed: {desc.get('ExitMessage') or desc.get('FailureReason')}")

    return {
        "analysis_json_uri": f"{output_s3}analysis.json",
        "report_html_uri": f"{output_s3}report.html",
        "output_prefix": output_s3,
    }


def main() -> None:
    cfg = load_config()

    dataset_uri = prepare_bias_dataset(cfg)
    print("bias dataset:", dataset_uri)

    job_name = f"diabetes130-clarify-bias-{int(time.time())}"
    result = run_bias_report(cfg, dataset_uri, job_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
