"""Data monitor — SageMaker Model Monitor, Data Quality type.

Baseline: statistics + constraints computed from the training data (what
"normal" looks like). Monitoring: the production split — reserved back in
Week 3 (decision #5) exactly for "monitoring and drift work in a later
module" — scored through a real, data-captured Batch Transform job and
compared against the baseline. Violations mean the incoming feature
distribution has drifted from what the model was trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3

from aws_jobs import execution_role_arn
from config import load_config
from monitor_common import (
    MODEL_MONITOR_ANALYZER_IMAGE,
    run_processing_job,
    s3_input,
    s3_output,
    sagemaker_client,
)

JOB_DEFINITION_NAME = "diabetes130-data-quality-job-definition"
SCHEDULE_NAME = "diabetes130-data-quality-schedule"


def prepare_baseline_csv(cfg) -> tuple[str, list[str]]:
    """Train split, feature columns only, CSV with header — this defines
    what "normal" data looks like. Column order just needs to be
    consistent between this baseline and the production batch compared
    against it later (data-quality statistics are matched by column name,
    not position) — deriving it fresh from the split avoids depending on
    a specific deployed model artifact being present locally."""
    from model_data import features_and_target, load_split

    train_df = load_split("train")
    X_train, _ = features_and_target(train_df)
    feature_order = list(X_train.columns)

    local_path = Path("/tmp/dq_baseline.csv")
    X_train.to_csv(local_path, index=False)

    s3 = boto3.client("s3", region_name=cfg.region)
    key = f"{cfg.prefixes['monitoring']}data-quality/baseline-dataset/baseline.csv"
    s3.upload_file(str(local_path), cfg.bucket, key)
    return f"s3://{cfg.bucket}/{key}", feature_order


def run_baseline_job(cfg, baseline_csv_uri: str, job_name: str) -> dict:
    role_arn = execution_role_arn()
    sm = sagemaker_client()

    output_s3 = f"s3://{cfg.bucket}/{cfg.prefixes['monitoring']}data-quality/baseline-output/{job_name}/"

    desc = run_processing_job(
        sm, job_name, MODEL_MONITOR_ANALYZER_IMAGE, role_arn,
        inputs=[s3_input("baseline_dataset_input", baseline_csv_uri.rsplit("/", 1)[0] + "/",
                          "/opt/ml/processing/input/baseline_dataset")],
        outputs=[s3_output("monitoring_output", output_s3, "/opt/ml/processing/output")],
        env={
            "dataset_format": json.dumps({"csv": {"header": True, "output_columns_position": "START"}}),
            "dataset_source": "/opt/ml/processing/input/baseline_dataset",
            "output_path": "/opt/ml/processing/output",
            "publish_cloudwatch_metrics": "Disabled",
        },
    )
    if desc["ProcessingJobStatus"] != "Completed":
        raise RuntimeError(f"baseline job {job_name} failed: {desc.get('ExitMessage') or desc.get('FailureReason')}")

    return {
        "statistics_uri": f"{output_s3}statistics.json",
        "constraints_uri": f"{output_s3}constraints.json",
        "output_prefix": output_s3,
    }


def prepare_production_csv(cfg, feature_order: list[str]) -> str:
    """Production split feature columns, same schema/order as the
    baseline -- this is "today's incoming batch" for drift comparison."""
    from model_data import features_and_target, load_split

    prod_df = load_split("production")
    X_prod, _ = features_and_target(prod_df)
    X_prod = X_prod[feature_order]

    local_path = Path("/tmp/dq_production.csv")
    X_prod.to_csv(local_path, index=False)

    s3 = boto3.client("s3", region_name=cfg.region)
    key = f"{cfg.prefixes['monitoring']}data-quality/production-dataset/production.csv"
    s3.upload_file(str(local_path), cfg.bucket, key)
    return f"s3://{cfg.bucket}/{key}"


def run_monitoring_execution(cfg, production_csv_uri: str, baseline: dict, job_name: str) -> dict:
    """Compare the production split against the baseline constraints --
    the actual drift check.

    NOTE: publish_cloudwatch_metrics=Enabled only works when the analyzer
    container is invoked by a real MonitoringSchedule (it needs schedule
    context to timestamp metrics) -- a standalone, manually-launched
    Processing Job like this one fails outright with "CloudWatch
    publishing is available only for jobs from MonitoringSchedules."
    Disabled here; the dashboard instead reads violation counts out of
    constraint_violations.json and pushes them as custom metrics itself
    (see monitor_dashboard.py), which also works for one-off runs.
    """
    role_arn = execution_role_arn()
    sm = sagemaker_client()

    output_s3 = f"s3://{cfg.bucket}/{cfg.prefixes['monitoring']}data-quality/monitoring-output/{job_name}/"

    desc = run_processing_job(
        sm, job_name, MODEL_MONITOR_ANALYZER_IMAGE, role_arn,
        inputs=[
            s3_input("dataset_input", production_csv_uri.rsplit("/", 1)[0] + "/",
                      "/opt/ml/processing/input/dataset"),
            s3_input("baseline_constraints_input", baseline["constraints_uri"].rsplit("/", 1)[0] + "/",
                      "/opt/ml/processing/baseline/constraints"),
            s3_input("baseline_statistics_input", baseline["statistics_uri"].rsplit("/", 1)[0] + "/",
                      "/opt/ml/processing/baseline/stats"),
        ],
        outputs=[s3_output("monitoring_output", output_s3, "/opt/ml/processing/output")],
        env={
            "dataset_format": json.dumps({"csv": {"header": True, "output_columns_position": "START"}}),
            "dataset_source": "/opt/ml/processing/input/dataset",
            "baseline_constraints": "/opt/ml/processing/baseline/constraints/constraints.json",
            "baseline_statistics": "/opt/ml/processing/baseline/stats/statistics.json",
            "output_path": "/opt/ml/processing/output",
            "publish_cloudwatch_metrics": "Disabled",
        },
    )
    if desc["ProcessingJobStatus"] != "Completed":
        raise RuntimeError(f"monitoring job {job_name} failed: {desc.get('ExitMessage') or desc.get('FailureReason')}")

    return {
        "constraint_violations_uri": f"{output_s3}constraint_violations.json",
        "statistics_uri": f"{output_s3}statistics.json",
        "output_prefix": output_s3,
    }


def main() -> None:
    import time

    cfg = load_config()

    baseline_csv_uri, feature_order = prepare_baseline_csv(cfg)
    print("baseline dataset:", baseline_csv_uri)
    print(f"{len(feature_order)} feature columns")

    baseline_job_name = f"diabetes130-dq-baseline-{int(time.time())}"
    baseline = run_baseline_job(cfg, baseline_csv_uri, baseline_job_name)
    print(json.dumps(baseline, indent=2))

    production_csv_uri = prepare_production_csv(cfg, feature_order)
    print("production dataset:", production_csv_uri)

    monitor_job_name = f"diabetes130-dq-monitor-{int(time.time())}"
    result = run_monitoring_execution(cfg, production_csv_uri, baseline, monitor_job_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
