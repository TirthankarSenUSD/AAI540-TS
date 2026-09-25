"""CloudWatch dashboard for the ML system.

Neither the data-quality nor model-quality monitoring executions run here
are fired by a live SageMaker MonitoringSchedule (see the NOTE in
monitor_data_quality.py), so SageMaker won't auto-publish their results to
CloudWatch. This module reads the real JSON output each job already
produced and pushes a handful of representative metrics as custom
CloudWatch metrics, then assembles a dashboard combining those with the
infrastructure metrics SageMaker *does* auto-publish for Batch Transform
jobs (CPU/Memory/Disk utilization under /aws/sagemaker/TransformJobs).
"""
from __future__ import annotations

import json

import boto3

from config import load_config
from monitor_common import cloudwatch_client

CUSTOM_NAMESPACE = "Diabetes130/Monitoring"
DASHBOARD_NAME = "diabetes130-ml-system"


def push_data_quality_metrics(cfg, violations_s3_uri: str) -> dict:
    s3 = boto3.client("s3", region_name=cfg.region)
    bucket, key = violations_s3_uri.replace("s3://", "").split("/", 1)
    body = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    violations = body.get("violations", [])

    by_type: dict[str, int] = {}
    for v in violations:
        by_type[v["constraint_check_type"]] = by_type.get(v["constraint_check_type"], 0) + 1

    cw = cloudwatch_client()
    metric_data = [{"MetricName": "DataQualityViolationCount", "Value": float(len(violations)), "Unit": "Count"}]
    for check_type, count in by_type.items():
        metric_data.append({
            "MetricName": "DataQualityViolationCount",
            "Dimensions": [{"Name": "CheckType", "Value": check_type}],
            "Value": float(count),
            "Unit": "Count",
        })
    cw.put_metric_data(Namespace=CUSTOM_NAMESPACE, MetricData=metric_data)
    return {"total_violations": len(violations), "by_type": by_type}


def push_model_quality_metrics(cfg, statistics_s3_uri: str) -> dict:
    s3 = boto3.client("s3", region_name=cfg.region)
    bucket, key = statistics_s3_uri.replace("s3://", "").split("/", 1)
    body = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    metrics = body["binary_classification_metrics"]

    tracked = ["auc", "accuracy", "precision", "recall", "f1", "f0_5", "f2",
               "true_positive_rate", "false_positive_rate"]
    values = {name: metrics[name]["value"] for name in tracked if name in metrics}

    cw = cloudwatch_client()
    cw.put_metric_data(
        Namespace=CUSTOM_NAMESPACE,
        MetricData=[
            {"MetricName": f"ModelQuality_{name}", "Value": float(v), "Unit": "None"}
            for name, v in values.items()
        ],
    )
    return values


def build_dashboard_body(cfg, transform_job_host: str) -> dict:
    region = cfg.region
    widgets = [
        {
            "type": "text", "x": 0, "y": 0, "width": 24, "height": 2,
            "properties": {
                "markdown": (
                    "# Diabetes 130 Readmission — ML System Dashboard\n"
                    "Batch Transform on a daily cycle (no persistent endpoint). "
                    "Data/model quality monitored against training/validation baselines; "
                    "production split used as the reserved drift-monitoring dataset."
                )
            },
        },
        {
            "type": "metric", "x": 0, "y": 2, "width": 12, "height": 6,
            "properties": {
                "title": "Batch Transform — Resource Utilization (last real job)",
                "view": "timeSeries", "region": region,
                "metrics": [
                    ["/aws/sagemaker/TransformJobs", "CPUUtilization", "Host", transform_job_host],
                    ["/aws/sagemaker/TransformJobs", "MemoryUtilization", "Host", transform_job_host],
                    ["/aws/sagemaker/TransformJobs", "DiskUtilization", "Host", transform_job_host],
                ],
                "period": 60, "stat": "Average",
            },
        },
        {
            "type": "metric", "x": 12, "y": 2, "width": 12, "height": 6,
            "properties": {
                "title": "Data Quality — Constraint Violations",
                "view": "singleValue", "region": region,
                "metrics": [
                    [CUSTOM_NAMESPACE, "DataQualityViolationCount",
                     {"stat": "Maximum", "label": "Total violations"}],
                    [CUSTOM_NAMESPACE, "DataQualityViolationCount", "CheckType", "data_type_check",
                     {"stat": "Maximum", "label": "Data type"}],
                    [CUSTOM_NAMESPACE, "DataQualityViolationCount", "CheckType", "completeness_check",
                     {"stat": "Maximum", "label": "Completeness"}],
                    [CUSTOM_NAMESPACE, "DataQualityViolationCount", "CheckType", "categorical_values_check",
                     {"stat": "Maximum", "label": "Unseen categories"}],
                ],
                "period": 86400,
            },
        },
        {
            "type": "metric", "x": 0, "y": 8, "width": 24, "height": 6,
            "properties": {
                "title": "Model Quality — Production vs. Validation Baseline",
                "view": "singleValue", "region": region,
                "metrics": [
                    [CUSTOM_NAMESPACE, "ModelQuality_auc", {"stat": "Maximum", "label": "AUC"}],
                    [CUSTOM_NAMESPACE, "ModelQuality_accuracy", {"stat": "Maximum", "label": "Accuracy"}],
                    [CUSTOM_NAMESPACE, "ModelQuality_precision", {"stat": "Maximum", "label": "Precision"}],
                    [CUSTOM_NAMESPACE, "ModelQuality_recall", {"stat": "Maximum", "label": "Recall"}],
                    [CUSTOM_NAMESPACE, "ModelQuality_f1", {"stat": "Maximum", "label": "F1"}],
                ],
                "period": 86400,
            },
        },
        {
            "type": "metric", "x": 0, "y": 14, "width": 12, "height": 6,
            "properties": {
                "title": "Fairness — Disparate Impact (0.8-1.25 is the standard four-fifths-rule band)",
                "view": "singleValue", "region": region,
                "metrics": [
                    [CUSTOM_NAMESPACE, "BiasDisparateImpact", "Facet", "race=AfricanAmerican",
                     {"stat": "Maximum", "label": "race=AfricanAmerican"}],
                    [CUSTOM_NAMESPACE, "BiasDisparateImpact", "Facet", "race=Caucasian",
                     {"stat": "Maximum", "label": "race=Caucasian"}],
                    [CUSTOM_NAMESPACE, "BiasDisparateImpact", "Facet", "gender=Female",
                     {"stat": "Maximum", "label": "gender=Female"}],
                ],
                "period": 86400,
            },
        },
        {
            "type": "metric", "x": 12, "y": 14, "width": 12, "height": 6,
            "properties": {
                "title": "Fairness — Accuracy Difference (facet vs. everyone else)",
                "view": "singleValue", "region": region,
                "metrics": [
                    [CUSTOM_NAMESPACE, "BiasAccuracyDifference", "Facet", "race=AfricanAmerican",
                     {"stat": "Maximum", "label": "race=AfricanAmerican"}],
                    [CUSTOM_NAMESPACE, "BiasAccuracyDifference", "Facet", "race=Caucasian",
                     {"stat": "Maximum", "label": "race=Caucasian"}],
                    [CUSTOM_NAMESPACE, "BiasAccuracyDifference", "Facet", "gender=Female",
                     {"stat": "Maximum", "label": "gender=Female"}],
                ],
                "period": 86400,
            },
        },
    ]
    return {"widgets": widgets}


def create_dashboard(cfg, transform_job_host: str) -> None:
    cw = cloudwatch_client()
    body = build_dashboard_body(cfg, transform_job_host)
    cw.put_dashboard(DashboardName=DASHBOARD_NAME, DashboardBody=json.dumps(body))


def main() -> None:
    cfg = load_config()

    dq = push_data_quality_metrics(
        cfg, "s3://771119245550-diabetes130/monitoring/data-quality/monitoring-output/"
             "diabetes130-dq-monitor-1790325047/constraint_violations.json",
    )
    print("data quality metrics pushed:", json.dumps(dq, indent=2))

    create_dashboard(cfg, "diabetes130-batch-transform-v2-1790229539/i-0f13745d7879fde8e")
    print(f"dashboard created: {DASHBOARD_NAME}")


if __name__ == "__main__":
    main()
