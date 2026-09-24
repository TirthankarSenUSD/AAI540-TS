"""Task 1 — benchmark models: the trivial floor and the heuristic baseline.

The point is a defensible floor to measure against, not a good model.

1a. Majority-class floor — predict negative for everything.
1b. Heuristic (`number_inpatient >= 1`) plus a 2-feature logistic
    regression on `number_inpatient` / `number_emergency`, which is also
    trained through SageMaker script mode (SKLearn container) so the
    train-job / model-artifact / registry path is exercised once on
    something trivial before Task 2 does it with a model that can fail.
    See `models/benchmark_sklearn/train.py` for the SageMaker entry point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

HEURISTIC_FEATURES = ["number_inpatient", "number_emergency"]


def majority_class_metrics(y: pd.Series) -> dict:
    """Task 1a — predict the majority class (negative) for everything."""
    y_pred = np.zeros(len(y), dtype=int)
    base_rate = y.mean()
    return {
        "accuracy": float((y_pred == y).mean()),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "pr_auc": float(base_rate),  # a constant-score predictor's PR-AUC is the base rate
        "base_rate": float(base_rate),
    }


def heuristic_predict(df: pd.DataFrame) -> np.ndarray:
    """Task 1b — predict positive if number_inpatient >= 1."""
    return (df["number_inpatient"] >= 1).astype(int).to_numpy()


def heuristic_metrics(df: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = heuristic_predict(df)
    return {
        "accuracy": float((y_pred == y.to_numpy()).mean()),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_pred)),
    }


def fit_logistic_benchmark(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Continuous-score benchmark: logistic regression on the two prior-
    utilization counts, so the benchmark can be compared on ranking metrics
    (PR-AUC, precision@k) and not only at a fixed threshold."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train[HEURISTIC_FEATURES], y_train)
    return model


def logistic_benchmark_metrics(model: LogisticRegression, X: pd.DataFrame, y: pd.Series) -> dict:
    scores = model.predict_proba(X[HEURISTIC_FEATURES])[:, 1]
    y_pred = (scores >= 0.5).astype(int)
    return {
        "accuracy": float((y_pred == y.to_numpy()).mean()),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }



# ---------------------------------------------------------------------------
# Task 1b — deploy through SageMaker (script mode, SKLearn container).
# Written and locally smoke-tested (see models/benchmark_sklearn/train.py);
# not yet invoked against AWS.
# ---------------------------------------------------------------------------

def prepare_benchmark_channels(cfg, local_dir: str) -> dict:
    """Write the 2-feature CSVs the SKLearn entry point expects and upload
    them to S3, returning the channel URIs."""
    from pathlib import Path

    import boto3

    from model_data import load_split
    from benchmark import HEURISTIC_FEATURES

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=cfg.region)

    channels = {}
    for split_name, channel_name in [("train", "train"), ("validation", "validation")]:
        df = load_split(split_name)
        cols = HEURISTIC_FEATURES + ["readmit_30"]
        csv_path = local_path / f"{channel_name}.csv"
        df[cols].to_csv(csv_path, index=False)

        key = f"models/benchmark-sklearn/{channel_name}/{channel_name}.csv"
        s3.upload_file(str(csv_path), cfg.bucket, key)
        channels[channel_name] = f"s3://{cfg.bucket}/{key}"
    return channels


def launch_benchmark_training_job(job_name: str | None = None) -> str:
    """Launch the real SageMaker SKLearn script-mode training job for Task
    1b. Not called by main() — invoke explicitly once approved."""
    import time
    from pathlib import Path

    import boto3

    from aws_jobs import execution_role_arn, framework_image_uri, script_mode_hyperparameters, \
        upload_source_dir, wait_for_training_job
    from config import load_config

    cfg = load_config()
    job_name = job_name or f"diabetes130-benchmark-{int(time.time())}"

    channels = prepare_benchmark_channels(cfg, "/tmp/benchmark_channels")
    source_dir_uri = upload_source_dir(
        str(Path(__file__).resolve().parent.parent / "models" / "benchmark_sklearn"),
        cfg, f"models/benchmark-sklearn/code-{job_name}",
    )
    image_uri = framework_image_uri("sklearn", cfg.region, "1.2-1", "ml.m5.large")
    role_arn = execution_role_arn()

    sm = boto3.client("sagemaker", region_name=cfg.region)
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={"TrainingImage": image_uri, "TrainingInputMode": "File"},
        RoleArn=role_arn,
        HyperParameters=script_mode_hyperparameters("train.py", source_dir_uri),
        InputDataConfig=[
            {"ChannelName": name, "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": uri.rsplit("/", 1)[0], "S3DataDistributionType": "FullyReplicated",
            }}, "ContentType": "text/csv"}
            for name, uri in channels.items()
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{cfg.bucket}/models/benchmark-sklearn/output/"},
        ResourceConfig={"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 5},
        StoppingCondition={"MaxRuntimeInSeconds": 900},
    )
    desc = wait_for_training_job(sm, job_name)
    if desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"training job {job_name} did not complete: {desc.get('FailureReason')}")
    return job_name


def register_benchmark_model_package(training_job_name: str) -> str:
    """After the training job completes, register it as a model package
    with PendingManualApproval and the recorded validation metrics attached."""
    import boto3

    from aws_jobs import execution_role_arn, framework_image_uri
    from config import load_config
    from deploy import ensure_model_package_group

    cfg = load_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    s3 = boto3.client("s3", region_name=cfg.region)

    desc = sm.describe_training_job(TrainingJobName=training_job_name)
    model_data_url = desc["ModelArtifacts"]["S3ModelArtifacts"]

    # metrics.json was written by models/benchmark_sklearn/train.py next to model.joblib
    metrics_key = model_data_url.split(f"{cfg.bucket}/", 1)[1].replace("model.tar.gz", "metrics.json")
    try:
        metrics_body = s3.get_object(Bucket=cfg.bucket, Key=metrics_key)["Body"].read()
        metrics_s3_uri = f"s3://{cfg.bucket}/{metrics_key}"
    except Exception:
        metrics_s3_uri = None

    group_name = "diabetes130-benchmark-models"
    ensure_model_package_group(sm, group_name, "Task 1b heuristic/logistic benchmark models")

    image_uri = framework_image_uri("sklearn", cfg.region, "1.2-1", "ml.m5.large")
    kwargs = dict(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
        },
    )
    if metrics_s3_uri:
        kwargs["ModelMetrics"] = {
            "ModelQuality": {"Statistics": {"ContentType": "application/json", "S3Uri": metrics_s3_uri}},
        }
    response = sm.create_model_package(**kwargs)
    return response["ModelPackageArn"]


def main() -> None:
    import json
    from pathlib import Path

    from model_data import features_and_target, load_split

    train_df = load_split("train")
    val_df = load_split("validation")
    X_train, y_train = features_and_target(train_df)
    X_val, y_val = features_and_target(val_df)

    results = {
        "majority_class_floor": {
            "train": majority_class_metrics(y_train),
            "validation": majority_class_metrics(y_val),
        },
        "heuristic_rule": {
            "train": heuristic_metrics(train_df, y_train),
            "validation": heuristic_metrics(val_df, y_val),
        },
    }

    logreg = fit_logistic_benchmark(X_train, y_train)
    results["logistic_benchmark"] = {
        "train": logistic_benchmark_metrics(logreg, X_train, y_train),
        "validation": logistic_benchmark_metrics(logreg, X_val, y_val),
        "coefficients": dict(zip(HEURISTIC_FEATURES, logreg.coef_[0].tolist())),
        "intercept": float(logreg.intercept_[0]),
    }

    print(json.dumps(results, indent=2))

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "task1_benchmarks.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
