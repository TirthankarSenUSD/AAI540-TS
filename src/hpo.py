"""Task 2 — hyperparameter tuning for the real XGBoost model.

SageMaker Automatic Model Tuning, Bayesian strategy, objective metric
`validation:aucpr` — not `auc`, PR-AUC is what matters at ~11% prevalence.
20-30 trials: enough signal on a dataset with a hard performance ceiling
(~0.67 ROC-AUC) without tuning into noise or burning budget on 200 trials.

Written and ready; not yet invoked against AWS — launch explicitly once
approved (`python src/hpo.py`), since this is the one step in the brief
that costs real, non-trivial compute (multiple parallel training jobs).
"""
from __future__ import annotations

import time
from pathlib import Path

import boto3

from aws_jobs import (
    debugger_rule_configurations,
    execution_role_arn,
    framework_image_uri,
    script_mode_hyperparameters,
    upload_source_dir,
    wait_for_training_job,
    wait_for_tuning_job,
)
from config import load_config

MODELS_XGBOOST_DIR = Path(__file__).resolve().parent.parent / "models" / "xgboost"

HYPERPARAMETER_RANGES = {
    "ParameterRanges": {
        "IntegerParameterRanges": [
            {"Name": "max_depth", "MinValue": "3", "MaxValue": "8"},
            {"Name": "num_round", "MinValue": "100", "MaxValue": "1000"},
        ],
        "ContinuousParameterRanges": [
            {"Name": "eta", "MinValue": "0.01", "MaxValue": "0.3", "ScalingType": "Logarithmic"},
            {"Name": "subsample", "MinValue": "0.6", "MaxValue": "1.0", "ScalingType": "Linear"},
            {"Name": "colsample_bytree", "MinValue": "0.6", "MaxValue": "1.0", "ScalingType": "Linear"},
            {"Name": "min_child_weight", "MinValue": "1", "MaxValue": "10", "ScalingType": "Linear"},
            # SageMaker's HPO service validates tunable hyperparameter names
            # for this image against the built-in XGBoost algorithm's fixed
            # list, even in script mode — it must be "lambda", not xgboost's
            # sklearn-API name "reg_lambda". train.py accepts --lambda with
            # dest=reg_lambda to bridge the two.
            {"Name": "lambda", "MinValue": "0", "MaxValue": "10", "ScalingType": "Linear"},
        ],
    },
}

MAX_TRIALS = 25
MAX_PARALLEL = 4


def export_split_to_csv(cfg, split_name: str) -> str:
    """The SageMaker XGBoost training toolkit doesn't auto-install a
    source-dir requirements.txt, so the container never gets pyarrow —
    Parquet reads fail inside train.py. Export a CSV copy of the split
    once and point training jobs at that instead of fighting the
    container's default package set."""
    import awswrangler as wr
    wr.engine.set("python")

    df = wr.s3.read_parquet(cfg.s3_uri("features", f"split={split_name}/"))
    dest_prefix = cfg.s3_uri("features", f"csv/split={split_name}/")
    wr.s3.to_csv(df, dest_prefix, index=False, dataset=True, mode="overwrite")
    return dest_prefix


def launch_xgboost_hpo(tuning_job_name: str | None = None,
                        max_trials: int = MAX_TRIALS, max_parallel: int = MAX_PARALLEL) -> str:
    cfg = load_config()
    tuning_job_name = tuning_job_name or f"diabetes130-xgb-hpo-{int(time.time())}"
    role_arn = execution_role_arn()

    source_dir_uri = upload_source_dir(str(MODELS_XGBOOST_DIR), cfg, f"models/xgboost/code-{tuning_job_name}")
    image_uri = framework_image_uri("xgboost", cfg.region, "1.7-1", "ml.m5.xlarge")

    train_csv_uri = export_split_to_csv(cfg, "train")
    val_csv_uri = export_split_to_csv(cfg, "validation")

    static_hp = script_mode_hyperparameters("train.py", source_dir_uri, extra={
        "early_stopping_rounds": 20,
        "scale_pos_weight": 1.0,  # per decision #2: no resampling; revisit via this knob if needed, not class rebalancing
    })

    sm = boto3.client("sagemaker", region_name=cfg.region)
    sm.create_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name,
        HyperParameterTuningJobConfig={
            "Strategy": "Bayesian",
            "HyperParameterTuningJobObjective": {"Type": "Maximize", "MetricName": "validation:aucpr"},
            "ResourceLimits": {"MaxNumberOfTrainingJobs": max_trials, "MaxParallelTrainingJobs": max_parallel},
            **HYPERPARAMETER_RANGES,
        },
        TrainingJobDefinition={
            "StaticHyperParameters": static_hp,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
                "MetricDefinitions": [
                    {"Name": "validation:aucpr", "Regex": r"validation:aucpr=([0-9\.]+)"},
                ],
            },
            "RoleArn": role_arn,
            "InputDataConfig": [
                {"ChannelName": "train", "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix", "S3Uri": train_csv_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }}, "ContentType": "text/csv"},
                {"ChannelName": "validation", "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix", "S3Uri": val_csv_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }}, "ContentType": "text/csv"},
            ],
            "OutputDataConfig": {"S3OutputPath": f"s3://{cfg.bucket}/models/xgboost/hpo-output/"},
            "ResourceConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 10},
            "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
            # NOTE: DebugHookConfig/DebugRuleConfigurations are not valid
            # fields inside a HPO TrainingJobDefinition (boto3 rejects them
            # with ParamValidationError) — Debugger is only attachable to a
            # standalone create_training_job. See launch_final_training_job
            # below: after HPO picks the best hyperparameters, one more
            # debugged training run reproduces the model for deployment.
        },
    )
    return tuning_job_name


def best_training_job(tuning_job_name: str) -> dict:
    cfg = load_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    desc = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)
    return desc["BestTrainingJob"]


def launch_final_training_job(hyperparameters: dict, source_dir_uri: str,
                               job_name: str | None = None) -> str:
    """One more training run at the HPO-selected hyperparameters, this time
    with SageMaker Debugger's Overfit / LossNotDecreasing / ClassImbalance
    rules attached — this becomes the artifact used for calibration and
    deployment."""
    cfg = load_config()
    job_name = job_name or f"diabetes130-xgb-final-{int(time.time())}"
    role_arn = execution_role_arn()
    image_uri = framework_image_uri("xgboost", cfg.region, "1.7-1", "ml.m5.xlarge")

    static_hp = script_mode_hyperparameters("train.py", source_dir_uri, extra={
        "early_stopping_rounds": 20,
        "scale_pos_weight": 1.0,
        **hyperparameters,
    })

    sm = boto3.client("sagemaker", region_name=cfg.region)
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri, "TrainingInputMode": "File",
            # NOTE: unlike inside a HyperParameterTuningJob's
            # TrainingJobDefinition (where this same field is required and
            # works), a direct create_training_job call for this XGBoost
            # image rejects custom MetricDefinitions — "can't override the
            # metric definitions for Amazon SageMaker algorithms". The
            # validation:aucpr line is still printed to stdout/CloudWatch,
            # just not auto-scraped into a queryable metric for this job.
        },
        RoleArn=role_arn,
        HyperParameters=static_hp,
        InputDataConfig=[
            {"ChannelName": "train", "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": cfg.s3_uri("features", "csv/split=train/"),
                "S3DataDistributionType": "FullyReplicated",
            }}, "ContentType": "text/csv"},
            {"ChannelName": "validation", "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": cfg.s3_uri("features", "csv/split=validation/"),
                "S3DataDistributionType": "FullyReplicated",
            }}, "ContentType": "text/csv"},
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{cfg.bucket}/models/xgboost/final-output/"},
        ResourceConfig={"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
        DebugHookConfig={"S3OutputPath": f"s3://{cfg.bucket}/models/xgboost/debug-output/{job_name}/"},
        DebugRuleConfigurations=debugger_rule_configurations(),
    )
    return job_name


def main() -> None:
    job_name = launch_xgboost_hpo()
    print(f"launched HPO job: {job_name}")

    cfg = load_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    desc = wait_for_tuning_job(sm, job_name)
    if desc["HyperParameterTuningJobStatus"] != "Completed":
        raise RuntimeError(f"HPO job {job_name} did not complete successfully")

    best = desc["BestTrainingJob"]
    print(f"best training job: {best['TrainingJobName']}")
    print(f"best validation:aucpr: {best['FinalHyperParameterTuningJobObjectiveMetric']['Value']}")
    print(f"best hyperparameters: {best['TunedHyperParameters']}")

    best_job_desc = sm.describe_training_job(TrainingJobName=best["TrainingJobName"])
    source_dir_uri = best_job_desc["HyperParameters"]["sagemaker_submit_directory"].strip('"')

    tuned_hp = {k: v for k, v in best["TunedHyperParameters"].items()}
    final_job = launch_final_training_job(tuned_hp, source_dir_uri)
    print(f"launched final debugged training job: {final_job}")
    final_desc = wait_for_training_job(sm, final_job)
    if final_desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"final training job {final_job} did not complete: {final_desc.get('FailureReason')}")

    print(f"final model artifacts: {final_desc['ModelArtifacts']['S3ModelArtifacts']}")
    return final_job


if __name__ == "__main__":
    main()
