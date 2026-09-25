"""CI/CD DAG -- a real SageMaker Pipeline: Train -> Evaluate -> quality-gate
Condition -> RegisterModel (if the gate passes) / Fail (if it doesn't).

This is a *separate* artifact from the hand-built, hand-calibrated,
human-approved production model already deployed via src/deploy.py --
the point here is demonstrating pipeline orchestration (a real DAG,
runnable in both a successful and a failed state), not replacing that
model. It registers into its own group, `diabetes130-pipeline-models`,
never the production `diabetes130-xgboost-models` group.

Reuses the best hyperparameters found by Week 4's real HPO run (fixed,
not re-tuned -- a CI/CD pipeline meant to be re-run on demand shouldn't
re-run a 25-trial search every time) and the CSV channels already
exported to S3 for that HPO run.

The quality gate is a Pipeline Parameter (AucThreshold) so the exact same
pipeline definition can be executed twice to produce both states asked
for: once with an achievable threshold (success, reaches RegisterModel)
and once with an unreachable one (failure, stops at the Fail step) --
this demonstrates the gate actually gating, rather than faking a failure
by breaking something unrelated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import boto3

from aws_jobs import execution_role_arn, framework_image_uri, script_mode_hyperparameters, upload_source_dir
from config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_XGBOOST_DIR = REPO_ROOT / "models" / "xgboost"

PIPELINE_NAME = "diabetes130-cicd-pipeline"
MODEL_PACKAGE_GROUP = "diabetes130-pipeline-models"

# Best hyperparameters from Week 4's real 25-trial HPO run
# (diabetes130-xgb-hpo-1790226750, best trial -021, validation PR-AUC 0.2125).
BEST_HYPERPARAMETERS = {
    "max_depth": "8",
    "eta": "0.010291182166748574",
    "lambda": "0.41615545335638604",
    "subsample": "0.8180599077204804",
    "colsample_bytree": "0.6207786869726521",
    "min_child_weight": "8.047139184883324",
    "num_round": "468",
}


def build_pipeline_definition(cfg, image_uri: str, role_arn: str, source_dir_uri: str,
                               auc_threshold: float = 0.6) -> dict:
    train_csv_uri = cfg.s3_uri("features", "csv/split=train/")
    val_csv_uri = cfg.s3_uri("features", "csv/split=validation/")
    test_csv_uri = cfg.s3_uri("features", "csv/split=test/")
    eval_output_uri = f"s3://{cfg.bucket}/{cfg.prefixes['monitoring']}pipeline/eval-output/"
    train_output_uri = f"s3://{cfg.bucket}/{cfg.prefixes['monitoring']}pipeline/train-output/"

    static_hp = script_mode_hyperparameters(
        "train.py", source_dir_uri,
        extra={"early_stopping_rounds": 20, "scale_pos_weight": 1.0, **BEST_HYPERPARAMETERS},
    )

    train_step = {
        "Name": "TrainXGBoost",
        "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {"TrainingImage": image_uri, "TrainingInputMode": "File"},
            "RoleArn": role_arn,
            "HyperParameters": static_hp,
            "InputDataConfig": [
                {"ChannelName": "train", "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix", "S3Uri": train_csv_uri, "S3DataDistributionType": "FullyReplicated",
                }}, "ContentType": "text/csv"},
                {"ChannelName": "validation", "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix", "S3Uri": val_csv_uri, "S3DataDistributionType": "FullyReplicated",
                }}, "ContentType": "text/csv"},
            ],
            "OutputDataConfig": {"S3OutputPath": train_output_uri},
            "ResourceConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 10},
            "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
        },
    }

    evaluate_step = {
        "Name": "EvaluateModel",
        "Type": "Processing",
        "Arguments": {
            "ProcessingResources": {
                "ClusterConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 20},
            },
            "AppSpecification": {
                "ImageUri": image_uri,
                "ContainerEntrypoint": ["/bin/bash", "-c"],
                "ContainerArguments": [
                    "mkdir -p /opt/ml/processing/input/code "
                    "&& tar -xzf /opt/ml/processing/input/archive/sourcedir.tar.gz -C /opt/ml/processing/input/code "
                    "&& python3 /opt/ml/processing/input/code/evaluate_step.py",
                ],
            },
            # Environment values must be plain strings -- a `{"Get":
            # "Parameters.X"}` reference here is rejected ("Cannot assign
            # property reference ... to argument of type [String]"), even
            # though the identical mechanism works for ModelDataUrl below.
            # Baking the threshold in at definition-build time instead;
            # see start_success_and_failure_executions() for how the two
            # demo runs get two different pipeline definitions.
            "Environment": {"AUC_THRESHOLD": str(auc_threshold)},
            "RoleArn": role_arn,
            "ProcessingInputs": [
                {"InputName": "model", "S3Input": {
                    "S3Uri": {"Get": "Steps.TrainXGBoost.ModelArtifacts.S3ModelArtifacts"},
                    "LocalPath": "/opt/ml/processing/input/model",
                    "S3DataType": "S3Prefix", "S3InputMode": "File",
                }},
                {"InputName": "test_data", "S3Input": {
                    "S3Uri": test_csv_uri, "LocalPath": "/opt/ml/processing/input/test",
                    "S3DataType": "S3Prefix", "S3InputMode": "File",
                }},
                {"InputName": "code", "S3Input": {
                    "S3Uri": source_dir_uri.rsplit("/", 1)[0] + "/",
                    "LocalPath": "/opt/ml/processing/input/archive",
                    "S3DataType": "S3Prefix", "S3InputMode": "File",
                }},
            ],
            "ProcessingOutputConfig": {"Outputs": [
                {"OutputName": "evaluation", "S3Output": {
                    "S3Uri": eval_output_uri, "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob",
                }},
            ]},
        },
        "PropertyFiles": [
            {"PropertyFileName": "EvalReport", "OutputName": "evaluation", "FilePath": "evaluation.json"},
        ],
    }

    register_step = {
        "Name": "RegisterModel",
        "Type": "RegisterModel",
        "DependsOn": ["EvaluateModel"],
        "Arguments": {
            "ModelPackageGroupName": MODEL_PACKAGE_GROUP,
            "ModelApprovalStatus": "PendingManualApproval",
            "InferenceSpecification": {
                "Containers": [{
                    "Image": image_uri,
                    "ModelDataUrl": {"Get": "Steps.TrainXGBoost.ModelArtifacts.S3ModelArtifacts"},
                }],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
                "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge"],
            },
        },
    }

    # Linear DAG: Train -> Evaluate -> Register. The quality gate is
    # enforced inside EvaluateModel's own script (exits non-zero on a
    # miss) rather than via a native Condition step -- see
    # evaluate_step.py's docstring for why. A failing gate makes
    # EvaluateModel itself fail, so RegisterModel (which DependsOn it)
    # never runs and the whole execution shows Failed.
    return {
        "Version": "2020-12-01",
        "Steps": [train_step, evaluate_step, register_step],
    }


def create_or_update_pipeline(cfg, auc_threshold: float = 0.6, source_dir_uri: str | None = None) -> str:
    """The quality-gate threshold is baked into the pipeline definition at
    build time (see build_pipeline_definition's Environment note), so
    producing the two demo states (success/failure) means updating the
    pipeline definition with a different threshold before each run, not
    passing a runtime execution parameter."""
    role_arn = execution_role_arn()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    image_uri = framework_image_uri("xgboost", cfg.region, "1.7-1", "ml.m5.xlarge")

    if source_dir_uri is None:
        source_dir_uri = upload_source_dir(
            str(MODELS_XGBOOST_DIR), cfg, f"models/xgboost/pipeline-code-{int(time.time())}"
        )

    definition = build_pipeline_definition(cfg, image_uri, role_arn, source_dir_uri, auc_threshold)
    definition_json = json.dumps(definition)

    try:
        sm.describe_pipeline(PipelineName=PIPELINE_NAME)
        sm.update_pipeline(
            PipelineName=PIPELINE_NAME, PipelineDefinition=definition_json, RoleArn=role_arn,
        )
        print(f"updated pipeline {PIPELINE_NAME} (auc_threshold={auc_threshold})")
    except sm.exceptions.ResourceNotFound:
        import uuid

        sm.create_pipeline(
            PipelineName=PIPELINE_NAME, PipelineDefinition=definition_json, RoleArn=role_arn,
            ClientRequestToken=uuid.uuid4().hex + uuid.uuid4().hex,
        )
        print(f"created pipeline {PIPELINE_NAME} (auc_threshold={auc_threshold})")

    return source_dir_uri


def start_execution(cfg, execution_name: str | None = None) -> str:
    sm = boto3.client("sagemaker", region_name=cfg.region)
    execution_name = execution_name or f"exec-{int(time.time())}"
    resp = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=execution_name,
    )
    return resp["PipelineExecutionArn"]


def wait_for_execution(cfg, execution_arn: str, poll_seconds: int = 20) -> dict:
    sm = boto3.client("sagemaker", region_name=cfg.region)
    while True:
        desc = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        status = desc["PipelineExecutionStatus"]
        print(f"  {execution_arn.split('/')[-1]}: {status}")
        if status in ("Succeeded", "Failed", "Stopped"):
            return desc
        time.sleep(poll_seconds)


def main() -> None:
    cfg = load_config()
    create_or_update_pipeline(cfg)


if __name__ == "__main__":
    main()
