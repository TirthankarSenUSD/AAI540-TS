"""Shared boto3 helpers for launching SageMaker training/tuning jobs.

Uses boto3 directly, not the SageMaker SDK's Estimator/HyperparameterTuner
classes — the installed SDK (v3, `sagemaker==3.22.1`) has restructured
those into a different training API (`sagemaker.train`) that doesn't match
the classic `sagemaker.estimator.Estimator` surface these scripts were
written against, mirroring the same decision made for Feature Store in
Week 3 (`src/feature_store.py`). The `ImageRetriever` from
`sagemaker.core.image_retriever` is stable enough to use directly for
resolving framework container URIs.
"""
from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path

import boto3

from config import load_config


def execution_role_arn() -> str:
    metadata_path = Path("/opt/ml/metadata/resource-metadata.json")
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if "ExecutionRoleArn" in meta:
            return meta["ExecutionRoleArn"]
    raise RuntimeError("could not determine the SageMaker execution role automatically")


def framework_image_uri(framework: str, region: str, version: str, instance_type: str) -> str:
    from sagemaker.core.image_retriever.image_retriever import ImageRetriever
    return ImageRetriever().retrieve(
        framework=framework, region=region, version=version, instance_type=instance_type,
    )


def upload_source_dir(local_dir: str, cfg, s3_prefix: str) -> str:
    """Package a script-mode source directory (entry point + any sibling
    files) as sourcedir.tar.gz and upload it — this is the contract the
    SageMaker framework containers' training toolkit expects when given
    `sagemaker_program` / `sagemaker_submit_directory` hyperparameters."""
    local_path = Path(local_dir)
    tarball_path = local_path.parent / f"{local_path.name}_sourcedir.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        for item in local_path.iterdir():
            if item.name.startswith("_") or item.name == "__pycache__":
                continue
            tar.add(item, arcname=item.name)

    s3 = boto3.client("s3", region_name=cfg.region)
    key = f"{s3_prefix}/sourcedir.tar.gz"
    s3.upload_file(str(tarball_path), cfg.bucket, key)
    tarball_path.unlink()
    return f"s3://{cfg.bucket}/{key}"


def script_mode_hyperparameters(entry_point: str, submit_directory: str, extra: dict | None = None) -> dict:
    hp = {
        "sagemaker_program": entry_point,
        "sagemaker_submit_directory": submit_directory,
    }
    if extra:
        hp.update({k: str(v) for k, v in extra.items()})
    return hp


def wait_for_training_job(sm_client, job_name: str, poll_seconds: int = 20) -> dict:
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        print(f"  {job_name}: {status}")
        if status in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(poll_seconds)


def wait_for_tuning_job(sm_client, job_name: str, poll_seconds: int = 30) -> dict:
    while True:
        desc = sm_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job_name)
        status = desc["HyperParameterTuningJobStatus"]
        counts = desc.get("TrainingJobStatusCounters", {})
        print(f"  {job_name}: {status} {counts}")
        if status in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(poll_seconds)


def debugger_rule_configurations() -> list[dict]:
    """Overfit, LossNotDecreasing, ClassImbalance — built-in rules, cost
    nothing extra beyond the (small) rule-evaluation instance.

    The rule-evaluator image account (503895931360) is AWS's published
    per-region Debugger rules account; hardcoded for us-east-1 here since
    that's the only region config.yaml points at for this project.
    """
    return [
        {
            "RuleConfigurationName": "Overfit",
            "RuleEvaluatorImage": "503895931360.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rules:latest",
            "RuleParameters": {"rule_to_invoke": "Overfit"},
        },
        {
            "RuleConfigurationName": "LossNotDecreasing",
            "RuleEvaluatorImage": "503895931360.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rules:latest",
            "RuleParameters": {"rule_to_invoke": "LossNotDecreasing"},
        },
        {
            "RuleConfigurationName": "ClassImbalance",
            "RuleEvaluatorImage": "503895931360.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rules:latest",
            "RuleParameters": {"rule_to_invoke": "ClassImbalance"},
        },
    ]
