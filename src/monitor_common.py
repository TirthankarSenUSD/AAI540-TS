"""Shared helpers for the monitoring module: baselines, data/model quality
monitors, infrastructure monitors, dashboard, Clarify reports.

Uses the same boto3-first pattern established in Week 4 — the SageMaker
SDK's classic `sagemaker.model_monitor.DefaultModelMonitor` /
`ModelQualityMonitor` classes don't exist in the installed SDK v3's
restructured API, but the underlying mechanism (a Processing Job running
the built-in Model Monitor analyzer container with specific environment
variables) is just boto3 `create_processing_job`, which is stable.
"""
from __future__ import annotations

import time
from pathlib import Path

import boto3

from aws_jobs import execution_role_arn
from config import load_config

# AWS's published Model Monitor pre-built container account, per region.
# Hardcoded for us-east-1 since that's the only region config.yaml points
# at for this project (same simplification as the Debugger rule image in
# Week 4's src/aws_jobs.py).
MODEL_MONITOR_ANALYZER_IMAGE = (
    "156813124566.dkr.ecr.us-east-1.amazonaws.com/sagemaker-model-monitor-analyzer"
)


def wait_for_processing_job(sm_client, job_name: str, poll_seconds: int = 20) -> dict:
    while True:
        desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
        status = desc["ProcessingJobStatus"]
        print(f"  {job_name}: {status}")
        if status in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(poll_seconds)


def run_processing_job(
    sm_client, job_name: str, image_uri: str, role_arn: str,
    inputs: list[dict], outputs: list[dict], env: dict,
    instance_type: str = "ml.m5.xlarge", volume_gb: int = 20,
    max_runtime_seconds: int = 1800,
) -> dict:
    sm_client.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=role_arn,
        AppSpecification={"ImageUri": image_uri},
        Environment=env,
        ProcessingInputs=inputs,
        ProcessingOutputConfig={"Outputs": outputs},
        ProcessingResources={
            "ClusterConfig": {
                "InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": volume_gb,
            }
        },
        StoppingCondition={"MaxRuntimeInSeconds": max_runtime_seconds},
    )
    return wait_for_processing_job(sm_client, job_name)


def s3_input(input_name: str, s3_uri: str, local_path: str) -> dict:
    return {
        "InputName": input_name,
        "S3Input": {
            "S3Uri": s3_uri,
            "LocalPath": local_path,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
        },
    }


def s3_output(output_name: str, s3_uri: str, local_path: str) -> dict:
    return {
        "OutputName": output_name,
        "S3Output": {"S3Uri": s3_uri, "LocalPath": local_path, "S3UploadMode": "EndOfJob"},
    }


def sagemaker_client():
    cfg = load_config()
    return boto3.client("sagemaker", region_name=cfg.region)


def cloudwatch_client():
    cfg = load_config()
    return boto3.client("cloudwatch", region_name=cfg.region)


def events_client():
    cfg = load_config()
    return boto3.client("events", region_name=cfg.region)
