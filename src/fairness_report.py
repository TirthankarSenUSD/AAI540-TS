"""Launches the custom fairness/explainability report
(models/fairness_report/generate_report.py) as a real SageMaker
Processing Job -- see that script's docstring for why this exists instead
of SageMaker Clarify (Clarify processing is in maintenance mode and
unavailable to this AWS account).

Bundles src/, models/xgboost/, and the deployed model artifacts as the
job's code input, and uses a bash entrypoint to pip-install the couple of
packages this container doesn't ship with (pyarrow, awswrangler) before
running the report script -- Processing Jobs (unlike Training Jobs) allow
an arbitrary shell entrypoint, so there's no toolkit-level "requirements.txt
isn't auto-installed" issue to work around here (that issue was specific
to the Training Job toolkit, encountered in Week 4).
"""
from __future__ import annotations

import shutil
import tarfile
import time
from pathlib import Path

import boto3

from aws_jobs import execution_role_arn, framework_image_uri
from config import load_config
from monitor_common import run_processing_job, s3_input, s3_output, sagemaker_client

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOYED_MODEL_DIR = REPO_ROOT / ".deployed_model"


def bundle_code(cfg) -> str:
    staging = Path("/tmp/fairness_report_code")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copytree(REPO_ROOT / "src", staging / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO_ROOT / "models" / "xgboost", staging / "models" / "xgboost",
                     ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO_ROOT / "models" / "fairness_report" / "generate_report.py", staging / "generate_report.py")
    # config.py resolves config.yaml relative to itself (src/../config.yaml)
    # -- needs to land one level above src/ in the bundle too.
    shutil.copy(REPO_ROOT / "config.yaml", staging / "config.yaml")
    shutil.copytree(DEPLOYED_MODEL_DIR, staging / "deployed_model",
                     ignore=shutil.ignore_patterns("code", "model.tar.gz"))

    tarball = Path("/tmp/fairness_report_code.tar.gz")
    with tarfile.open(tarball, "w:gz") as tar:
        for item in staging.iterdir():
            tar.add(item, arcname=item.name)

    s3 = boto3.client("s3", region_name=cfg.region)
    key = f"{cfg.prefixes['monitoring']}fairness-report/code/code.tar.gz"
    s3.upload_file(str(tarball), cfg.bucket, key)
    return f"s3://{cfg.bucket}/{key}"


def run_fairness_report(cfg, job_name: str) -> dict:
    role_arn = execution_role_arn()
    sm = sagemaker_client()
    image_uri = framework_image_uri("xgboost", cfg.region, "1.7-1", "ml.m5.xlarge")

    code_uri = bundle_code(cfg)
    output_s3 = f"s3://{cfg.bucket}/{cfg.prefixes['monitoring']}fairness-report/output/{job_name}/"

    sm.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=role_arn,
        AppSpecification={
            "ImageUri": image_uri,
            "ContainerEntrypoint": ["/bin/bash", "-c"],
            "ContainerArguments": [
                "mkdir -p /opt/ml/processing/input/code "
                "&& tar -xzf /opt/ml/processing/input/archive/code.tar.gz -C /opt/ml/processing/input/code "
                # numpy pinned explicitly: a plain `pip install pyarrow
                # awswrangler` pulls a newer numpy that's ABI-incompatible
                # with this container's pre-built pandas, breaking `import
                # pandas` outright (ValueError: numpy.dtype size changed).
                "&& pip install -q pyarrow awswrangler numpy==1.24.1 "
                "&& python3 /opt/ml/processing/input/code/generate_report.py",
            ],
        },
        ProcessingInputs=[{
            "InputName": "code",
            "S3Input": {
                "S3Uri": code_uri, "LocalPath": "/opt/ml/processing/input/archive",
                "S3DataType": "S3Prefix", "S3InputMode": "File",
            },
        }],
        ProcessingOutputConfig={"Outputs": [
            {"OutputName": "report", "S3Output": {
                "S3Uri": output_s3, "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob",
            }},
        ]},
        ProcessingResources={
            "ClusterConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 20},
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
    )

    from monitor_common import wait_for_processing_job
    desc = wait_for_processing_job(sm, job_name)
    if desc["ProcessingJobStatus"] != "Completed":
        raise RuntimeError(f"fairness report {job_name} failed: {desc.get('ExitMessage') or desc.get('FailureReason')}")

    return {"report_uri": f"{output_s3}fairness_explainability_report.json"}


def main() -> None:
    import json

    cfg = load_config()
    job_name = f"diabetes130-fairness-report-{int(time.time())}"
    result = run_fairness_report(cfg, job_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
