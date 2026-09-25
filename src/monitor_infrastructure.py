"""Infrastructure monitor.

This system deliberately has no persistent endpoint (Batch Transform on a
daily cycle, per Week 4's decision), so there's no long-lived
AWS/SageMaker endpoint-invocation metric stream to alarm on. Two
complementary pieces instead:

1. Failure alerting - an EventBridge rule matching SageMaker Transform/
   Training/Processing job state changes (Failed/Stopped) routed straight
   to an SNS topic. This works for any future job run, not just ones
   already executed.
2. Resource utilization - SageMaker auto-publishes per-job
   CPUUtilization/MemoryUtilization/DiskUtilization to CloudWatch under
   /aws/sagemaker/TransformJobs, dimensioned by Host (job name +
   instance id). These are inherently ephemeral (a new Host per job run),
   so alarms on them are demonstrated against a real, already-completed
   job rather than kept as permanent alarms with a stable target.
"""
from __future__ import annotations

import json

import boto3

from config import load_config
from monitor_common import cloudwatch_client, events_client

ALERT_TOPIC_NAME = "diabetes130-ml-alerts"
FAILURE_RULE_PREFIX = "diabetes130-job-failure"

JOB_TYPES = [
    ("SageMaker Transform Job State Change", "TransformJobStatus"),
    ("SageMaker Training Job State Change", "TrainingJobStatus"),
    ("SageMaker Processing Job State Change", "ProcessingJobStatus"),
]


def create_alert_topic(cfg) -> str:
    sns = boto3.client("sns", region_name=cfg.region)
    resp = sns.create_topic(Name=ALERT_TOPIC_NAME)
    topic_arn = resp["TopicArn"]

    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowEventBridgePublish",
            "Effect": "Allow",
            "Principal": {"Service": "events.amazonaws.com"},
            "Action": "sns:Publish",
            "Resource": topic_arn,
        }],
    }
    sns.set_topic_attributes(TopicArn=topic_arn, AttributeName="Policy", AttributeValue=json.dumps(policy))
    return topic_arn


def subscribe_email(cfg, topic_arn: str, email: str) -> None:
    sns = boto3.client("sns", region_name=cfg.region)
    sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=email)


def create_job_failure_rules(cfg, topic_arn: str) -> list[str]:
    events = events_client()
    rule_names = []
    for detail_type, status_field in JOB_TYPES:
        rule_name = f"{FAILURE_RULE_PREFIX}-{status_field.lower()}"
        events.put_rule(
            Name=rule_name,
            EventPattern=json.dumps({
                "source": ["aws.sagemaker"],
                "detail-type": [detail_type],
                "detail": {status_field: ["Failed", "Stopped"]},
            }),
            State="ENABLED",
            Description=f"Alert on {detail_type} reaching Failed/Stopped",
        )
        events.put_targets(
            Rule=rule_name,
            Targets=[{"Id": "alert-topic", "Arn": topic_arn}],
        )
        rule_names.append(rule_name)
    return rule_names


def resource_utilization_alarms(cfg, job_name: str, host_dimension: str, topic_arn: str) -> list[str]:
    cw = cloudwatch_client()
    alarm_names = []
    for metric_name, threshold in [("CPUUtilization", 90), ("MemoryUtilization", 90), ("DiskUtilization", 90)]:
        alarm_name = f"diabetes130-{job_name}-{metric_name}"
        cw.put_metric_alarm(
            AlarmName=alarm_name,
            Namespace="/aws/sagemaker/TransformJobs",
            MetricName=metric_name,
            Dimensions=[{"Name": "Host", "Value": host_dimension}],
            Statistic="Average",
            Period=60,
            EvaluationPeriods=1,
            Threshold=threshold,
            ComparisonOperator="GreaterThanThreshold",
            TreatMissingData="notBreaching",
            AlarmActions=[topic_arn],
            AlarmDescription=f"{metric_name} > {threshold}% on transform job {job_name}",
        )
        alarm_names.append(alarm_name)
    return alarm_names


def main() -> None:
    cfg = load_config()

    topic_arn = create_alert_topic(cfg)
    print("alert topic:", topic_arn)

    rules = create_job_failure_rules(cfg, topic_arn)
    print("failure-alert rules:", rules)

    alarms = resource_utilization_alarms(
        cfg,
        job_name="diabetes130-batch-transform-v2-1790229539",
        host_dimension="diabetes130-batch-transform-v2-1790229539/i-0f13745d7879fde8e",
        topic_arn=topic_arn,
    )
    print("utilization alarms:", alarms)


if __name__ == "__main__":
    main()
