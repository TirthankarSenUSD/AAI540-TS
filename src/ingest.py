"""Task 1 — raw data into the S3 datalake.

Uploads the three source files to s3://<bucket>/raw/<source>/<filename>,
one prefix per source so Athena can catalog each independently. Ensures the
bucket has versioning, SSE-KMS encryption, and public access blocked. Writes
a provenance manifest (SHA-256, row count, ingestion timestamp) to
raw/_manifest.json in S3 and to Datasets/_manifest.json locally.

The raw zone is immutable — this script only ever adds objects, never
rewrites existing ones except the manifest, and is safe to re-run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "Datasets"

# (local filename, source name -> raw/<source>/, row-count strategy)
SOURCES = [
    {"filename": "diabetic_data.csv", "source": "encounters"},
    {"filename": "IDS_mapping.csv", "source": "id_mapping"},
    {"filename": "icd9dx2015.csv", "source": "icd9"},
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_data_rows(path: Path) -> int:
    """Row count excluding the header line."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def ensure_bucket(s3, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise

    kwargs = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)

    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})

    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "aws:kms",
                        "KMSMasterKeyID": "alias/aws/s3",
                    },
                    "BucketKeyEnabled": True,
                }
            ]
        },
    )

    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )


def upload_source(s3, bucket: str, raw_prefix: str, local_path: Path, source: str) -> dict:
    key = f"{raw_prefix}{source}/{local_path.name}"
    checksum = sha256_of(local_path)
    row_count = count_data_rows(local_path)

    s3.upload_file(str(local_path), bucket, key)

    return {
        "filename": local_path.name,
        "source": source,
        "s3_key": key,
        "sha256": checksum,
        "row_count": row_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    cfg = load_config()
    s3 = boto3.client("s3", region_name=cfg.region)

    ensure_bucket(s3, cfg.bucket, cfg.region)

    manifest_entries = []
    for source in SOURCES:
        local_path = DATASETS_DIR / source["filename"]
        if not local_path.exists():
            print(f"ERROR: missing local source file {local_path}", file=sys.stderr)
            sys.exit(1)
        entry = upload_source(s3, cfg.bucket, cfg.raw_prefix, local_path, source["source"])
        manifest_entries.append(entry)
        print(f"uploaded {entry['filename']} -> s3://{cfg.bucket}/{entry['s3_key']} "
              f"({entry['row_count']} rows, sha256={entry['sha256'][:12]}...)")

    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "files": manifest_entries}

    manifest_key = f"{cfg.raw_prefix}_manifest.json"
    s3.put_object(
        Bucket=cfg.bucket,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"manifest written to s3://{cfg.bucket}/{manifest_key}")

    local_manifest_path = DATASETS_DIR / "_manifest.json"
    local_manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest written to {local_manifest_path}")


if __name__ == "__main__":
    main()
