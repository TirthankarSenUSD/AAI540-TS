"""Loads config.yaml so every script reads bucket/paths from one place."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class Config:
    project: str
    region: str
    bucket: str
    prefixes: dict
    glue_database: str
    feature_group: str
    splits: dict
    seed: int

    @property
    def raw_prefix(self) -> str:
        return self.prefixes["raw"]

    @property
    def processed_prefix(self) -> str:
        return self.prefixes["processed"]

    @property
    def features_prefix(self) -> str:
        return self.prefixes["features"]

    @property
    def athena_results_prefix(self) -> str:
        return self.prefixes["athena_results"]

    def s3_uri(self, prefix_key: str, *parts: str) -> str:
        """Build an s3:// URI under one of the configured prefixes."""
        prefix = self.prefixes[prefix_key]
        suffix = "/".join(parts)
        return f"s3://{self.bucket}/{prefix}{suffix}"


def load_config(path: Path | str = _CONFIG_PATH) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        project=raw["project"],
        region=raw["region"],
        bucket=raw["bucket"],
        prefixes=raw["prefixes"],
        glue_database=raw["glue_database"],
        feature_group=raw["feature_group"],
        splits=raw["splits"],
        seed=raw["seed"],
    )
