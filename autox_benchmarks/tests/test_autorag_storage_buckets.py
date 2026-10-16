"""Tests for AutoRAG S3 bucket resolution."""

from __future__ import annotations

import pytest

from autorag_benchmark.storage_buckets import (
    resolve_dataset_upload_bucket,
    resolve_dataset_upload_bucket_from_env,
    resolve_pattern_artifacts_bucket,
)
from autorag_benchmark.s3_dataset_upload import get_s3_boto_config


def test_resolve_pattern_artifacts_bucket_from_config() -> None:
    cfg = {"storage": {"test_data_bucket_name": "test-bucket"}}
    assert resolve_pattern_artifacts_bucket(cfg) == "test-bucket"


def test_resolve_pattern_artifacts_bucket_explicit_override() -> None:
    cfg = {"storage": {"test_data_bucket_name": "test-bucket"}}
    assert resolve_pattern_artifacts_bucket(cfg, explicit="cli-bucket") == "cli-bucket"


def test_resolve_pattern_artifacts_bucket_missing_fails() -> None:
    with pytest.raises(ValueError, match="BENCHMARK_TEST_DATA_BUCKET_NAME"):
        resolve_pattern_artifacts_bucket({})


def test_resolve_dataset_upload_bucket_from_config() -> None:
    cfg = {"storage": {"input_data_bucket_name": "input-bucket"}}
    assert resolve_dataset_upload_bucket(cfg) == "input-bucket"


def test_dataset_upload_reads_s3_only_env_file(tmp_path, monkeypatch) -> None:
    for name in (
        "BENCHMARK_INPUT_DATA_BUCKET_NAME", "INPUT_DATA_BUCKET_NAME",
        "AWS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / "upload.env"
    env_file.write_text(
        "BENCHMARK_INPUT_DATA_BUCKET_NAME=dataset-bucket\n"
        "AWS_S3_ENDPOINT=https://s3.example.test\n"
        "AWS_ACCESS_KEY_ID=test-key\n"
        "AWS_SECRET_ACCESS_KEY=test-secret\n"
        "AWS_DEFAULT_REGION=eu-central-1\n",
        encoding="utf-8",
    )

    assert resolve_dataset_upload_bucket_from_env(env_file) == "dataset-bucket"
    assert get_s3_boto_config(env_file) == {
        "endpoint_url": "https://s3.example.test",
        "aws_access_key_id": "test-key",
        "aws_secret_access_key": "test-secret",
        "region_name": "eu-central-1",
    }


def test_dataset_upload_rejects_missing_explicit_env_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=".env file not found"):
        resolve_dataset_upload_bucket_from_env(tmp_path / "missing.env")
