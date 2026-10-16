"""Resolve AutoRAG S3 bucket names from merged config or environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _explicit_bucket(explicit: str | None) -> str | None:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    return None


def _bucket_from_storage(config: dict[str, Any], storage_key: str, env_names: tuple[str, ...]) -> str | None:
    storage = config.get("storage") or {}
    bucket = str(storage.get(storage_key) or "").strip()
    if bucket:
        return bucket
    for name in env_names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return None


def resolve_pattern_artifacts_bucket(
    config: dict[str, Any],
    *,
    explicit: str | None = None,
) -> str:
    """Bucket where RAG pipeline writes artifacts / pattern scores (test data bucket)."""
    resolved = _explicit_bucket(explicit)
    if resolved:
        return resolved
    bucket = _bucket_from_storage(
        config,
        "test_data_bucket_name",
        ("BENCHMARK_TEST_DATA_BUCKET_NAME", "TEST_DATA_BUCKET_NAME"),
    )
    if bucket:
        return bucket
    raise ValueError(
        "S3 bucket required for pattern artifacts: pass bucket= or set "
        "BENCHMARK_TEST_DATA_BUCKET_NAME in .env"
    )


def resolve_dataset_upload_bucket(
    config: dict[str, Any],
    *,
    explicit: str | None = None,
) -> str:
    """Bucket for generated RAG datasets (input / knowledge-base bucket)."""
    resolved = _explicit_bucket(explicit)
    if resolved:
        return resolved
    bucket = _bucket_from_storage(
        config,
        "input_data_bucket_name",
        ("BENCHMARK_INPUT_DATA_BUCKET_NAME", "INPUT_DATA_BUCKET_NAME"),
    )
    if bucket:
        return bucket
    raise ValueError(
        "S3 bucket required for dataset upload: pass --s3-bucket or set "
        "BENCHMARK_INPUT_DATA_BUCKET_NAME in .env"
    )


def resolve_dataset_upload_bucket_from_env(env_file: Path | None = None, explicit: str | None = None) -> str:
    """Resolve upload bucket for generate_rag_datasets (CLI / partial .env)."""
    from benchmark_common.credentials import credentials_dict_from_env, load_benchmark_dotenv

    path = load_benchmark_dotenv(env_file)
    if env_file is not None and path is None:
        raise FileNotFoundError(f".env file not found: {env_file.expanduser().resolve()}")

    resolved = _explicit_bucket(explicit)
    if resolved:
        return resolved
    try:
        return resolve_dataset_upload_bucket(credentials_dict_from_env())
    except ValueError:
        pass

    raise ValueError(
        "S3 bucket required for --upload-to-s3: pass --s3-bucket or set "
        "BENCHMARK_INPUT_DATA_BUCKET_NAME in .env"
    )
