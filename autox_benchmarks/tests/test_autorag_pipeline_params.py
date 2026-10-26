"""Unit tests for mapping AutoRAG manifest entries to pipeline arguments."""

from __future__ import annotations

from pathlib import Path

from autorag_benchmark.pipeline_params import build_pipeline_arguments
from autorag_benchmark.settings import BenchmarkSettings


def _settings(**overrides) -> BenchmarkSettings:
    values = dict(
        config_dir=Path("."),
        pipeline_yaml=None,
        input_data_secret_name="input-secret",
        input_data_bucket_name="input-bucket",
        test_data_secret_name="test-secret",
        test_data_bucket_name="test-bucket",
        maas_secret_name="maas-secret",
        vector_db_secret_name="milvus-secret",
        optimization_metric="answer_correctness",
        optimization_max_rag_patterns=4,
        poll_interval_seconds=30,
        timeout_seconds=3600,
        enable_caching=False,
        experiment_name="rag-benchmark",
        run_name_prefix="rag-benchmark",
        suite="base",
        rhoai_version="unknown",
        benchmark_s3_prefix="benchmarks/rag/base",
        preset="",
        upload_benchmark_results=False,
        embedding_models=["embedding-model"],
        generation_models=["generation-model"],
    )
    values.update(overrides)
    return BenchmarkSettings(**values)


def test_vector_db_secret_is_passed_as_db_secret_name() -> None:
    args = build_pipeline_arguments(
        {
            "id": "dataset",
            "input_data_key": "datasets/rag/dataset/knowledge_base",
            "test_data_key": "datasets/rag/dataset/benchmark_data.json",
        },
        _settings(),
    )

    assert args["db_secret_name"] == "milvus-secret"
    assert "vector_db_secret_name" not in args


def test_manifest_overrides_and_model_fallbacks() -> None:
    args = build_pipeline_arguments(
        {
            "id": "dataset",
            "input_data_key": "datasets/rag/dataset/knowledge_base",
            "test_data_key": "datasets/rag/dataset/benchmark_data.json",
            "optimization_metric": "faithfulness",
            "optimization_max_rag_patterns": 6,
            "embedding_models": ["custom-embedding"],
        },
        _settings(),
    )

    assert args["input_data_keys"] == ["datasets/rag/dataset/knowledge_base"]
    assert args["optimization_metric"] == "faithfulness"
    assert args["optimization_max_rag_patterns"] == 6
    assert args["embedding_models"] == ["custom-embedding"]
    assert args["generation_models"] == ["generation-model"]
