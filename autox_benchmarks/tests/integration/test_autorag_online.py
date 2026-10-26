"""Online integration test for the AutoRAG benchmark orchestrator (real KFP + S3).

Runs one full RAG optimization over the FanOutQA dataset kept in external S3 at
``s3://ai-eng-rag/datasets/rag/fanoutqa/txt/10_gold/``. With eight patterns this is
the same hour-scale run used for the other S3-backed AutoRAG datasets.

Prerequisites: ``autox_tests/.env.rag`` (or ``BENCHMARK_AUTORAG_ENV_FILE`` /
``BENCHMARK_ENV_FILE`` / ``autox_benchmarks/.env``) with KFP, buckets, pipeline
secrets, model lists, and ``AWS_*`` credentials. See ``tests/integration/README.md``.

Run:
  cd autox_benchmarks
  pytest tests/integration/test_autorag_online.py -v -s
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from autorag_benchmark.orchestrator import BenchmarkOrchestrator
from benchmark_common.run_state import is_success_state
from tests.conftest import REPO_ROOT, dry_run_arguments_from_row, read_results_csv
from tests.integration.conftest import AUTORAG_INTEGRATION_BENCHMARK_YAML

pytestmark = pytest.mark.integration

DATASET_ID = "fanoutqa-txt-10-gold"
INPUT_DATA_KEY = "datasets/rag/fanoutqa/txt/10_gold/knowledge_base"
TEST_DATA_KEY = "datasets/rag/fanoutqa/txt/10_gold/benchmark_data.json"


class TestAutoragIntegrationPrerequisites:
    """Explicit first checks; session setup validates the same prerequisites."""

    def test_credentials_kfp_and_s3_ready(
        self,
        autorag_integration_context,
        autorag_merged_config: tuple[dict[str, Any], Path],
    ) -> None:
        cfg, _ = autorag_merged_config
        kfp = cfg.get("kfp") or {}
        storage = cfg.get("storage") or {}
        assert str(kfp.get("host", "")).startswith("http")
        assert str(kfp.get("namespace", "")).strip()
        assert autorag_integration_context.kfp_client is not None
        assert storage.get("input_data_bucket_name")
        assert storage.get("test_data_bucket_name")

    def test_fanoutqa_dataset_present_in_s3(self, autorag_dataset: dict[str, Any]) -> None:
        assert autorag_dataset["id"] == DATASET_ID
        assert autorag_dataset["input_data_key"] == INPUT_DATA_KEY
        assert autorag_dataset["test_data_key"] == TEST_DATA_KEY


class TestFanoutqaBenchmarkRun:
    """End-to-end: one FanOutQA optimization run through the benchmark orchestrator."""

    @pytest.fixture
    def orchestrator(self, autorag_integration_context) -> BenchmarkOrchestrator:
        return BenchmarkOrchestrator(
            AUTORAG_INTEGRATION_BENCHMARK_YAML,
            env_file=autorag_integration_context.env_file,
        )

    def test_orchestrator_fanoutqa_run_succeeds(
        self,
        orchestrator: BenchmarkOrchestrator,
        autorag_output_csv: Path,
    ) -> None:
        code = orchestrator.execute(
            output_csv=autorag_output_csv,
            dry_run=False,
            fail_fast=True,
            dataset_filter="all",
        )
        assert code == 0, "orchestrator returned non-zero exit code"

        df = read_results_csv(autorag_output_csv)
        assert len(df) >= 1, "expected at least one result row"
        assert set(df["dataset_id"]) == {DATASET_ID}
        run_ids = {str(value).strip() for value in df["run_id"]}
        assert len(run_ids) == 1 and next(iter(run_ids)), "expected one KFP run_id"

        states = {str(value).upper() for value in df["state"]}
        assert len(states) == 1, f"expected one terminal state, got {states}"
        state = next(iter(states))
        assert is_success_state(state), (
            f"FanOutQA run failed: state={state}, error={df.iloc[0].get('error', '')}"
        )

        assert "final_score" in df.columns, "pattern scores were not extracted from S3"
        scores = df["final_score"].dropna()
        assert not scores.empty, "no pattern produced a final_score"
        assert (scores.astype(float) >= 0).all(), f"negative pattern score: {scores.tolist()}"

    def test_cli_dry_run_targets_fanoutqa_dataset(
        self,
        autorag_integration_context,
        autorag_output_csv: Path,
    ) -> None:
        out = autorag_output_csv.parent / "cli_dry_run.csv"
        cmd = [
            sys.executable,
            "scripts/autorag_benchmark_orchestrator.py",
            "--config",
            str(AUTORAG_INTEGRATION_BENCHMARK_YAML),
            "--output",
            str(out),
            "--dry-run",
        ]
        if autorag_integration_context.env_file is not None:
            cmd += ["--env-file", str(autorag_integration_context.env_file)]
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
            timeout=300,
        )
        assert result.returncode == 0, (result.stderr or "") + (result.stdout or "")

        df = read_results_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["dataset_id"] == DATASET_ID
        assert str(df.iloc[0]["state"]).upper() == "DRY_RUN"
        arguments = dry_run_arguments_from_row(df.iloc[0])
        assert arguments["input_data_keys"] == [INPUT_DATA_KEY]
        assert arguments["test_data_key"] == TEST_DATA_KEY
        assert arguments["optimization_metric"] == "answer_correctness"
