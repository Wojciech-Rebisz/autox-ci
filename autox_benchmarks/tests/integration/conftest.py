"""Session setup for online AutoML and AutoRAG integration tests (.env based)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from benchmark_common.credentials import load_benchmark_dotenv, resolve_env_file_path
from benchmark_common.kfp_client import create_kfp_client, resolve_kfp_token
from benchmark_common.s3_client import make_s3_client, s3_cfg_usable
from benchmark_common.s3_upload import _s3_object_not_found
from tests.conftest import REPO_ROOT

INTEGRATION_CONFIG_DIR = REPO_ROOT / "tests" / "fixtures" / "automl" / "integration"
INTEGRATION_BENCHMARK_YAML = INTEGRATION_CONFIG_DIR / "benchmark.yaml"
LOCAL_SMOKE_TRAIN_CSV = INTEGRATION_CONFIG_DIR / "breast-w_n200.csv"
SMOKE_TRAIN_DATA_KEY = "benchmark/smoke/breast-w_n200.csv"
TABULAR_PIPELINE = REPO_ROOT / "pipelines" / "autogluon-tabular-training-pipeline.yaml"

AUTORAG_INTEGRATION_CONFIG_DIR = REPO_ROOT / "tests" / "fixtures" / "autorag" / "integration"
AUTORAG_INTEGRATION_BENCHMARK_YAML = AUTORAG_INTEGRATION_CONFIG_DIR / "benchmark.yaml"
AUTORAG_INTEGRATION_ENV_CANDIDATES = (
    REPO_ROOT.parent / "autox_tests" / ".env.rag",
    REPO_ROOT / ".env",
)


@dataclass(frozen=True)
class IntegrationContext:
    env_file: Path | None
    merged_config: dict[str, Any]
    config_dir: Path
    kfp_client: Any


@dataclass(frozen=True)
class AutoragIntegrationContext:
    env_file: Path | None
    merged_config: dict[str, Any]
    config_dir: Path
    kfp_client: Any
    s3_client: Any


def _collect_prerequisite_errors() -> list[str]:
    errors: list[str] = []

    if not INTEGRATION_BENCHMARK_YAML.is_file():
        errors.append(f"Missing integration benchmark config: {INTEGRATION_BENCHMARK_YAML}")
    if not LOCAL_SMOKE_TRAIN_CSV.is_file():
        errors.append(f"Missing bundled smoke CSV fixture: {LOCAL_SMOKE_TRAIN_CSV}")
    if not TABULAR_PIPELINE.is_file():
        errors.append(f"Missing tabular pipeline YAML: {TABULAR_PIPELINE}")

    env_path = load_benchmark_dotenv()
    if env_path is None:
        errors.append(
            "Missing .env — copy .env.example to autox_benchmarks/.env and fill in KFP/S3 settings"
        )
        return errors

    try:
        from automl_benchmark.config_loader import load_merged_benchmark_config

        cfg, _config_dir = load_merged_benchmark_config(INTEGRATION_BENCHMARK_YAML)
    except Exception as exc:
        errors.append(f"Failed to load benchmark config with credentials: {exc}")
        return errors

    timeout_env = os.environ.get("BENCHMARK_INTEGRATION_TIMEOUT_SECONDS", "").strip()
    if timeout_env:
        cfg.setdefault("run", {})["timeout_seconds"] = float(timeout_env)

    kfp = cfg.get("kfp") or {}
    if not str(kfp.get("host", "")).startswith("http"):
        errors.append("BENCHMARK_KFP_HOST (or RHOAI_KFP_URL) must be an http(s) URL in .env")
    if not str(kfp.get("namespace", "")).strip():
        errors.append("BENCHMARK_KFP_NAMESPACE (or RHOAI_PROJECT_NAME) is required in .env")
    if not str(kfp.get("experiment_name", "")).strip():
        errors.append("BENCHMARK_KFP_EXPERIMENT_NAME is required in .env")
    if not resolve_kfp_token(kfp):
        token_env = str(kfp.get("token_env", "KFP_API_TOKEN"))
        errors.append(
            f"KFP token required (BENCHMARK_KFP_TOKEN in .env or {token_env} in shell)"
        )

    storage = cfg.get("storage") or {}
    if not str(storage.get("train_data_bucket_name", "")).strip():
        errors.append("BENCHMARK_TRAIN_DATA_BUCKET_NAME is required in .env")

    s3_cfg = cfg.get("s3")
    if not s3_cfg_usable(s3_cfg if isinstance(s3_cfg, dict) else None):
        errors.append("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required in .env")

    pipeline_cfg = cfg.get("pipeline") or {}
    if not str(pipeline_cfg.get("package_path", "")).strip() and not TABULAR_PIPELINE.is_file():
        errors.append("BENCHMARK_TABULAR_PACKAGE_PATH or bundled tabular pipeline YAML is required")

    return errors


def _ensure_smoke_training_csv(cfg: dict[str, Any]) -> None:
    storage = cfg.get("storage") or {}
    bucket = str(storage.get("train_data_bucket_name", "")).strip()
    s3_cfg = cfg.get("s3")
    assert s3_cfg_usable(s3_cfg if isinstance(s3_cfg, dict) else None)

    client = make_s3_client(s3_cfg)
    try:
        client.head_object(Bucket=bucket, Key=SMOKE_TRAIN_DATA_KEY)
        return
    except Exception as exc:
        if not _s3_object_not_found(exc):
            raise

    client.upload_file(
        str(LOCAL_SMOKE_TRAIN_CSV),
        bucket,
        SMOKE_TRAIN_DATA_KEY,
        ExtraArgs={"ContentType": "text/csv"},
    )


def _build_integration_context() -> IntegrationContext:
    errors = _collect_prerequisite_errors()
    if errors:
        pytest.fail(
            "Integration prerequisites not met:\n  - " + "\n  - ".join(errors),
            pytrace=False,
        )

    from automl_benchmark.config_loader import load_merged_benchmark_config

    cfg, config_dir = load_merged_benchmark_config(INTEGRATION_BENCHMARK_YAML)
    timeout_env = os.environ.get("BENCHMARK_INTEGRATION_TIMEOUT_SECONDS", "").strip()
    if timeout_env:
        cfg.setdefault("run", {})["timeout_seconds"] = float(timeout_env)

    try:
        _ensure_smoke_training_csv(cfg)
    except Exception as exc:
        pytest.fail(f"Could not ensure smoke training CSV on S3: {exc}", pytrace=False)

    try:
        kfp_client = create_kfp_client(cfg)
        kfp_client.list_experiments(page_size=1)
    except Exception as exc:
        pytest.fail(f"KFP connectivity check failed: {exc}", pytrace=False)

    return IntegrationContext(
        env_file=resolve_env_file_path(),
        merged_config=cfg,
        config_dir=config_dir,
        kfp_client=kfp_client,
    )


def _resolve_autorag_env_file() -> Path | None:
    explicit = os.environ.get("BENCHMARK_AUTORAG_ENV_FILE") or os.environ.get("BENCHMARK_ENV_FILE")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    for candidate in AUTORAG_INTEGRATION_ENV_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _autorag_dataset_entry(cfg: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    from benchmark_common.manifest import load_dataset_entries

    datasets = load_dataset_entries(cfg, config_dir)
    if len(datasets) != 1:
        raise ValueError(
            f"AutoRAG integration manifest must contain exactly one dataset, got {len(datasets)}"
        )
    return datasets[0]


def _validate_autorag_dataset_objects(
    dataset: dict[str, Any],
    cfg: dict[str, Any],
    s3_client: Any,
) -> list[str]:
    """Verify the manifest dataset exists in S3 and its benchmark references real documents."""
    errors: list[str] = []
    storage = cfg.get("storage") or {}
    input_bucket = str(storage.get("input_data_bucket_name", "")).strip()
    test_bucket = str(storage.get("test_data_bucket_name", "")).strip()
    input_key = str(dataset.get("input_data_key", "")).strip().rstrip("/")
    test_key = str(dataset.get("test_data_key", "")).strip().lstrip("/")
    if not input_key:
        return ["Dataset manifest entry missing input_data_key"]
    if not test_key:
        return ["Dataset manifest entry missing test_data_key"]

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        document_keys = {
            obj["Key"]
            for page in paginator.paginate(Bucket=input_bucket, Prefix=f"{input_key}/")
            for obj in page.get("Contents") or []
        }
    except Exception as exc:
        return [f"Cannot list s3://{input_bucket}/{input_key}/: {exc}"]
    if not document_keys:
        errors.append(f"No input documents under s3://{input_bucket}/{input_key}/")

    try:
        payload = s3_client.get_object(Bucket=test_bucket, Key=test_key)["Body"].read()
        benchmark = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        return [*errors, f"Cannot read s3://{test_bucket}/{test_key}: {exc}"]
    if not isinstance(benchmark, list) or not benchmark:
        return [*errors, f"Benchmark data at s3://{test_bucket}/{test_key} is empty or not a list"]

    referenced = {
        key
        for record in benchmark
        for key in (record.get("correct_answer_document_keys") or [])
    }
    missing = sorted(referenced - document_keys)
    if missing:
        errors.append(
            f"Benchmark references {len(missing)} document(s) missing under "
            f"s3://{input_bucket}/{input_key}/ (e.g. {missing[:3]})"
        )
    return errors


def _collect_autorag_prerequisite_errors(
    cfg: dict[str, Any],
    config_dir: Path,
    s3_client: Any,
) -> list[str]:
    try:
        dataset = _autorag_dataset_entry(cfg, config_dir)
    except Exception as exc:
        return [f"Invalid AutoRAG dataset manifest: {exc}"]
    errors = _validate_autorag_dataset_objects(dataset, cfg, s3_client)
    kfp = cfg.get("kfp") or {}
    storage = cfg.get("storage") or {}
    pipeline = cfg.get("pipeline") or {}
    if not str(kfp.get("host", "")).startswith("http"):
        errors.append(
            "KFP host is required (BENCHMARK_KFP_HOST or RHOAI_KFP_URL in the env file)"
        )
    if not str(kfp.get("namespace", "")).strip():
        errors.append("KFP namespace is required (BENCHMARK_KFP_NAMESPACE or RHOAI_PROJECT_NAME)")
    if not resolve_kfp_token(kfp):
        errors.append("KFP token is required (BENCHMARK_KFP_TOKEN or RHOAI_TOKEN)")
    for name in ("input_data_bucket_name", "test_data_bucket_name"):
        if not str(storage.get(name, "")).strip():
            errors.append(f"storage.{name} is required")
    for name in ("input_data_secret_name", "test_data_secret_name", "maas_secret_name", "vector_db_secret_name"):
        if not str(pipeline.get(name, "")).strip():
            errors.append(f"pipeline.{name} is required")
    return errors


def _build_autorag_integration_context() -> AutoragIntegrationContext:
    if not AUTORAG_INTEGRATION_BENCHMARK_YAML.is_file():
        pytest.fail(
            f"Missing AutoRAG integration benchmark config: {AUTORAG_INTEGRATION_BENCHMARK_YAML}",
            pytrace=False,
        )
    env_file = _resolve_autorag_env_file()
    if env_file is None:
        pytest.fail(
            "Missing AutoRAG env file. Set BENCHMARK_AUTORAG_ENV_FILE (or BENCHMARK_ENV_FILE), "
            "or provide autox_tests/.env.rag / autox_benchmarks/.env.",
            pytrace=False,
        )

    timeout_env = os.environ.get("BENCHMARK_INTEGRATION_TIMEOUT_SECONDS", "").strip()
    if timeout_env and not os.environ.get("BENCHMARK_TIMEOUT_SECONDS"):
        os.environ["BENCHMARK_TIMEOUT_SECONDS"] = timeout_env

    try:
        from autorag_benchmark.config_loader import load_merged_benchmark_config

        cfg, config_dir = load_merged_benchmark_config(AUTORAG_INTEGRATION_BENCHMARK_YAML, env_file)
    except Exception as exc:
        pytest.fail(f"Failed to load AutoRAG benchmark config: {exc}", pytrace=False)

    s3_cfg = cfg.get("s3")
    if not s3_cfg_usable(s3_cfg if isinstance(s3_cfg, dict) else None):
        pytest.fail(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required in the env file",
            pytrace=False,
        )
    s3_client = make_s3_client(s3_cfg)

    errors = _collect_autorag_prerequisite_errors(cfg, config_dir, s3_client)
    if errors:
        pytest.fail(
            "AutoRAG integration prerequisites not met:\n  - " + "\n  - ".join(errors),
            pytrace=False,
        )

    try:
        kfp_client = create_kfp_client(cfg)
        kfp_client.list_experiments(page_size=1)
    except Exception as exc:
        pytest.fail(f"KFP connectivity check failed: {exc}", pytrace=False)

    return AutoragIntegrationContext(
        env_file=env_file,
        merged_config=cfg,
        config_dir=config_dir,
        kfp_client=kfp_client,
        s3_client=s3_client,
    )


@pytest.fixture(scope="session")
def integration_context() -> IntegrationContext:
    return _build_integration_context()


@pytest.fixture(scope="session")
def autorag_integration_context() -> AutoragIntegrationContext:
    return _build_autorag_integration_context()


@pytest.fixture(scope="session", autouse=True)
def _integration_session_gate(request: pytest.FixtureRequest) -> None:
    """Validate prerequisites once for whichever online suites were selected."""
    module_names = {item.module.__name__ for item in request.session.items if item.module}
    if any(name.endswith("test_automl_online") for name in module_names):
        request.getfixturevalue("integration_context")
    if any(name.endswith("test_autorag_online") for name in module_names):
        request.getfixturevalue("autorag_integration_context")


@pytest.fixture(scope="session")
def integration_env_file(integration_context: IntegrationContext) -> Path | None:
    return integration_context.env_file


@pytest.fixture(scope="session")
def integration_benchmark_yaml() -> Path:
    assert INTEGRATION_BENCHMARK_YAML.is_file()
    return INTEGRATION_BENCHMARK_YAML


@pytest.fixture(scope="session")
def integration_merged_config(
    integration_context: IntegrationContext,
) -> tuple[dict[str, Any], Path]:
    return integration_context.merged_config, integration_context.config_dir


@pytest.fixture(scope="session")
def integration_kfp_client(integration_context: IntegrationContext):
    return integration_context.kfp_client


@pytest.fixture
def integration_output_csv(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("integration_out") / "smoke_benchmark_runs.csv"


@pytest.fixture(scope="session")
def autorag_merged_config(
    autorag_integration_context: AutoragIntegrationContext,
) -> tuple[dict[str, Any], Path]:
    return autorag_integration_context.merged_config, autorag_integration_context.config_dir


@pytest.fixture(scope="session")
def autorag_dataset(autorag_integration_context: AutoragIntegrationContext) -> dict[str, Any]:
    return _autorag_dataset_entry(
        autorag_integration_context.merged_config,
        autorag_integration_context.config_dir,
    )


@pytest.fixture
def autorag_output_csv(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("autorag_integration_out") / "fanoutqa_benchmark_runs.csv"
