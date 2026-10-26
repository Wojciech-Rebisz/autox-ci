# Online integration tests (AutoML + AutoRAG)

Exercises **real KFP and S3**. The AutoML suite uses the smallest tabular smoke path
(**breast-w-smoke**, 200-row bundled sample, **`top_n: 1`**). The AutoRAG suite runs one
full optimization over the **FanOutQA** dataset stored in external S3
(`s3://ai-eng-rag/datasets/rag/fanoutqa/txt/10_gold/`), which takes roughly an hour with
eight patterns — the same shape as the other S3-backed AutoRAG datasets.

## Prerequisite

Copy [`.env.example`](../../.env.example) to **`autox_benchmarks/.env`** and set:

- `BENCHMARK_KFP_HOST`, `BENCHMARK_KFP_NAMESPACE`, `BENCHMARK_KFP_EXPERIMENT_NAME`, `BENCHMARK_KFP_TOKEN` (or `KFP_API_TOKEN`)
- `BENCHMARK_TRAIN_DATA_BUCKET_NAME`, `BENCHMARK_TRAIN_DATA_SECRET_NAME`
- `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

Names align with `autox_tests` (e.g. `RHOAI_KFP_URL`, `RHOAI_TOKEN` work as aliases).

No manual S3 upload: the suite uploads `tests/fixtures/automl/integration/breast-w_n200.csv` when that object is missing on your train-data bucket.

## Run

```bash
cd autox_benchmarks
pytest tests/integration/test_automl_online.py -v -s
pytest tests/integration/test_autorag_online.py -v -s
```

Optional:

```bash
export BENCHMARK_INTEGRATION_TIMEOUT_SECONDS=3600
export KFP_API_TOKEN="$(oc whoami -t)"   # if not in .env
```

`pytest tests/` does **not** collect integration tests unless you pass `tests/integration/` or set `BENCHMARK_RUN_INTEGRATION=1`.

## What is tested

| Test | Checks |
|------|--------|
| `test_credentials_kfp_and_s3_ready` | `.env`, KFP connectivity, smoke CSV on bucket (upload if needed) |
| `test_orchestrator_smoke_run_succeeds` | Full orchestrator smoke run |
| `test_cli_smoke_run_subprocess` | Same via `scripts/benchmark_orchestrator.py` |

Session setup validates prerequisites once. If anything is missing or unreachable, the suite fails immediately.

Expect several minutes to tens of minutes depending on cluster load (AutoGluon with `top_n=1` on 200 rows).

## AutoRAG (FanOutQA)

The AutoRAG integration test reuses the AutoRAG settings from `autox_tests/.env.rag` by
default; override the file with `BENCHMARK_AUTORAG_ENV_FILE` (or `BENCHMARK_ENV_FILE`).
Alternatively create `autox_benchmarks/.env` from `.env.example`.

Required values (aliases in parentheses are accepted):

- `BENCHMARK_KFP_HOST` (`RHOAI_KFP_URL`), `BENCHMARK_KFP_NAMESPACE` (`RHOAI_PROJECT_NAME`), `BENCHMARK_KFP_TOKEN` (`RHOAI_TOKEN`)
- `BENCHMARK_INPUT_DATA_BUCKET_NAME` / `BENCHMARK_TEST_DATA_BUCKET_NAME` (`INPUT_DATA_BUCKET_NAME` / `TEST_DATA_BUCKET_NAME`); both point at `ai-eng-rag`
- `BENCHMARK_INPUT_DATA_SECRET_NAME`, `BENCHMARK_TEST_DATA_SECRET_NAME`, `BENCHMARK_MAAS_SECRET_NAME`, `BENCHMARK_VECTOR_DB_SECRET_NAME`
- `BENCHMARK_EMBEDDING_MODELS` / `BENCHMARK_GENERATION_MODELS`
- `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

| Test | Checks |
|------|--------|
| `test_credentials_kfp_and_s3_ready` | KFP connectivity, buckets, and secrets are configured |
| `test_fanoutqa_dataset_present_in_s3` | Manifest keys and benchmark document references exist in the bucket |
| `test_orchestrator_fanoutqa_run_succeeds` | Full optimization run: `SUCCEEDED`, one `run_id`, pattern `final_score` extracted from S3 |
| `test_cli_dry_run_targets_fanoutqa_dataset` | CLI dry-run emits the FanOutQA input/test keys |

The run uses `tests/fixtures/autorag/integration/benchmark.yaml` and
`dataset_manifest.yaml` (eight patterns, `balanced` preset, `answer_correctness`, results
upload disabled).
Override the wall-clock cap with `BENCHMARK_INTEGRATION_TIMEOUT_SECONDS` (default `7200`).
`BENCHMARK_OPTIMIZATION_MAX_RAG_PATTERNS` also works, but the pipeline itself validates
the value against the range **4–10**, so it cannot be used to shrink a smoke run below
four patterns.
