"""Compact benchmark metric summaries for the terminal."""

from __future__ import annotations

import math
from typing import Any


_FIELDS: dict[str, tuple[tuple[str, str, int | None], ...]] = {
    "HPO": (
        ("pattern", "pattern_name", None),
        ("score", "final_score", 3),
        ("faithfulness", "mean_faithfulness", 3),
        ("answer_correctness", "mean_answer_correctness", 3),
        ("context_correctness", "mean_context_correctness", 3),
    ),
    "indexing": (
        ("state", "indexing_state", None),
        ("documents", "indexing_total_documents", 0),
        ("chunks", "indexing_total_chunks", 0),
        ("success_rate", "indexing_ingestion_success_rate", 3),
        ("duration_s", "indexing_duration_seconds", 1),
    ),
    "E2E": (
        ("questions", "e2e_question_count", 0),
        ("exact_match", "e2e_answer_exact_match_mean", 3),
        ("token_f1", "e2e_answer_token_f1_mean", 3),
        ("answer_correctness", "e2e_unitxt_answer_correctness_mean", 3),
        ("faithfulness", "e2e_unitxt_faithfulness_mean", 3),
        ("ragas_answer_relevancy", "e2e_ragas_answer_relevancy_mean", 3),
        ("ragas_faithfulness", "e2e_ragas_faithfulness_mean", 3),
        ("retrieval_hit", "e2e_retrieval_hit_mean", 3),
    ),
}


def format_basic_metrics(stage: str, row: dict[str, Any]) -> str:
    """Format available metrics from an existing result row, with no placeholders."""
    parts: list[str] = []
    for label, key, digits in _FIELDS[stage]:
        value = row.get(key)
        if value is None or value == "":
            continue
        if digits is None:
            parts.append(f"{label}={value}")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parts.append(f"{label}={number:.{digits}f}")
    return ", ".join(parts)
