"""Full ArXiv PDF corpus generation keeps all positives and hard negatives."""

from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub

from autorag_benchmark.datasets import open_ragbench


def test_full_pdf_corpus_includes_unreferenced_pdfs(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    fixtures = {
        "queries": {
            "q1": {"query": "First?"},
            "q2": {"query": "Second?"},
            "q3": {"query": "Third?"},
        },
        "answers": {"q1": "one", "q2": "two", "q3": "three"},
        "qrels": {
            "q1": {"doc_id": "positive-1"},
            "q2": {"doc_id": "positive-1"},
            "q3": {"doc_id": "positive-2"},
        },
        "pdf_urls": {
            "positive-1": "https://example.invalid/1.pdf",
            "positive-2": "https://example.invalid/2.pdf",
            "negative": "https://example.invalid/3.pdf",
        },
    }
    for name, value in fixtures.items():
        (source / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: str(source / Path(kwargs["filename"]).name),
    )
    downloaded: list[str] = []

    def fake_download(url: str, output_path: Path) -> bool:
        if not output_path.exists():
            downloaded.append(output_path.name)
            output_path.write_bytes(b"%PDF-1.4")
        return True

    monkeypatch.setattr(open_ragbench, "_download_pdf", fake_download)
    kb_dir = tmp_path / "knowledge_base"
    bench_path = tmp_path / "benchmark_data.json"

    documents, questions = open_ragbench.prepare(
        kb_dir,
        bench_path,
        num_samples=3,
        output_format="pdf",
        include_all_pdfs=True,
    )

    assert (documents, questions) == (3, 3)
    assert sorted(downloaded) == [
        "open_ragbench_negative.pdf",
        "open_ragbench_positive-1.pdf",
        "open_ragbench_positive-2.pdf",
    ]
    records = json.loads(bench_path.read_text(encoding="utf-8"))
    assert records[0]["correct_answer_document_keys"] == ["open_ragbench_positive-1.pdf"]
    assert records[1]["correct_answer_document_keys"] == ["open_ragbench_positive-1.pdf"]
    assert records[2]["correct_answer_document_keys"] == ["open_ragbench_positive-2.pdf"]
