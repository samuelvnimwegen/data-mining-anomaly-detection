"""Tests for corpus exploration helpers."""

from __future__ import annotations

from data_mining_assignment.tasks.exploration import compare_normalization_variants, summarize_corpus


def test_summarize_corpus_detects_html_and_symbol_heavy_rows() -> None:
    """Checks corpus summary captures html and symbol-heavy documents."""
    summary = summarize_corpus(
        [
            "Normal text about engines and systems.",
            "<html><body>Offer!!! $$$ ###</body></html>",
            "%%%%%%% noisy %%%%%%%",
        ]
    )

    assert summary.document_count == 3
    assert summary.html_like_document_count >= 1
    assert summary.symbol_heavy_document_count >= 1


def test_compare_normalization_variants_returns_ratio() -> None:
    """Checks normalization comparison gives a token reduction ratio."""
    comparison = compare_normalization_variants(
        raw_texts=["This is a sample sentence with many tokens"],
        normalized_texts=["sample sentence token"],
    )

    assert comparison["raw_token_count"] > comparison["normalized_token_count"]
    assert 0.0 < comparison["token_reduction_ratio"] < 1.0
