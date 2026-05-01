"""Tests for corpus exploration helpers."""

from __future__ import annotations
from dataclasses import asdict

from exploration import (
    attach_original_text_by_doc_id,
    build_anomaly_candidate_table,
    compare_normalization_variants,
    sample_top_anomaly_texts,
    summarize_corpus,
)


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


def test_build_anomaly_candidate_table_orders_by_lowest_score() -> None:
    """Checks anomaly table ranking uses ascending anomaly score."""
    anomaly_table = build_anomaly_candidate_table(
        document_ids=["DOC_001", "DOC_002", "DOC_003"],
        anomaly_scores=[-0.10, -0.42, -0.21],
        anomaly_mask=[False, True, True],
    )

    assert anomaly_table.iloc[0]["doc_id"] == "DOC_002"
    assert anomaly_table.iloc[0]["anomaly_rank"] == 1


def test_sample_top_anomaly_texts_adds_matching_snippets() -> None:
    """Checks top anomaly sampler attaches snippets by document id."""
    anomaly_table = build_anomaly_candidate_table(
        document_ids=["DOC_001", "DOC_002", "DOC_003"],
        anomaly_scores=[-0.10, -0.42, -0.21],
    )

    top_rows = sample_top_anomaly_texts(
        anomaly_candidate_table=anomaly_table,
        document_ids=["DOC_001", "DOC_002", "DOC_003"],
        raw_texts=[
            "normal vehicle maintenance log",
            "buy now ### broken html <div>",
            "sudden random symbol burst !!!!!",
        ],
        top_k=2,
    )

    assert "snippet" in top_rows.columns
    assert top_rows.iloc[0]["doc_id"] == "DOC_002"
    assert "buy now" in top_rows.iloc[0]["snippet"]


def test_summarize_corpus_supports_dataclass_asdict_conversion() -> None:
    """Checks summary dataclass can be serialized with asdict."""
    summary = summarize_corpus(["Simple text", "Another row"])
    summary_as_dictionary = asdict(summary)

    assert "document_count" in summary_as_dictionary
    assert summary_as_dictionary["document_count"] == 2


def test_attach_original_text_by_doc_id_adds_text_column() -> None:
    """Checks helper appends aligned text column by doc_id."""
    anomaly_table = build_anomaly_candidate_table(
        document_ids=["DOC_001", "DOC_002"],
        anomaly_scores=[-0.2, -0.5],
    )

    enriched_table = attach_original_text_by_doc_id(
        anomaly_table=anomaly_table,
        document_ids=["DOC_001", "DOC_002"],
        raw_texts=["normal line", "outlier line"],
    )

    assert "text" in enriched_table.columns
    assert enriched_table.iloc[0]["doc_id"] == "DOC_002"
    assert enriched_table.iloc[0]["text"] == "outlier line"


def test_attach_original_text_by_doc_id_validates_input() -> None:
    """Checks helper raises for missing doc_id or length mismatch."""
    anomaly_table_without_doc_id = build_anomaly_candidate_table(
        document_ids=["DOC_001"],
        anomaly_scores=[-0.2],
    )[["score", "anomaly_rank"]]

    try:
        attach_original_text_by_doc_id(
            anomaly_table=anomaly_table_without_doc_id,
            document_ids=["DOC_001"],
            raw_texts=["text"],
        )
        assert False, "Expected KeyError for missing doc_id column."
    except KeyError:
        assert True

    try:
        attach_original_text_by_doc_id(
            anomaly_table=build_anomaly_candidate_table(
                document_ids=["DOC_001"],
                anomaly_scores=[-0.2],
            ),
            document_ids=["DOC_001", "DOC_002"],
            raw_texts=["only one text"],
        )
        assert False, "Expected ValueError for mismatched lengths."
    except ValueError:
        assert True
