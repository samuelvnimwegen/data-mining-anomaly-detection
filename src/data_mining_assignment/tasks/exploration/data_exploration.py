"""Data exploration helpers for corpus diagnostics and normalization checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean

import pandas as pd


@dataclass(slots=True)
class ExplorationSummary:
    """Stores high-level corpus statistics.

    Attributes:
        document_count: Total number of documents.
        average_character_count: Mean character count per document.
        average_word_count: Mean word count per document.
        html_like_document_count: Number of docs with HTML-like tags.
        symbol_heavy_document_count: Number of docs with high non-word ratio.
    """

    document_count: int
    average_character_count: float
    average_word_count: float
    html_like_document_count: int
    symbol_heavy_document_count: int


def summarize_corpus(raw_texts: list[str]) -> ExplorationSummary:
    """Creates a compact summary from raw text data.

    Args:
        raw_texts: Raw article texts.

    Returns:
        ExplorationSummary: Summary with key corpus signals.
    """
    if not raw_texts:
        return ExplorationSummary(0, 0.0, 0.0, 0, 0)

    character_count_list = [len(single_text) for single_text in raw_texts]
    word_count_list = [len(re.findall(r"\w+", single_text)) for single_text in raw_texts]
    html_like_document_count = sum(1 for single_text in raw_texts if re.search(r"<[^>]+>", single_text))

    symbol_heavy_document_count = 0
    for single_text in raw_texts:
        total_character_count = max(len(single_text), 1)
        non_word_character_count = len(re.findall(r"[^\w\s]", single_text))
        if non_word_character_count / total_character_count > 0.15:
            symbol_heavy_document_count += 1

    return ExplorationSummary(
        document_count=len(raw_texts),
        average_character_count=mean(character_count_list),
        average_word_count=mean(word_count_list),
        html_like_document_count=html_like_document_count,
        symbol_heavy_document_count=symbol_heavy_document_count,
    )


def compare_normalization_variants(raw_texts: list[str], normalized_texts: list[str]) -> dict[str, float]:
    """Compares token load before and after normalization.

    Args:
        raw_texts: Raw article texts.
        normalized_texts: Normalized article texts.

    Returns:
        dict[str, float]: Reduction and ratio metrics.
    """
    raw_token_count = sum(len(re.findall(r"\w+", single_text.lower())) for single_text in raw_texts)
    normalized_token_count = sum(len(single_text.split()) for single_text in normalized_texts)

    if raw_token_count == 0:
        return {
            "raw_token_count": 0.0,
            "normalized_token_count": 0.0,
            "token_reduction_ratio": 0.0,
        }

    token_reduction_ratio = 1.0 - (normalized_token_count / raw_token_count)
    return {
        "raw_token_count": float(raw_token_count),
        "normalized_token_count": float(normalized_token_count),
        "token_reduction_ratio": token_reduction_ratio,
    }


def build_anomaly_candidate_table(
    document_ids: list[str],
    anomaly_scores: list[float],
    anomaly_mask: list[bool] | None = None,
) -> pd.DataFrame:
    """Builds a ranked anomaly table from model scores.

    Args:
        document_ids: Ordered document ids.
        anomaly_scores: Raw anomaly scores where lower means more abnormal.
        anomaly_mask: Optional boolean anomaly mask from model prediction.

    Returns:
        pd.DataFrame: Ranked anomaly candidates with score details.
    """
    anomaly_candidate_table = pd.DataFrame(
        {
            "doc_id": document_ids,
            "score": anomaly_scores,
        }
    )
    if anomaly_mask is not None:
        anomaly_candidate_table["predicted_anomaly"] = anomaly_mask

    anomaly_candidate_table = anomaly_candidate_table.sort_values("score", ascending=True).reset_index(drop=True)
    anomaly_candidate_table["anomaly_rank"] = anomaly_candidate_table.index + 1
    return anomaly_candidate_table


def sample_top_anomaly_texts(
    anomaly_candidate_table: pd.DataFrame,
    document_ids: list[str],
    raw_texts: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Returns top ranked anomaly rows with raw text snippets.

    Args:
        anomaly_candidate_table: Ranked candidate table.
        document_ids: Ordered document ids aligned with raw texts.
        raw_texts: Raw text list aligned with original doc order.
        top_k: Number of top anomalies to show.

    Returns:
        pd.DataFrame: Top anomaly rows with short text snippets.
    """
    top_anomaly_rows = anomaly_candidate_table.head(top_k).copy()
    raw_text_lookup_by_doc_id = {document_id: text for document_id, text in zip(document_ids, raw_texts, strict=False)}

    # This maps ranked rows back to snippets for quick manual review.
    top_anomaly_rows["snippet"] = [
        str(raw_text_lookup_by_doc_id.get(document_id, ""))[:220].replace("\n", " ")
        for document_id in top_anomaly_rows["doc_id"].tolist()
    ]
    return top_anomaly_rows
