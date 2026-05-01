"""Data exploration helpers for corpus diagnostics and normalization checks.

This module contains small utilities used in notebooks and tests to quickly
inspect corpus-level signals that are useful for both clustering and anomaly
detection. The helpers are intentionally lightweight and deterministic so they
can be used in unit tests and in interactive notebooks without heavy
dependencies.

The main responsibilities are:
- Provide a compact summary of corpus-level statistics.
- Compare token counts before and after normalization.
- Build a ranked anomaly candidate table from model scores.
- Sample raw text snippets for manual inspection of top anomaly candidates.

All functions use simple Python types and return pandas DataFrame or plain
structures to make outputs easy to inspect and save to CSV within notebooks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean

import pandas as pd


@dataclass(slots=True)
class ExplorationSummary:
    """Stores high-level corpus statistics used by diagnostics.

    Attributes:
        document_count: Total number of documents in the corpus.
        average_character_count: Mean number of characters per document.
        average_word_count: Mean number of words per document.
        html_like_document_count: Count of docs that contain HTML-like tags.
        symbol_heavy_document_count: Count of docs with a high ratio of symbols.
    """

    document_count: int
    average_character_count: float
    average_word_count: float
    html_like_document_count: int
    symbol_heavy_document_count: int


def summarize_corpus(raw_texts: list[str]) -> ExplorationSummary:
    """Create a compact summary from raw text data.

    This function computes a few fast diagnostics that help identify obvious
    problems in the corpus such as many HTML snippets or documents with heavy
    non-word symbol usage (which are common in the injected corrupted files).

    Args:
        raw_texts: List of raw document strings in their original order.

    Returns:
        ExplorationSummary: Dataclass with several corpus-level statistics.

    Raises:
        None: The function is defensive and returns zeroed metrics for empty input.
    """
    # Return an empty summary when no input is given.
    if not raw_texts:
        return ExplorationSummary(0, 0.0, 0.0, 0, 0)

    # Count characters and words per document for mean calculations.
    character_count_list = [len(single_text) for single_text in raw_texts]
    word_count_list = [len(re.findall(r"\w+", single_text)) for single_text in raw_texts]

    # Detect documents that include HTML-like tags (e.g. <div>, <p>).
    html_like_document_count = sum(1 for single_text in raw_texts if re.search(r"<[^>]+>", single_text))

    # Count documents where non-word characters make up a high fraction.
    # This often flags spam, repeated punctuation, or corrupted formatting.
    symbol_heavy_document_count = 0
    for single_text in raw_texts:
        total_character_count = max(len(single_text), 1)
        non_word_character_count = len(re.findall(r"[^\w\s]", single_text))
        # Mark document as symbol-heavy when symbols exceed 15% of chars.
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
    """Compare token load before and after normalization.

    This helper is useful to quantify how much token pruning a normalization
    pipeline performs. A large reduction indicates aggressive cleaning; a small
    reduction may indicate normalization is mild.

    Args:
        raw_texts: List of original raw document strings.
        normalized_texts: List of normalized document strings produced by the
            tokenizer/normalizer (must align with `raw_texts` order).

    Returns:
        Dict with keys: 'raw_token_count', 'normalized_token_count',
        'token_reduction_ratio' where the ratio is between 0.0 and 1.0.

    Notes:
        The function counts tokens using a simple word regex on raw text and a
        whitespace split on normalized text. This keeps the metric fast and
        robust across different normalization strategies.
    """
    # Count tokens in raw texts using word regex for a stable baseline.
    raw_token_count = sum(len(re.findall(r"\w+", single_text.lower())) for single_text in raw_texts)
    # Count tokens in the normalized view using whitespace split.
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
    """Build a ranked anomaly candidate table from model scores.

    The table is sorted so that the most abnormal documents appear first. The
    function optionally attaches a boolean `predicted_anomaly` column when the
    calling model already returns a mask, but ranking is always performed on
    the continuous scores to permit deterministic top-k selection later.

    Args:
        document_ids: Ordered list of document identifiers aligned with scores.
        anomaly_scores: Numeric anomaly scores where lower typically means more
            abnormal (as with IsolationForest's score_samples output).
        anomaly_mask: Optional boolean mask produced by a model indicating
            predicted anomalies; used only for extra context in the table.

    Returns:
        DataFrame containing columns: 'doc_id', 'score', optional
        'predicted_anomaly', and 'anomaly_rank' (1-based rank, lowest score = 1).
    """
    anomaly_candidate_table = pd.DataFrame(
        {
            "doc_id": document_ids,
            "score": anomaly_scores,
        }
    )
    # Attach model boolean predictions if available for additional filtering.
    if anomaly_mask is not None:
        anomaly_candidate_table["predicted_anomaly"] = anomaly_mask

    # Sort ascending so that the smallest (most abnormal) scores come first.
    anomaly_candidate_table = anomaly_candidate_table.sort_values("score", ascending=True).reset_index(drop=True)
    anomaly_candidate_table["anomaly_rank"] = anomaly_candidate_table.index + 1
    return anomaly_candidate_table


def sample_top_anomaly_texts(
    anomaly_candidate_table: pd.DataFrame,
    document_ids: list[str],
    raw_texts: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Return top-ranked anomaly rows with short raw text snippets.

    This helper maps the ranked anomaly rows back to the original raw
    document texts and returns a short snippet useful for manual inspection in
    notebooks. The snippet is trimmed and line breaks are replaced to make the
    output table display nicely.

    Args:
        anomaly_candidate_table: Ranked candidate table produced by
            `build_anomaly_candidate_table`.
        document_ids: Ordered document ids aligned with `raw_texts`.
        raw_texts: The original raw text list aligned with document ids.
        top_k: Number of top anomalies to include in the sample (default 10).

    Returns:
        DataFrame with the top-k anomaly rows and an added 'snippet' column
        containing a short preview of the raw text for each row.
    """
    # Pick the top-k rows from the ranked table.
    top_anomaly_rows = anomaly_candidate_table.head(top_k).copy()

    # Build a lightweight lookup to map doc_id back to raw text.
    raw_text_lookup_by_doc_id = {document_id: text for document_id, text in zip(document_ids, raw_texts, strict=False)}

    # Attach a short snippet (first 220 chars) with line breaks removed.
    top_anomaly_rows["snippet"] = [
        str(raw_text_lookup_by_doc_id.get(document_id, ""))[:220].replace("\n", " ")
        for document_id in top_anomaly_rows["doc_id"].tolist()
    ]
    return top_anomaly_rows


def attach_original_text_by_doc_id(
    anomaly_table: pd.DataFrame,
    document_ids: list[str],
    raw_texts: list[str],
    output_text_column_name: str = "text",
) -> pd.DataFrame:
    """Adds original raw text to a table that contains a doc_id column.

    This helper keeps anomaly export logic consistent across notebooks. It maps
    each `doc_id` row back to the raw text from the original dataset and stores
    it in a dedicated output column.

    Args:
        anomaly_table: Input table that must include a `doc_id` column.
        document_ids: Ordered list of document identifiers from the dataset.
        raw_texts: Raw text values aligned with `document_ids`.
        output_text_column_name: Name of the output text column.

    Returns:
        pd.DataFrame: Copy of the input table with an added text column.

    Raises:
        ValueError: If input lists are not aligned in length.
        KeyError: If `doc_id` is missing in the input table.
    """
    if len(document_ids) != len(raw_texts):
        raise ValueError("document_ids and raw_texts must have the same length.")

    if "doc_id" not in anomaly_table.columns:
        raise KeyError("Input table must include a doc_id column.")

    # Build a simple lookup so each exported row can include original text.
    raw_text_lookup_by_doc_id = {document_id: text for document_id, text in zip(document_ids, raw_texts, strict=False)}

    enriched_anomaly_table = anomaly_table.copy()
    enriched_anomaly_table[output_text_column_name] = [
        str(raw_text_lookup_by_doc_id.get(document_id, ""))
        for document_id in enriched_anomaly_table["doc_id"].astype(str).tolist()
    ]
    return enriched_anomaly_table
