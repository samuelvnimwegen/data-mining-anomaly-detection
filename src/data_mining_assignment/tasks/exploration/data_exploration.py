"""Data exploration helpers for corpus diagnostics and normalization checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean


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
