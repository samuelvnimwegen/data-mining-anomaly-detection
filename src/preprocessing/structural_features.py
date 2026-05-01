"""Structural text feature extraction for blind anomaly detection."""

from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StructuralFeatureExtractor:
    """Extracts simple structure-based features from raw text.

    These features are task-agnostic and do not use known anomaly tokens.
    They focus on shape, repetition, and formatting patterns in documents.
    """

    eps: float = 1e-9

    def transform(self, raw_texts: list[str]) -> np.ndarray:
        """Builds structural features for each document.

        Args:
            raw_texts: Input document texts in original order.

        Returns:
            np.ndarray: Dense feature matrix with one row per document.
        """
        feature_rows: list[list[float]] = []
        for raw_text in raw_texts:
            feature_rows.append(self._extract_one_text_features(raw_text))
        return np.asarray(feature_rows, dtype=np.float64)

    def _extract_one_text_features(self, raw_text: str) -> list[float]:
        """Extracts structural metrics from one text.

        Args:
            raw_text: Raw text for one document.

        Returns:
            list[float]: Numeric feature values.
        """
        text_value = str(raw_text)
        lower_text_value = text_value.lower()

        char_count = float(max(len(text_value), 1))
        token_list = re.findall(r"\b\w+\b", lower_text_value)
        token_count = float(max(len(token_list), 1))
        unique_token_count = float(len(set(token_list)))

        uppercase_count = float(sum(character.isupper() for character in text_value))
        digit_count = float(sum(character.isdigit() for character in text_value))
        punctuation_count = float(sum(not character.isalnum() and not character.isspace() for character in text_value))

        url_count = float(len(re.findall(r"https?://\S+|www\.\S+", lower_text_value)))
        email_count = float(len(re.findall(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", text_value)))
        html_tag_count = float(len(re.findall(r"<[^>]+>", text_value)))

        # Lookaheads require the token to contain at least one letter AND one digit
        # (e.g. "B2B", "ref123"). Tokens with only letters or only digits are excluded.
        mixed_alnum_token_count = float(len(re.findall(r"\b(?=\w*[a-zA-Z])(?=\w*\d)\w+\b", text_value)))
        placeholder_count = float(len(re.findall(r"\{[^}]+}", text_value)))

        repeated_token_ratio = self._compute_repeated_token_ratio(token_list)
        max_token_run = self._compute_max_token_run(token_list)
        lexical_entropy = self._compute_lexical_entropy(token_list)
        bigram_type_token_ratio = self._compute_bigram_type_token_ratio(token_list)
        compression_ratio = self._compute_compression_ratio(text_value)

        average_token_length = float(sum(len(token) for token in token_list)) / token_count
        type_token_ratio = unique_token_count / token_count

        return [
            char_count,
            token_count,
            average_token_length,
            type_token_ratio,
            repeated_token_ratio,
            max_token_run,
            lexical_entropy,
            uppercase_count / char_count,
            digit_count / char_count,
            punctuation_count / char_count,
            url_count,
            email_count,
            html_tag_count,
            mixed_alnum_token_count / token_count,
            placeholder_count / token_count,
            bigram_type_token_ratio,
            compression_ratio,
        ]

    def get_feature_names(self) -> list[str]:
        """Returns names for structural features.

        Returns:
            list[str]: Ordered feature names.
        """
        return [
            "char_count",
            "token_count",
            "average_token_length",
            "type_token_ratio",
            "repeated_token_ratio",
            "max_token_run",
            "lexical_entropy",
            "uppercase_ratio",
            "digit_ratio",
            "punctuation_ratio",
            "url_count",
            "email_count",
            "html_tag_count",
            "mixed_alnum_token_ratio",
            "placeholder_token_ratio",
            "bigram_type_token_ratio",
            "compression_ratio",
        ]

    def _compute_repeated_token_ratio(self, token_list: list[str]) -> float:
        """Calculates repeated-token share.

        Args:
            token_list: Lowercased word tokens.

        Returns:
            float: Share of repeated tokens.
        """
        if not token_list:
            return 0.0

        token_counter: dict[str, int] = {}
        for token in token_list:
            token_counter[token] = token_counter.get(token, 0) + 1

        repeated_count = sum(count for count in token_counter.values() if count > 1)
        return float(repeated_count) / float(len(token_list))

    def _compute_max_token_run(self, token_list: list[str]) -> float:
        """Finds the longest run of the same token.

        Args:
            token_list: Lowercased word tokens.

        Returns:
            float: Length of the longest same-token run.
        """
        if not token_list:
            return 0.0

        max_run = 1
        current_run = 1
        for index in range(1, len(token_list)):
            if token_list[index] == token_list[index - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        return float(max_run)

    def _compute_lexical_entropy(self, token_list: list[str]) -> float:
        """Calculates normalized token entropy.

        Args:
            token_list: Lowercased word tokens.

        Returns:
            float: Entropy in [0, 1] for non-empty token lists.
        """
        if not token_list:
            return 0.0

        token_counter: dict[str, int] = {}
        for token in token_list:
            token_counter[token] = token_counter.get(token, 0) + 1

        probabilities = [count / len(token_list) for count in token_counter.values()]
        # eps prevents log(0) when one token has probability 1.
        entropy_value = -sum(probability * math.log(probability + self.eps) for probability in probabilities)

        # Divide by log(n_unique_tokens), the maximum possible entropy for this vocabulary,
        # to normalise the result to [0, 1]. A score near 0 means one token dominates.
        max_entropy_value = math.log(len(token_counter) + self.eps)
        if max_entropy_value <= 0.0:
            return 0.0
        return float(entropy_value / max_entropy_value)

    def _compute_bigram_type_token_ratio(self, token_list: list[str]) -> float:
        """Calculates the ratio of unique bigrams to total bigrams.

        A low value indicates heavy phrase repetition, which is a strong
        signal for templated or spam-like documents.

        Args:
            token_list: Lowercased word tokens.

        Returns:
            float: Bigram type-token ratio in [0, 1].
        """
        if len(token_list) < 2:
            return 1.0

        bigrams = [token_list[index] + " " + token_list[index + 1] for index in range(len(token_list) - 1)]
        return float(len(set(bigrams))) / float(len(bigrams))

    def _compute_compression_ratio(self, raw_text: str) -> float:
        """Calculates the gzip compression ratio of the raw text.

        Repetitive documents compress to a much smaller fraction of their
        original size than diverse natural-language documents, making this
        a parameter-free anomaly signal.

        Args:
            raw_text: Original document text before any normalization.

        Returns:
            float: Ratio of compressed size to raw UTF-8 size in (0, 1].
        """
        encoded_text = raw_text.encode("utf-8")
        if not encoded_text:
            return 1.0
        compressed_text = gzip.compress(encoded_text, compresslevel=6)
        return float(len(compressed_text)) / float(len(encoded_text))
