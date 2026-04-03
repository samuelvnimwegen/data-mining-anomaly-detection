"""Text preprocessing tools for vectorization."""

from __future__ import annotations

from collections.abc import Iterable

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TextPreprocessor:
    """Builds TF-IDF vectors from text documents.

    Args:
        max_features: Maximum number of terms kept in the vocabulary.
        min_document_frequency: Minimum number of documents for a term.
        max_document_frequency: Maximum document ratio for a term.
        ngram_range: N-gram range used by TF-IDF.
        analyzer_mode: Feature analyzer mode, such as word or char_wb.
    """

    def __init__(
        self,
        max_features: int = 15000,
        min_document_frequency: int = 1,
        max_document_frequency: float = 0.95,
        ngram_range: tuple[int, int] = (1, 2),
        analyzer_mode: str = "word",
    ) -> None:
        """Creates a configured TF-IDF vectorizer.

        Args:
            max_features: Maximum number of terms kept in the vocabulary.
            min_document_frequency: Minimum number of documents for a term.
            max_document_frequency: Maximum document ratio for a term.
            ngram_range: N-gram range used by TF-IDF.
            analyzer_mode: Feature analyzer mode, such as word or char_wb.
        """
        self.max_features = max_features
        self.min_document_frequency = min_document_frequency
        self.max_document_frequency = max_document_frequency
        self.ngram_range = ngram_range
        self.analyzer_mode = analyzer_mode
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer=self.analyzer_mode,
            max_df=self.max_document_frequency,
            max_features=self.max_features,
            min_df=self.min_document_frequency,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )

    def fit_transform(self, document_texts: Iterable[str]) -> spmatrix:
        """Fits TF-IDF and transforms document texts.

        Args:
            document_texts: Iterable with normalized document text.

        Returns:
            spmatrix: Sparse TF-IDF matrix.
        """
        return self.vectorizer.fit_transform(document_texts)

    def transform(self, document_texts: Iterable[str]) -> spmatrix:
        """Transforms document texts with an already fitted vectorizer.

        Args:
            document_texts: Iterable with normalized document text.

        Returns:
            spmatrix: Sparse TF-IDF matrix.
        """
        return self.vectorizer.transform(document_texts)
