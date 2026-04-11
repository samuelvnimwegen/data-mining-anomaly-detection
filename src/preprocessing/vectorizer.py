"""Text vectorization tools for clustering and anomaly detection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.sparse import spmatrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextPreprocessor:
    """Builds configurable vector features from normalized documents.

    Args:
        vectorization_model_name: Model name, such as `tfidf`, `bow`, `tfidf_lsa_dense`, or `tfidf_svd_dense`.
        max_features: Maximum number of terms kept in the vocabulary.
        min_document_frequency: Minimum number of documents for a term.
        max_document_frequency: Maximum document ratio for a term.
        ngram_range: N-gram range used by the vectorizer.
        analyzer_mode: Analyzer mode, such as `word` or `char_wb`.
        dense_embedding_dimension: Dimension used when dense SVD vectors are selected.
        random_seed: Random seed for repeatable dense projection.
    """

    def __init__(
        self,
        vectorization_model_name: str = "tfidf",
        max_features: int = 15000,
        min_document_frequency: int = 1,
        max_document_frequency: float = 0.95,
        ngram_range: tuple[int, int] = (1, 2),
        analyzer_mode: str = "word",
        dense_embedding_dimension: int = 256,
        random_seed: int = 42,
    ) -> None:
        """Creates configured vectorization components.

        Args:
            vectorization_model_name: Model name, such as `tfidf`, `bow`, `tfidf_lsa_dense`, or `tfidf_svd_dense`.
            max_features: Maximum number of terms kept in the vocabulary.
            min_document_frequency: Minimum number of documents for a term.
            max_document_frequency: Maximum document ratio for a term.
            ngram_range: N-gram range used by the vectorizer.
            analyzer_mode: Analyzer mode, such as `word` or `char_wb`.
            dense_embedding_dimension: Dimension used when dense SVD vectors are selected.
            random_seed: Random seed for repeatable dense projection.
        """
        self.vectorization_model_name = self._normalize_model_name(vectorization_model_name)
        self.max_features = max_features
        self.min_document_frequency = min_document_frequency
        self.max_document_frequency = max_document_frequency
        self.ngram_range = ngram_range
        self.analyzer_mode = analyzer_mode
        self.dense_embedding_dimension = dense_embedding_dimension
        self.random_seed = random_seed

        self.vectorizer = self._build_vectorizer()
        self.dense_projector: TruncatedSVD | None = None
        if self.vectorization_model_name == "tfidf_lsa_dense":
            self.dense_projector = TruncatedSVD(
                n_components=self.dense_embedding_dimension,
                random_state=self.random_seed,
            )

    def fit_transform(self, document_texts: Iterable[str]) -> spmatrix | np.ndarray:
        """Fits the vectorizer and transforms document texts.

        Args:
            document_texts: Iterable with normalized document text.

        Returns:
            spmatrix | np.ndarray: Sparse vectors or dense vectors.

        Raises:
            ValueError: If the vectorization model is not supported.
        """
        sparse_matrix = self.vectorizer.fit_transform(document_texts)
        if self.vectorization_model_name != "tfidf_lsa_dense":
            return sparse_matrix

        assert self.dense_projector is not None

        # This keeps dense size valid for very small datasets.
        reduced_dimension = min(self.dense_embedding_dimension, max(1, sparse_matrix.shape[1] - 1))
        self.dense_projector = TruncatedSVD(
            n_components=reduced_dimension,
            random_state=self.random_seed,
        )
        return self.dense_projector.fit_transform(sparse_matrix)

    def transform(self, document_texts: Iterable[str]) -> spmatrix | np.ndarray:
        """Transforms document texts with an already fitted model.

        Args:
            document_texts: Iterable with normalized document text.

        Returns:
            spmatrix | np.ndarray: Sparse vectors or dense vectors.
        """
        sparse_matrix = self.vectorizer.transform(document_texts)
        if self.vectorization_model_name != "tfidf_lsa_dense":
            return sparse_matrix

        assert self.dense_projector is not None
        return self.dense_projector.transform(sparse_matrix)

    def get_feature_names(self) -> list[str]:
        """Returns vectorizer feature names when available.

        Returns:
            list[str]: List of feature names. Returns empty list for dense-only views.
        """
        if self.vectorization_model_name == "tfidf_lsa_dense":
            return []
        return list(self.vectorizer.get_feature_names_out())

    def _build_vectorizer(self) -> CountVectorizer | TfidfVectorizer:
        """Builds the selected sparse vectorizer.

        Returns:
            CountVectorizer | TfidfVectorizer: Configured sparse vectorizer.

        Raises:
            ValueError: If the vectorization model is not supported.
        """
        shared_arguments = {
            "lowercase": False,
            "analyzer": self.analyzer_mode,
            "max_df": self.max_document_frequency,
            "max_features": self.max_features,
            "min_df": self.min_document_frequency,
            "ngram_range": self.ngram_range,
        }

        if self.vectorization_model_name == "tfidf":
            return TfidfVectorizer(
                sublinear_tf=True,
                **shared_arguments,
            )

        if self.vectorization_model_name == "bow":
            return CountVectorizer(**shared_arguments)

        if self.vectorization_model_name == "tfidf_lsa_dense":
            return TfidfVectorizer(
                sublinear_tf=True,
                **shared_arguments,
            )

        raise ValueError(
            "Unsupported vectorization_model_name. Use one of: tfidf, bow, tfidf_lsa_dense, tfidf_svd_dense."
        )

    def _normalize_model_name(self, vectorization_model_name: str) -> str:
        """Normalizes model aliases to one internal name.

        Args:
            vectorization_model_name: Raw model name from config.

        Returns:
            str: Internal model name.
        """
        if vectorization_model_name == "tfidf_svd_dense":
            # Keep this alias for compatibility with earlier experiments.
            return "tfidf_lsa_dense"
        return vectorization_model_name
