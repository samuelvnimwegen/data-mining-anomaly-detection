"""Tests for vectorization model variants."""

from __future__ import annotations

import numpy as np
from scipy.sparse import spmatrix

from data_mining_assignment.tasks.preprocessing import TextPreprocessor


def test_tfidf_vectorization_returns_sparse_matrix() -> None:
    """Checks TF-IDF mode returns sparse vectors and feature names."""
    preprocessor = TextPreprocessor(
        vectorization_model_name="tfidf",
        min_document_frequency=1,
        max_document_frequency=1.0,
        ngram_range=(1, 1),
    )

    transformed_matrix = preprocessor.fit_transform(["engine failure", "engine alert"])

    assert isinstance(transformed_matrix, spmatrix)
    assert transformed_matrix.shape[0] == 2
    assert "engine" in preprocessor.get_feature_names()


def test_bow_vectorization_returns_sparse_matrix() -> None:
    """Checks BoW mode returns sparse vectors and feature names."""
    preprocessor = TextPreprocessor(
        vectorization_model_name="bow",
        min_document_frequency=1,
        max_document_frequency=1.0,
        ngram_range=(1, 1),
    )

    transformed_matrix = preprocessor.fit_transform(["pump issue", "pump failure"])

    assert isinstance(transformed_matrix, spmatrix)
    assert transformed_matrix.shape[0] == 2
    assert "pump" in preprocessor.get_feature_names()


def test_tfidf_lsa_dense_vectorization_returns_dense_matrix() -> None:
    """Checks explicit LSA mode returns dense vectors."""
    preprocessor = TextPreprocessor(
        vectorization_model_name="tfidf_lsa_dense",
        min_document_frequency=1,
        max_document_frequency=1.0,
        ngram_range=(1, 1),
        dense_embedding_dimension=4,
        random_seed=42,
    )

    transformed_matrix = preprocessor.fit_transform(
        [
            "aircraft engine inspection",
            "medical symptom report",
            "car gearbox vibration",
        ]
    )

    assert isinstance(transformed_matrix, np.ndarray)
    assert transformed_matrix.shape[0] == 3
    assert transformed_matrix.shape[1] <= 4
    assert preprocessor.get_feature_names() == []


def test_tfidf_svd_dense_alias_maps_to_lsa_mode() -> None:
    """Checks legacy SVD alias still works and maps to the LSA mode."""
    preprocessor = TextPreprocessor(
        vectorization_model_name="tfidf_svd_dense",
        min_document_frequency=1,
        max_document_frequency=1.0,
        ngram_range=(1, 1),
        dense_embedding_dimension=4,
        random_seed=42,
    )

    transformed_matrix = preprocessor.fit_transform(
        [
            "signal integrity report",
            "engine telemetry diagnostics",
            "anomaly spike detection",
        ]
    )

    assert preprocessor.vectorization_model_name == "tfidf_lsa_dense"
    assert isinstance(transformed_matrix, np.ndarray)
