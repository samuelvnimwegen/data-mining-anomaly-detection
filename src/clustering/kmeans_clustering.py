"""Clustering logic for unsupervised topic grouping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass(slots=True)
class ClusteringResult:
    """Stores outputs from a clustering run.

    Attributes:
        labels: Cluster label per document row.
        selected_cluster_count: Final number of clusters used by the model.
    """

    labels: np.ndarray
    selected_cluster_count: int


class TextClusterer:
    """Clusters vectorized text data with K-Means.

    Args:
        random_seed: Random seed for repeatable model behavior.
    """

    def __init__(self, random_seed: int = 42) -> None:
        """Sets clusterer settings.

        Args:
            random_seed: Random seed for repeatable model behavior.
        """
        self.random_seed = random_seed

    def run_clustering(
        self,
        tfidf_matrix: spmatrix,
        preferred_cluster_count: int | None = None,
        candidate_cluster_counts: Sequence[int] = (4, 6, 8, 10),
    ) -> ClusteringResult:
        """Runs K-Means and returns labels.

        Args:
            tfidf_matrix: Sparse TF-IDF matrix.
            preferred_cluster_count: Optional fixed cluster count.
            candidate_cluster_counts: Candidate counts used for selection.

        Returns:
            ClusteringResult: Labels and selected cluster count.
        """
        selected_cluster_count = preferred_cluster_count or self._select_cluster_count(
            tfidf_matrix=tfidf_matrix,
            candidate_cluster_counts=candidate_cluster_counts,
        )

        clustering_model = KMeans(
            n_clusters=selected_cluster_count,
            n_init="auto",
            random_state=self.random_seed,
        )
        predicted_labels = clustering_model.fit_predict(tfidf_matrix)
        return ClusteringResult(labels=predicted_labels, selected_cluster_count=selected_cluster_count)

    def run_clustering_on_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 8,
    ) -> ClusteringResult:
        """Runs K-Means on dense sentence embeddings.

        Args:
            embeddings: Dense embedding matrix of shape ``(n_docs, n_dims)``.
            n_clusters: Number of clusters (fixed; no silhouette selection).

        Returns:
            ClusteringResult: Labels and selected cluster count.
        """
        clustering_model = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            random_state=self.random_seed,
        )
        predicted_labels = clustering_model.fit_predict(embeddings)
        return ClusteringResult(labels=predicted_labels, selected_cluster_count=n_clusters)

    def _select_cluster_count(self, tfidf_matrix: spmatrix, candidate_cluster_counts: Sequence[int]) -> int:
        """Selects the best cluster count using silhouette score.

        Args:
            tfidf_matrix: Sparse TF-IDF matrix.
            candidate_cluster_counts: Candidate cluster counts.

        Returns:
            int: Selected cluster count.
        """
        document_count = tfidf_matrix.shape[0]
        filtered_cluster_counts = [count for count in candidate_cluster_counts if 2 <= count < document_count]
        if not filtered_cluster_counts:
            return 2 if document_count >= 2 else 1

        best_cluster_count = filtered_cluster_counts[0]
        best_silhouette_score = -1.0

        for candidate_cluster_count in filtered_cluster_counts:
            candidate_model = KMeans(
                n_clusters=candidate_cluster_count,
                n_init="auto",
                random_state=self.random_seed,
            )
            candidate_labels = candidate_model.fit_predict(tfidf_matrix)

            # Silhouette needs more than one unique label.
            if len(set(candidate_labels)) <= 1:
                continue

            candidate_silhouette_score = silhouette_score(tfidf_matrix, candidate_labels)
            if candidate_silhouette_score > best_silhouette_score:
                best_silhouette_score = candidate_silhouette_score
                best_cluster_count = candidate_cluster_count

        return best_cluster_count


def create_cluster_output(document_ids: Sequence[str], labels: np.ndarray) -> pd.DataFrame:
    """Builds the cluster output DataFrame.

    Args:
        document_ids: Ordered document ids.
        labels: Cluster labels aligned with document ids.

    Returns:
        pd.DataFrame: DataFrame with `doc_id` and `label`.
    """
    return pd.DataFrame({"doc_id": list(document_ids), "label": labels})
