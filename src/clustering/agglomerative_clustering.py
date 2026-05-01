"""Agglomerative (hierarchical) clustering for unsupervised topic grouping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import spmatrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from clustering.kmeans_clustering import ClusteringResult


class AgglomerativeTextClusterer:
    """Clusters vectorized text using Ward-linkage agglomerative clustering.

    Agglomerative clustering builds a hierarchy of nested clusters by
    repeatedly merging the two closest clusters (Ward linkage minimizes
    within-cluster variance at each merge step).

    Because Ward linkage requires Euclidean distances on a dense matrix, a
    Truncated SVD projection is applied to the sparse TF-IDF input before
    clustering.  The SVD step also reduces noise in the high-dimensional
    term space.

    Args:
        svd_dimension: Number of SVD components to retain before clustering.
        random_seed: Random seed for repeatable SVD projection.
    """

    def __init__(self, svd_dimension: int = 100, random_seed: int = 42) -> None:
        """Creates a configured agglomerative clusterer.

        Args:
            svd_dimension: Number of SVD components to retain before clustering.
            random_seed: Random seed for repeatable SVD projection.
        """
        self.svd_dimension = svd_dimension
        self.random_seed = random_seed

    def run_clustering(
        self,
        tfidf_matrix: spmatrix | np.ndarray,
        preferred_cluster_count: int | None = None,
        candidate_cluster_counts: Sequence[int] = (4, 6, 8, 10),
    ) -> ClusteringResult:
        """Runs agglomerative clustering and returns labels.

        Args:
            tfidf_matrix: Sparse or dense TF-IDF feature matrix.
            preferred_cluster_count: Optional fixed cluster count to use.
            candidate_cluster_counts: Candidate counts evaluated when no
                fixed count is given; the one with the highest silhouette
                score is selected.

        Returns:
            ClusteringResult: Cluster labels and the chosen cluster count.
        """
        dense_matrix = self._project_to_dense(tfidf_matrix)

        selected_cluster_count = preferred_cluster_count or self._select_cluster_count(
            dense_matrix=dense_matrix,
            candidate_cluster_counts=candidate_cluster_counts,
        )

        clustering_model = AgglomerativeClustering(
            n_clusters=selected_cluster_count,
            linkage="ward",
        )
        predicted_labels = clustering_model.fit_predict(dense_matrix)
        return ClusteringResult(labels=predicted_labels, selected_cluster_count=selected_cluster_count)

    def _project_to_dense(self, tfidf_matrix: spmatrix | np.ndarray) -> np.ndarray:
        """Projects a sparse TF-IDF matrix to a dense SVD embedding.

        If the input is already dense, it is returned as-is.

        Args:
            tfidf_matrix: Sparse or dense feature matrix.

        Returns:
            np.ndarray: Dense matrix with at most `svd_dimension` columns.
        """
        if isinstance(tfidf_matrix, np.ndarray):
            return tfidf_matrix

        # TruncatedSVD requires n_components < n_features, so cap at shape[1]-1.
        reduced_dimension = min(self.svd_dimension, max(1, tfidf_matrix.shape[1] - 1))
        projector = TruncatedSVD(n_components=reduced_dimension, random_state=self.random_seed)
        return projector.fit_transform(tfidf_matrix)

    def _select_cluster_count(
        self,
        dense_matrix: np.ndarray,
        candidate_cluster_counts: Sequence[int],
    ) -> int:
        """Selects the best cluster count using silhouette score.

        Args:
            dense_matrix: Dense feature matrix for silhouette evaluation.
            candidate_cluster_counts: Candidate cluster counts to evaluate.

        Returns:
            int: Cluster count with the highest silhouette score.
        """
        document_count = dense_matrix.shape[0]
        filtered_cluster_counts = [count for count in candidate_cluster_counts if 2 <= count < document_count]
        if not filtered_cluster_counts:
            return 2 if document_count >= 2 else 1

        best_cluster_count = filtered_cluster_counts[0]
        best_silhouette_score = -1.0

        for candidate_cluster_count in filtered_cluster_counts:
            candidate_model = AgglomerativeClustering(
                n_clusters=candidate_cluster_count,
                linkage="ward",
            )
            candidate_labels = candidate_model.fit_predict(dense_matrix)

            if len(set(candidate_labels)) <= 1:
                continue

            candidate_silhouette_score = silhouette_score(dense_matrix, candidate_labels)
            if candidate_silhouette_score > best_silhouette_score:
                best_silhouette_score = candidate_silhouette_score
                best_cluster_count = candidate_cluster_count

        return best_cluster_count
