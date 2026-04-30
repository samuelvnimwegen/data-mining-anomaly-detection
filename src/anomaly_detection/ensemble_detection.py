"""Ensemble anomaly detector combining Isolation Forest, LOF, and kNN."""

from __future__ import annotations

import numpy as np

from anomaly_detection.isolation_forest_detection import TextAnomalyDetector
from anomaly_detection.knn_detection import KNNAnomalyDetector
from anomaly_detection.lof_detection import LocalOutlierFactorDetector


class EnsembleAnomalyDetector:
    """Combines Isolation Forest, LOF, and kNN scores via rank averaging.

    Each detector produces a raw score where lower values indicate stronger
    anomaly evidence.  Before combining, each score array is converted to
    normalised ranks in [0, 1], so that no single detector can dominate the
    final ranking regardless of score magnitude or scale.

    Args:
        contamination_ratio: Expected ratio of anomalies in the corpus.
        n_neighbors_lof: Number of neighbors passed to the LOF detector.
        n_neighbors_knn: Number of neighbors passed to the kNN detector.
        random_seed: Random seed forwarded to the Isolation Forest component.
    """

    def __init__(
        self,
        contamination_ratio: float = 0.02,
        n_neighbors_lof: int = 20,
        n_neighbors_knn: int = 5,
        random_seed: int = 42,
    ) -> None:
        """Creates all three component detectors.

        Args:
            contamination_ratio: Expected ratio of anomalies in the corpus.
            n_neighbors_lof: Number of neighbors passed to the LOF detector.
            n_neighbors_knn: Number of neighbors passed to the kNN detector.
            random_seed: Random seed forwarded to the Isolation Forest component.
        """
        self.contamination_ratio = contamination_ratio
        self.isolation_forest_detector = TextAnomalyDetector(
            contamination_ratio=contamination_ratio,
            random_seed=random_seed,
        )
        self.lof_detector = LocalOutlierFactorDetector(
            n_neighbors=n_neighbors_lof,
            contamination_ratio=contamination_ratio,
        )
        self.knn_detector = KNNAnomalyDetector(
            n_neighbors=n_neighbors_knn,
            contamination_ratio=contamination_ratio,
        )

    def run_detection(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runs all three detectors and returns rank-averaged ensemble scores.

        Args:
            feature_matrix: Dense feature matrix with one row per document.

        Returns:
            tuple[np.ndarray, np.ndarray]: Boolean anomaly mask and ensemble scores.
                Scores use the convention that lower values are more anomalous.
        """
        _, if_scores = self.isolation_forest_detector.run_detection(feature_matrix)
        _, lof_scores = self.lof_detector.run_detection(feature_matrix)
        _, knn_scores = self.knn_detector.run_detection(feature_matrix)

        ensemble_scores = self._rank_average([if_scores, lof_scores, knn_scores])

        threshold_value = np.quantile(ensemble_scores, self.contamination_ratio)
        detected_anomaly_mask = ensemble_scores <= threshold_value
        return detected_anomaly_mask, ensemble_scores

    def _rank_average(self, score_arrays: list[np.ndarray]) -> np.ndarray:
        """Normalises each score array by rank and returns the element-wise mean.

        All input arrays use the convention that lower scores are more
        anomalous, and the returned array preserves that convention.

        Args:
            score_arrays: List of score arrays with identical length.

        Returns:
            np.ndarray: Rank-averaged scores where lower = more anomalous.
        """
        document_count = len(score_arrays[0])
        normalised_arrays: list[np.ndarray] = []

        for scores in score_arrays:
            ascending_order = np.argsort(scores)
            ranks = np.empty(document_count, dtype=float)
            ranks[ascending_order] = np.arange(document_count, dtype=float)
            normalised_arrays.append(ranks / max(document_count - 1, 1))

        return np.mean(normalised_arrays, axis=0)
