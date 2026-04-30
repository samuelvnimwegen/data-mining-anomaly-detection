"""Local Outlier Factor anomaly detection for text documents."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class LocalOutlierFactorDetector:
    """Finds anomalous documents using Local Outlier Factor.

    LOF compares the local density of each document to the local density of
    its neighbors.  Documents in sparse neighborhoods score much lower than
    their neighbors and are flagged as outliers.

    Args:
        n_neighbors: Number of neighbors used for density estimation.
        contamination_ratio: Expected ratio of anomalies in the corpus.
    """

    def __init__(self, n_neighbors: int = 20, contamination_ratio: float = 0.02) -> None:
        """Creates a configured LOF detector.

        Args:
            n_neighbors: Number of neighbors used for density estimation.
            contamination_ratio: Expected ratio of anomalies in the corpus.
        """
        self.n_neighbors = n_neighbors
        self.contamination_ratio = contamination_ratio
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination_ratio,
        )

    def run_detection(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runs LOF anomaly detection on a dense feature matrix.

        Args:
            feature_matrix: Dense feature matrix with one row per document.

        Returns:
            tuple[np.ndarray, np.ndarray]: Boolean anomaly mask and LOF scores.
                Scores follow the convention that lower values are more anomalous,
                matching the Isolation Forest score convention.
        """
        predicted_labels = self.model.fit_predict(feature_matrix)

        # LOF labels: -1 marks outliers, 1 marks inliers.
        detected_anomaly_mask = predicted_labels == -1

        # negative_outlier_factor_ is negative LOF; more negative = more anomalous.
        anomaly_scores = self.model.negative_outlier_factor_
        return detected_anomaly_mask, anomaly_scores
