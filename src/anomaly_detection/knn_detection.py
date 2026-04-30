"""k-Nearest Neighbors distance-based anomaly detection for text documents."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNAnomalyDetector:
    """Finds anomalous documents using mean k-NN cosine distance scoring.

    Documents that are far from all of their nearest neighbors (i.e., they
    sit in a sparse region of the feature space) receive high anomaly scores.
    This is a simple but effective distance-based outlier method.

    Args:
        n_neighbors: Number of neighbors used for distance scoring.
        contamination_ratio: Expected ratio of anomalies in the corpus.
    """

    def __init__(self, n_neighbors: int = 5, contamination_ratio: float = 0.02) -> None:
        """Creates a configured kNN anomaly detector.

        Args:
            n_neighbors: Number of neighbors used for distance scoring.
            contamination_ratio: Expected ratio of anomalies in the corpus.
        """
        self.n_neighbors = n_neighbors
        self.contamination_ratio = contamination_ratio

    def run_detection(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runs kNN anomaly scoring on a dense feature matrix.

        Args:
            feature_matrix: Dense feature matrix with one row per document.

        Returns:
            tuple[np.ndarray, np.ndarray]: Boolean anomaly mask and anomaly scores.
                Scores follow the convention that lower values are more anomalous,
                so the returned scores are the negated mean distances.
        """
        # Request n_neighbors + 1 because the query point is its own neighbour.
        nn_model = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric="cosine")
        nn_model.fit(feature_matrix)
        distances, _ = nn_model.kneighbors(feature_matrix)

        # Column 0 is always distance 0 (self), so exclude it.
        mean_distances = distances[:, 1:].mean(axis=1)

        threshold_value = np.quantile(mean_distances, 1.0 - self.contamination_ratio)
        detected_anomaly_mask = mean_distances >= threshold_value

        # Negate so that the lowest score = most anomalous, matching IF/LOF.
        anomaly_scores = -mean_distances
        return detected_anomaly_mask, anomaly_scores
