"""Anomaly detection logic for unsafe or rare articles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.ensemble import IsolationForest


class TextAnomalyDetector:
    """Finds anomalous documents using Isolation Forest.

    Args:
        contamination_ratio: Expected ratio of anomalies.
        random_seed: Random seed for repeatable model behavior.
    """

    def __init__(self, contamination_ratio: float = 0.02, random_seed: int = 42) -> None:
        """Sets anomaly detector settings.

        Args:
            contamination_ratio: Expected ratio of anomalies.
            random_seed: Random seed for repeatable model behavior.
        """
        self.contamination_ratio = contamination_ratio
        self.random_seed = random_seed
        self.model = IsolationForest(
            contamination=self.contamination_ratio,
            random_state=self.random_seed,
        )

    def run_detection(self, tfidf_matrix: spmatrix | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runs anomaly detection on vectorized text.

        Args:
            tfidf_matrix: Sparse or dense feature matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]: Anomaly mask and anomaly scores.
        """
        predicted_labels = self.model.fit_predict(tfidf_matrix)

        # IsolationForest uses -1 for anomalies and 1 for normal rows.
        detected_anomaly_mask = predicted_labels == -1

        # Lower scores mean more abnormal rows.
        anomaly_scores = self.model.score_samples(tfidf_matrix)
        return detected_anomaly_mask, anomaly_scores


def create_anomaly_output(
    document_ids: Sequence[str],
    anomaly_mask: np.ndarray,
    anomaly_scores: np.ndarray,
    expected_anomaly_count: int | None = None,
) -> pd.DataFrame:
    """Builds the anomaly output DataFrame.

    Args:
        document_ids: Ordered document ids.
        anomaly_mask: Boolean mask where True marks anomalies.
        anomaly_scores: Isolation Forest score per document.
        expected_anomaly_count: Optional fixed number of rows to output.

    Returns:
        pd.DataFrame: DataFrame with `anomaly` rank and `doc_id`.
    """
    anomaly_rows = pd.DataFrame(
        {
            "doc_id": list(document_ids),
            "is_anomaly": anomaly_mask,
            "score": anomaly_scores,
        }
    )

    # Sort by score first so deterministic top-k selection is possible.
    ranked_rows = anomaly_rows.sort_values("score", ascending=True).reset_index(drop=True)

    if expected_anomaly_count is not None:
        selected_row_count = max(0, min(expected_anomaly_count, len(ranked_rows)))
        selected_rows = ranked_rows.head(selected_row_count).copy()
    else:
        selected_rows = ranked_rows[ranked_rows["is_anomaly"]].copy()

    if selected_rows.empty:
        return pd.DataFrame(columns=["anomaly", "doc_id"])

    selected_rows = selected_rows.reset_index(drop=True)
    selected_rows["anomaly"] = selected_rows.index + 1
    return selected_rows[["anomaly", "doc_id"]]
