"""Tests for anomaly output ranking behavior."""

from __future__ import annotations

import numpy as np

from anomaly_detection import create_anomaly_output


def test_create_anomaly_output_respects_expected_count() -> None:
    """Checks deterministic top-k output size and rank order."""
    document_ids = ["A", "B", "C", "D"]
    anomaly_mask = np.array([False, True, False, True])
    anomaly_scores = np.array([-0.1, -0.4, -0.3, -0.2], dtype=float)

    output_data_frame = create_anomaly_output(
        document_ids=document_ids,
        anomaly_mask=anomaly_mask,
        anomaly_scores=anomaly_scores,
        expected_anomaly_count=3,
    )

    assert output_data_frame["doc_id"].tolist() == ["B", "C", "D"]
    assert output_data_frame["anomaly"].tolist() == [1, 2, 3]


def test_create_anomaly_output_falls_back_to_mask_when_count_is_none() -> None:
    """Checks mask-based output path still works."""
    document_ids = ["A", "B", "C"]
    anomaly_mask = np.array([False, True, False])
    anomaly_scores = np.array([-0.3, -0.1, -0.2], dtype=float)

    output_data_frame = create_anomaly_output(
        document_ids=document_ids,
        anomaly_mask=anomaly_mask,
        anomaly_scores=anomaly_scores,
        expected_anomaly_count=None,
    )

    assert output_data_frame["doc_id"].tolist() == ["B"]
    assert output_data_frame["anomaly"].tolist() == [1]

