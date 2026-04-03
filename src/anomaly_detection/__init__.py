"""Task 3 anomaly detection package."""

from data_mining_assignment.tasks.anomaly_detection.isolation_forest_detection import (
    TextAnomalyDetector,
    create_anomaly_output,
)

__all__ = ["TextAnomalyDetector", "create_anomaly_output"]
