"""Project path configuration for Assignment 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelinePaths:
    """Stores all main file paths used by the pipeline.

    Attributes:
        input_articles_csv: Source CSV with document id and text.
        output_clusters_csv: Destination CSV for cluster labels.
        output_anomalies_csv: Destination CSV for anomaly ranking.
    """

    input_articles_csv: Path
    output_clusters_csv: Path
    output_anomalies_csv: Path

    @classmethod
    def from_project_root(cls, project_root_path: Path) -> PipelinePaths:
        """Builds default paths from the project root.

        Args:
            project_root_path: Absolute path to the repository root.

        Returns:
            PipelinePaths: Default path setup for this project.
        """
        return cls(
            input_articles_csv=project_root_path / "data" / "raw" / "articles.csv",
            output_clusters_csv=project_root_path / "data" / "results" / "clusters.csv",
            output_anomalies_csv=project_root_path / "data" / "results" / "anomalies.csv",
        )
