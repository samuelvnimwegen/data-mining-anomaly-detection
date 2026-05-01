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
        output_bag_of_words_csv: Destination CSV for the bag-of-words matrix.
        processed_text_views_csv: Cached normalized text views.
        processed_clustering_matrix_npz: Cached clustering sparse matrix.
        processed_anomaly_matrix_npy: Cached anomaly dense matrix.
    """

    input_articles_csv: Path
    output_clusters_csv: Path
    output_anomalies_csv: Path
    output_bag_of_words_csv: Path
    processed_text_views_csv: Path | None = None
    processed_clustering_matrix_npz: Path | None = None
    processed_anomaly_matrix_npy: Path | None = None
    processed_sentence_embeddings_npy: Path | None = None

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
            output_bag_of_words_csv=project_root_path / "data" / "results" / "bag_of_words_matrix.csv",
            processed_text_views_csv=project_root_path / "data" / "processed" / "normalized_text_views.csv",
            processed_clustering_matrix_npz=project_root_path / "data" / "processed" / "clustering_tfidf_matrix.npz",
            processed_anomaly_matrix_npy=project_root_path / "data" / "processed" / "anomaly_lsa_matrix.npy",
            processed_sentence_embeddings_npy=project_root_path / "data" / "processed" / "sentence_embeddings.npy",
        )
