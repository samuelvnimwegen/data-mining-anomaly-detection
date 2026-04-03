"""Tests for default project path configuration."""

from __future__ import annotations

from pathlib import Path

from data_mining_assignment.core.paths import PipelinePaths


def test_pipeline_paths_use_raw_and_processed_folders() -> None:
    """Checks default path mapping points to raw input and processed caches."""
    project_root_path = Path("/tmp/example-project")
    pipeline_paths = PipelinePaths.from_project_root(project_root_path)

    assert pipeline_paths.input_articles_csv == project_root_path / "data" / "raw" / "articles.csv"
    assert (
        pipeline_paths.processed_text_views_csv
        == project_root_path / "data" / "processed" / "normalized_text_views.csv"
    )
    assert (
        pipeline_paths.processed_clustering_matrix_npz
        == project_root_path / "data" / "processed" / "clustering_tfidf_matrix.npz"
    )
    assert (
        pipeline_paths.processed_anomaly_matrix_npy
        == project_root_path / "data" / "processed" / "anomaly_lsa_matrix.npy"
    )
