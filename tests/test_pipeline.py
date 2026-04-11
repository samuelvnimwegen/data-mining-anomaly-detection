"""Tests for the Assignment 3 pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from core import AssignmentPipeline, PipelinePaths


def test_pipeline_writes_cluster_and_anomaly_files(tmp_path: Path) -> None:
    """Checks that the full pipeline writes both assignment outputs.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    input_data_frame = pd.DataFrame(
        {
            "doc_id": [
                "DOC_001",
                "DOC_002",
                "DOC_003",
                "DOC_004",
                "DOC_005",
                "DOC_006",
            ],
            "text": [
                "Football match and team tactics.",
                "Basketball league and player stats.",
                "Deep learning models for image tasks.",
                "Neural network training and optimization.",
                "Healthy recipe with vegetables and spices.",
                "Cooking tips and kitchen planning.",
            ],
        }
    )

    input_csv_path = tmp_path / "articles.csv"
    clusters_csv_path = tmp_path / "clusters.csv"
    anomalies_csv_path = tmp_path / "anomalies.csv"
    input_data_frame.to_csv(input_csv_path, index=False)

    pipeline_paths = PipelinePaths(
        input_articles_csv=input_csv_path,
        output_clusters_csv=clusters_csv_path,
        output_anomalies_csv=anomalies_csv_path,
        output_bag_of_words_csv=tmp_path / "bag_of_words_matrix.csv",
    )

    assignment_pipeline = AssignmentPipeline(
        pipeline_paths=pipeline_paths,
        preferred_cluster_count=3,
        contamination_ratio=0.2,
        random_seed=42,
    )
    cluster_output_data_frame, anomaly_output_data_frame = assignment_pipeline.run_full()

    assert clusters_csv_path.exists()
    assert anomalies_csv_path.exists()
    assert len(cluster_output_data_frame) == len(input_data_frame)
    assert {"doc_id", "label"}.issubset(cluster_output_data_frame.columns)
    assert {"anomaly", "doc_id"}.issubset(anomaly_output_data_frame.columns)


def test_pipeline_supports_custom_text_column_names(tmp_path: Path) -> None:
    """Checks that loading works when column names are not standard.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    input_data_frame = pd.DataFrame(
        {
            "document_id": ["A", "B", "C"],
            "content": ["Alpha beta", "Gamma delta", "Epsilon zeta"],
        }
    )
    input_csv_path = tmp_path / "articles.csv"
    input_data_frame.to_csv(input_csv_path, index=False)

    pipeline_paths = PipelinePaths(
        input_articles_csv=input_csv_path,
        output_clusters_csv=tmp_path / "clusters.csv",
        output_anomalies_csv=tmp_path / "anomalies.csv",
        output_bag_of_words_csv=tmp_path / "bag_of_words_matrix.csv",
    )
    assignment_pipeline = AssignmentPipeline(
        pipeline_paths=pipeline_paths,
        preferred_cluster_count=2,
        contamination_ratio=0.34,
    )

    cluster_output_data_frame = assignment_pipeline.run_clustering()

    assert cluster_output_data_frame["doc_id"].tolist() == ["A", "B", "C"]


def test_pipeline_exports_bag_of_words_with_popular_terms_first(tmp_path: Path) -> None:
    """Checks bag-of-words export writes term columns sorted by popularity."""
    input_data_frame = pd.DataFrame(
        {
            "doc_id": ["D1", "D2", "D3"],
            "text": [
                "apple banana",
                "apple carrot",
                "apple",
            ],
        }
    )
    input_csv_path = tmp_path / "articles.csv"
    bag_of_words_csv_path = tmp_path / "bag_of_words_matrix.csv"
    input_data_frame.to_csv(input_csv_path, index=False)

    pipeline_paths = PipelinePaths(
        input_articles_csv=input_csv_path,
        output_clusters_csv=tmp_path / "clusters.csv",
        output_anomalies_csv=tmp_path / "anomalies.csv",
        output_bag_of_words_csv=bag_of_words_csv_path,
    )
    assignment_pipeline = AssignmentPipeline(
        pipeline_paths=pipeline_paths,
        preferred_cluster_count=2,
        contamination_ratio=0.34,
    )

    export_metadata_data_frame = assignment_pipeline.run_bag_of_words_export()

    assert bag_of_words_csv_path.exists()
    assert export_metadata_data_frame.loc[0, "document_count"] == 3

    with bag_of_words_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        first_row = next(csv_reader)

    assert header[:3] == ["doc_id", "apple", "banana"]
    assert first_row[0] == "D1"
    assert first_row[1] == "1"
    assert first_row[2] == "1"
