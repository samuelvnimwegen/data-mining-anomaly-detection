"""Pipeline orchestration for clustering and anomaly detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_mining_assignment.core.data_io import ArticleDataset, save_anomalies, save_clusters
from data_mining_assignment.core.paths import PipelinePaths
from data_mining_assignment.tasks.anomaly_detection import TextAnomalyDetector, create_anomaly_output
from data_mining_assignment.tasks.clustering import TextClusterer, create_cluster_output
from data_mining_assignment.tasks.preprocessing import TextNormalizer, TextPreprocessor


class AssignmentPipeline:
    """Orchestrates preprocessing, clustering, and anomaly detection.

    Args:
        pipeline_paths: Input and output paths.
        preferred_cluster_count: Optional fixed cluster count.
        contamination_ratio: Expected anomaly ratio for Isolation Forest.
        random_seed: Random seed for repeatable model behavior.
    """

    def __init__(
        self,
        pipeline_paths: PipelinePaths,
        preferred_cluster_count: int | None = None,
        contamination_ratio: float = 0.02,
        random_seed: int = 42,
    ) -> None:
        """Sets up all pipeline components.

        Args:
            pipeline_paths: Input and output paths.
            preferred_cluster_count: Optional fixed cluster count.
            contamination_ratio: Expected anomaly ratio for Isolation Forest.
            random_seed: Random seed for repeatable model behavior.
        """
        self.pipeline_paths = pipeline_paths
        self.preferred_cluster_count = preferred_cluster_count

        self.dataset = ArticleDataset(input_csv_path=self.pipeline_paths.input_articles_csv)
        self.normalizer = TextNormalizer()

        self.clustering_preprocessor = TextPreprocessor(
            max_features=20000,
            min_document_frequency=1,
            max_document_frequency=0.92,
            ngram_range=(1, 2),
            analyzer_mode="word",
        )
        self.anomaly_preprocessor = TextPreprocessor(
            max_features=25000,
            min_document_frequency=1,
            max_document_frequency=1.0,
            ngram_range=(3, 5),
            analyzer_mode="char_wb",
        )

        self.clusterer = TextClusterer(random_seed=random_seed)
        self.anomaly_detector = TextAnomalyDetector(
            contamination_ratio=contamination_ratio,
            random_seed=random_seed,
        )

        self._cached_articles_data_frame: pd.DataFrame | None = None
        self._cached_clustering_tfidf_matrix = None
        self._cached_anomaly_tfidf_matrix = None

    @classmethod
    def from_project_root(
        cls,
        project_root_path: Path,
        preferred_cluster_count: int | None = None,
        contamination_ratio: float = 0.02,
        random_seed: int = 42,
    ) -> AssignmentPipeline:
        """Creates a pipeline from a repository root path.

        Args:
            project_root_path: Absolute path to the repository root.
            preferred_cluster_count: Optional fixed cluster count.
            contamination_ratio: Expected anomaly ratio for Isolation Forest.
            random_seed: Random seed for repeatable model behavior.

        Returns:
            AssignmentPipeline: Configured pipeline object.
        """
        return cls(
            pipeline_paths=PipelinePaths.from_project_root(project_root_path),
            preferred_cluster_count=preferred_cluster_count,
            contamination_ratio=contamination_ratio,
            random_seed=random_seed,
        )

    def run_clustering(self) -> pd.DataFrame:
        """Runs clustering and saves `outputs/clusters.csv`.

        Returns:
            pd.DataFrame: Cluster output rows.
        """
        self._ensure_features_ready()
        assert self._cached_articles_data_frame is not None
        assert self._cached_clustering_tfidf_matrix is not None

        clustering_result = self.clusterer.run_clustering(
            tfidf_matrix=self._cached_clustering_tfidf_matrix,
            preferred_cluster_count=self.preferred_cluster_count,
        )
        cluster_output_data_frame = create_cluster_output(
            document_ids=self._cached_articles_data_frame["doc_id"].tolist(),
            labels=clustering_result.labels,
        )
        save_clusters(cluster_output_data_frame, self.pipeline_paths.output_clusters_csv)
        return cluster_output_data_frame

    def run_anomaly_detection(self) -> pd.DataFrame:
        """Runs anomaly detection and saves `outputs/anomalies.csv`.

        Returns:
            pd.DataFrame: Anomaly output rows.
        """
        self._ensure_features_ready()
        assert self._cached_articles_data_frame is not None
        assert self._cached_anomaly_tfidf_matrix is not None

        anomaly_mask, anomaly_scores = self.anomaly_detector.run_detection(self._cached_anomaly_tfidf_matrix)
        anomaly_output_data_frame = create_anomaly_output(
            document_ids=self._cached_articles_data_frame["doc_id"].tolist(),
            anomaly_mask=anomaly_mask,
            anomaly_scores=anomaly_scores,
        )
        save_anomalies(anomaly_output_data_frame, self.pipeline_paths.output_anomalies_csv)
        return anomaly_output_data_frame

    def run_full(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs clustering and anomaly detection in one call.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Cluster and anomaly outputs.
        """
        cluster_output_data_frame = self.run_clustering()
        anomaly_output_data_frame = self.run_anomaly_detection()
        return cluster_output_data_frame, anomaly_output_data_frame

    def _ensure_features_ready(self) -> None:
        """Loads data and creates both clustering and anomaly features."""
        if self._cached_articles_data_frame is None:
            self._cached_articles_data_frame = self.dataset.load_articles()

        if self._cached_clustering_tfidf_matrix is None or self._cached_anomaly_tfidf_matrix is None:
            normalized_text_bundle = self.normalizer.normalize_for_both_tasks(
                self._cached_articles_data_frame["text"].tolist()
            )
            self._cached_clustering_tfidf_matrix = self.clustering_preprocessor.fit_transform(
                normalized_text_bundle.clustering_texts
            )
            self._cached_anomaly_tfidf_matrix = self.anomaly_preprocessor.fit_transform(
                normalized_text_bundle.anomaly_texts
            )
