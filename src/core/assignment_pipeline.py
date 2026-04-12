"""Pipeline orchestration for clustering and anomaly detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from anomaly_detection import TextAnomalyDetector, create_anomaly_output
from clustering import TextClusterer, create_cluster_output
from core.data_io import (
    ArticleDataset,
    load_processed_dense_matrix,
    load_processed_sparse_matrix,
    load_processed_text_views,
    save_anomalies,
    save_bag_of_words_matrix_csv,
    save_clusters,
    save_processed_dense_matrix,
    save_processed_sparse_matrix,
    save_processed_text_views,
)
from core.paths import PipelinePaths
from preprocessing import StructuralFeatureExtractor, TextNormalizer, TextPreprocessor


class AssignmentPipeline:
    """Orchestrates preprocessing, clustering, and anomaly detection.

    Args:
        pipeline_paths: Input and output paths.
        preferred_cluster_count: Optional fixed cluster count.
        contamination_ratio: Expected anomaly ratio for Isolation Forest.
        random_seed: Random seed for repeatable model behavior.
        expected_anomaly_count: Fixed number of anomalies to export.
    """

    def __init__(
        self,
        pipeline_paths: PipelinePaths,
        preferred_cluster_count: int | None = None,
        contamination_ratio: float = 0.02,
        random_seed: int = 42,
        expected_anomaly_count: int = 50,
    ) -> None:
        """Sets up all pipeline components.

        Args:
            pipeline_paths: Input and output paths.
            preferred_cluster_count: Optional fixed cluster count.
            contamination_ratio: Expected anomaly ratio for Isolation Forest.
            random_seed: Random seed for repeatable model behavior.
            expected_anomaly_count: Fixed number of anomalies to export.
        """
        self.pipeline_paths = pipeline_paths
        self.preferred_cluster_count = preferred_cluster_count
        self.expected_anomaly_count = expected_anomaly_count

        self.dataset = ArticleDataset(input_csv_path=self.pipeline_paths.input_articles_csv)
        self.normalizer = TextNormalizer()

        self.clustering_preprocessor = TextPreprocessor(
            vectorization_model_name="tfidf",
            max_features=20000,
            min_document_frequency=1,
            max_document_frequency=0.92,
            ngram_range=(1, 2),
            analyzer_mode="word",
        )
        self.anomaly_preprocessor = TextPreprocessor(
            vectorization_model_name="tfidf_lsa_dense",
            max_features=30000,
            min_document_frequency=1,
            max_document_frequency=1.0,
            ngram_range=(3, 5),
            analyzer_mode="char_wb",
            dense_embedding_dimension=256,
            random_seed=random_seed,
        )

        self.structural_feature_extractor = StructuralFeatureExtractor()

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
        expected_anomaly_count: int = 50,
    ) -> AssignmentPipeline:
        """Creates a pipeline from a repository root path.

        Args:
            project_root_path: Absolute path to the repository root.
            preferred_cluster_count: Optional fixed cluster count.
            contamination_ratio: Expected anomaly ratio for Isolation Forest.
            random_seed: Random seed for repeatable model behavior.
            expected_anomaly_count: Fixed number of anomalies to export.

        Returns:
            AssignmentPipeline: Configured pipeline object.
        """
        return cls(
            pipeline_paths=PipelinePaths.from_project_root(project_root_path),
            preferred_cluster_count=preferred_cluster_count,
            contamination_ratio=contamination_ratio,
            random_seed=random_seed,
            expected_anomaly_count=expected_anomaly_count,
        )

    def run_clustering(self) -> pd.DataFrame:
        """Runs clustering and saves `data/results/clusters.csv`.

        Returns:
            pd.DataFrame: Cluster output rows.
        """
        self._ensure_features_ready()
        assert self._cached_articles_data_frame is not None
        assert self._cached_clustering_tfidf_matrix is not None
        articles_data_frame: pd.DataFrame = self._cached_articles_data_frame

        clustering_result = self.clusterer.run_clustering(
            tfidf_matrix=self._cached_clustering_tfidf_matrix,
            preferred_cluster_count=self.preferred_cluster_count,
        )
        cluster_output_data_frame = create_cluster_output(
            document_ids=articles_data_frame["doc_id"].tolist(),
            labels=clustering_result.labels,
        )
        save_clusters(cluster_output_data_frame, self.pipeline_paths.output_clusters_csv)
        return cluster_output_data_frame

    def run_anomaly_detection(self) -> pd.DataFrame:
        """Runs anomaly detection and saves `data/results/anomalies.csv`.

        Returns:
            pd.DataFrame: Anomaly output rows.
        """
        self._ensure_features_ready()
        assert self._cached_articles_data_frame is not None
        assert self._cached_anomaly_tfidf_matrix is not None
        articles_data_frame: pd.DataFrame = self._cached_articles_data_frame

        anomaly_mask, anomaly_scores = self.anomaly_detector.run_detection(self._cached_anomaly_tfidf_matrix)
        anomaly_output_data_frame = create_anomaly_output(
            document_ids=articles_data_frame["doc_id"].tolist(),
            anomaly_mask=anomaly_mask,
            anomaly_scores=anomaly_scores,
            expected_anomaly_count=self.expected_anomaly_count,
        )
        save_anomalies(anomaly_output_data_frame, self.pipeline_paths.output_anomalies_csv)
        return anomaly_output_data_frame

    def run_bag_of_words_export(self) -> pd.DataFrame:
        """Builds and saves a popularity-sorted bag-of-words matrix.

        Returns:
            pd.DataFrame: One-row metadata about the written export.
        """
        if self._cached_articles_data_frame is None:
            self._cached_articles_data_frame = self.dataset.load_articles()

        normalized_text_bundle = self.normalizer.normalize_for_both_tasks(
            self._cached_articles_data_frame["text"].tolist()
        )

        bag_of_words_preprocessor = TextPreprocessor(
            vectorization_model_name="bow",
            max_features=self.clustering_preprocessor.max_features,
            min_document_frequency=1,
            max_document_frequency=1.0,
            ngram_range=(1, 1),
            analyzer_mode="word",
        )

        bag_of_words_matrix = bag_of_words_preprocessor.fit_transform(normalized_text_bundle.clustering_texts)
        if not isinstance(bag_of_words_matrix, spmatrix):
            raise ValueError("Bag-of-words export expects a sparse matrix.")

        feature_names = bag_of_words_preprocessor.get_feature_names()
        save_bag_of_words_matrix_csv(
            document_ids=self._cached_articles_data_frame["doc_id"].tolist(),
            bag_of_words_matrix=bag_of_words_matrix,
            feature_names=feature_names,
            output_csv_path=self.pipeline_paths.output_bag_of_words_csv,
        )
        return pd.DataFrame(
            {
                "output_csv_path": [str(self.pipeline_paths.output_bag_of_words_csv)],
                "document_count": [len(self._cached_articles_data_frame)],
                "term_count": [len(feature_names)],
            }
        )

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

        if self._cached_clustering_tfidf_matrix is not None and self._cached_anomaly_tfidf_matrix is not None:
            return

        if self._can_load_processed_features() and self._load_processed_features():
            return

        normalized_text_bundle = self.normalizer.normalize_for_both_tasks(
            self._cached_articles_data_frame["text"].tolist()
        )
        self._cached_clustering_tfidf_matrix = self.clustering_preprocessor.fit_transform(
            normalized_text_bundle.clustering_texts
        )
        self._cached_anomaly_tfidf_matrix = self.anomaly_preprocessor.fit_transform(
            normalized_text_bundle.anomaly_texts
        )

        if isinstance(self._cached_anomaly_tfidf_matrix, spmatrix):
            dense_anomaly_matrix = self._cached_anomaly_tfidf_matrix.toarray()
        else:
            dense_anomaly_matrix = np.asarray(self._cached_anomaly_tfidf_matrix, dtype=np.float64)

        structural_feature_matrix = self.structural_feature_extractor.transform(
            self._cached_articles_data_frame["text"].astype(str).tolist()
        )
        # Merge semantic and structural views for blind anomaly scoring.
        self._cached_anomaly_tfidf_matrix = np.hstack([dense_anomaly_matrix, structural_feature_matrix])

        self._save_processed_features(
            clustering_texts=normalized_text_bundle.clustering_texts,
            anomaly_texts=normalized_text_bundle.anomaly_texts,
        )

    def _can_load_processed_features(self) -> bool:
        """Checks whether all processed intermediate files exist."""
        if (
            self.pipeline_paths.processed_text_views_csv is None
            or self.pipeline_paths.processed_clustering_matrix_npz is None
            or self.pipeline_paths.processed_anomaly_matrix_npy is None
        ):
            return False

        return (
            self.pipeline_paths.processed_text_views_csv.exists()
            and self.pipeline_paths.processed_clustering_matrix_npz.exists()
            and self.pipeline_paths.processed_anomaly_matrix_npy.exists()
        )

    def _load_processed_features(self) -> bool:
        """Loads processed intermediates from disk.

        Returns:
            bool: True when cache is loaded and valid, else False.
        """
        assert self._cached_articles_data_frame is not None
        assert self.pipeline_paths.processed_text_views_csv is not None
        assert self.pipeline_paths.processed_clustering_matrix_npz is not None
        assert self.pipeline_paths.processed_anomaly_matrix_npy is not None

        articles_data_frame: pd.DataFrame = self._cached_articles_data_frame
        processed_text_views = load_processed_text_views(self.pipeline_paths.processed_text_views_csv)
        current_document_ids = articles_data_frame["doc_id"].tolist()

        if processed_text_views["doc_id"].astype(str).tolist() != current_document_ids:
            # Data changed, so processed cache is stale.
            return False

        self._cached_clustering_tfidf_matrix = load_processed_sparse_matrix(
            self.pipeline_paths.processed_clustering_matrix_npz
        )
        self._cached_anomaly_tfidf_matrix = load_processed_dense_matrix(
            self.pipeline_paths.processed_anomaly_matrix_npy
        )
        return True

    def _save_processed_features(self, clustering_texts: list[str], anomaly_texts: list[str]) -> None:
        """Writes processed intermediates to disk."""
        assert self._cached_articles_data_frame is not None
        assert self._cached_clustering_tfidf_matrix is not None
        assert self._cached_anomaly_tfidf_matrix is not None

        if (
            self.pipeline_paths.processed_text_views_csv is None
            or self.pipeline_paths.processed_clustering_matrix_npz is None
            or self.pipeline_paths.processed_anomaly_matrix_npy is None
        ):
            return

        articles_data_frame: pd.DataFrame = self._cached_articles_data_frame

        save_processed_text_views(
            document_ids=articles_data_frame["doc_id"].tolist(),
            clustering_texts=clustering_texts,
            anomaly_texts=anomaly_texts,
            output_csv_path=self.pipeline_paths.processed_text_views_csv,
        )
        save_processed_sparse_matrix(
            sparse_matrix=self._cached_clustering_tfidf_matrix,
            output_npz_path=self.pipeline_paths.processed_clustering_matrix_npz,
        )
        save_processed_dense_matrix(
            dense_matrix=self._cached_anomaly_tfidf_matrix,
            output_npy_path=self.pipeline_paths.processed_anomaly_matrix_npy,
        )
