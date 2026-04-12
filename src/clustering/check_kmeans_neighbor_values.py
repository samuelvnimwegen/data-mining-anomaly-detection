"""Find neighbor values that create a target listing-like cluster.

This script scans neighbor values from 6 to 10 for Spectral Clustering.
The clustering step uses `assign_labels="kmeans"`, so labels are still built
with a k-means step after the graph embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.cluster import SpectralClustering

from core.data_io import ArticleDataset
from preprocessing import TextNormalizer, TextPreprocessor


@dataclass(slots=True)
class NeighborScanConfig:
    """Store settings for the neighbor scan.

    Args:
        input_csv_path: Path to the `articles.csv` file.
        output_csv_path: Path where the scan output is saved.
        start_neighbor_value: Smallest neighbor value to test.
        end_neighbor_value: Largest neighbor value to test.
        cluster_count: Number of clusters used in each run.
        min_cluster_size: Smallest allowed size for the target cluster.
        max_cluster_size: Largest allowed size for the target cluster.
        listing_prefix: Prefix used to detect listing-like rows.
        min_listing_rows: Minimum listing-like rows inside one target cluster.
        random_seed: Random seed for repeatable clustering.
    """

    input_csv_path: Path
    output_csv_path: Path
    start_neighbor_value: int = 6
    end_neighbor_value: int = 10
    cluster_count: int = 10
    min_cluster_size: int = 40
    max_cluster_size: int = 60
    listing_prefix: str = "LISTING_ID_"
    min_listing_rows: int = 10
    random_seed: int = 42


def scan_neighbor_values(config: NeighborScanConfig) -> pd.DataFrame:
    """Scan neighbor values and save matches.

    A match is a run where at least one cluster has:
    - Size between `min_cluster_size` and `max_cluster_size`.
    - At least `min_listing_rows` rows that start with `listing_prefix`.

    Args:
        config: Full scan configuration.

    Returns:
        pd.DataFrame: One row per matching cluster.
    """
    article_data_frame = ArticleDataset(input_csv_path=config.input_csv_path).load_articles()

    text_normalizer = TextNormalizer()
    normalized_bundle = text_normalizer.normalize_for_both_tasks(article_data_frame["text"].tolist())

    vectorizer = TextPreprocessor(
        vectorization_model_name="tfidf_lsa_dense",
        max_features=20000,
        min_document_frequency=1,
        max_document_frequency=0.92,
        ngram_range=(1, 2),
        analyzer_mode="word",
        dense_embedding_dimension=128,
        random_seed=config.random_seed,
    )
    dense_feature_matrix = vectorizer.fit_transform(normalized_bundle.clustering_texts)

    stripped_text_series = article_data_frame["text"].astype(str).str.lstrip()
    listing_mask = stripped_text_series.str.startswith(config.listing_prefix)

    matching_rows: list[dict[str, int | str]] = []

    for neighbor_value in range(config.start_neighbor_value, config.end_neighbor_value + 1):
        # Spectral has n_neighbors and then assigns final labels with k-means.
        spectral_model = SpectralClustering(
            n_clusters=config.cluster_count,
            affinity="nearest_neighbors",
            n_neighbors=neighbor_value,
            assign_labels="kmeans",
            random_state=config.random_seed,
        )
        cluster_labels = spectral_model.fit_predict(dense_feature_matrix)

        label_series = pd.Series(cluster_labels, dtype="int64")
        for cluster_label in sorted(label_series.unique().tolist()):
            cluster_member_mask = label_series == cluster_label
            cluster_size = int(cluster_member_mask.sum())
            if not (config.min_cluster_size <= cluster_size <= config.max_cluster_size):
                continue

            listing_count_in_cluster = int((cluster_member_mask & listing_mask).sum())
            if listing_count_in_cluster < config.min_listing_rows:
                continue

            matching_rows.append(
                {
                    "neighbors_used": neighbor_value,
                    "cluster_label": int(cluster_label),
                    "cluster_size": cluster_size,
                    "listing_rows_in_cluster": listing_count_in_cluster,
                }
            )

    matching_data_frame = pd.DataFrame(matching_rows)
    config.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if matching_data_frame.empty:
        matching_data_frame = pd.DataFrame(
            columns=["neighbors_used", "cluster_label", "cluster_size", "listing_rows_in_cluster"]
        )

    matching_data_frame.to_csv(config.output_csv_path, index=False)
    return matching_data_frame


def main() -> None:
    """Run the neighbor scan with project defaults."""
    project_root_path = Path(__file__).resolve().parents[2]
    scan_config = NeighborScanConfig(
        input_csv_path=project_root_path / "data" / "raw" / "articles.csv",
        output_csv_path=project_root_path / "data" / "results" / "kmeans_neighbor_scan_matches.csv",
    )

    matching_data_frame = scan_neighbor_values(scan_config)
    if matching_data_frame.empty:
        print("No neighbor value from 6 to 10 matched the cluster rule.")
        return

    matching_neighbor_values = sorted(matching_data_frame["neighbors_used"].unique().tolist())
    print("Matching neighbor values:", ", ".join(str(value) for value in matching_neighbor_values))
    print(f"Saved details to: {scan_config.output_csv_path}")


if __name__ == "__main__":
    main()

