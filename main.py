"""Repository-level entrypoint for Assignment 3 tasks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_ROOT_PATH / "src"

if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))

from core import AssignmentPipeline  # noqa: E402


def main() -> None:
    """Runs selected assignment tasks.

    Comment or uncomment individual lines to run specific tasks.

    Clustering methods:
        "kmeans"       — K-Means with silhouette-based cluster count selection.
        "agglomerative"— Ward-linkage hierarchical clustering (SVD-reduced).

    Anomaly detection:
        use_ensemble=False — Isolation Forest on structural features (default,
                             best single-method performance).
        use_ensemble=True  — Rank-average ensemble of Isolation Forest, LOF,
                             and kNN on structural features.
    """
    assignment_pipeline = AssignmentPipeline.from_project_root(project_root_path=PROJECT_ROOT_PATH)

    # Task 2: clustering.
    # Option A — sentence embeddings (submitted result, silhouette 0.062).
    #   Requires notebook 09 to have been run first to cache the embeddings.
    assignment_pipeline.run_clustering(use_embeddings=True)
    # Option B — TF-IDF K-Means (baseline).
    # assignment_pipeline.run_clustering(clustering_method="kmeans")
    # Option C — TF-IDF Agglomerative (best TF-IDF silhouette).
    # assignment_pipeline.run_clustering(clustering_method="agglomerative")

    # Task 3: anomaly detection — Isolation Forest on structural features.
    assignment_pipeline.run_anomaly_detection(use_ensemble=False)
    # assignment_pipeline.run_anomaly_detection(use_ensemble=True)  # ensemble

    # Task 1: bag-of-words export.
    assignment_pipeline.run_bag_of_words_export()

    # Run everything in one call (uncomment to use).
    # assignment_pipeline.run_full(use_embeddings=True, use_ensemble=False)


if __name__ == "__main__":
    main()
