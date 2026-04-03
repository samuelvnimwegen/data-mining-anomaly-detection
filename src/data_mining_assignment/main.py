"""CLI style entrypoint for Assignment 3 tasks."""

from __future__ import annotations

from pathlib import Path

from data_mining_assignment.core import AssignmentPipeline


def main() -> None:
    """Runs selected assignment tasks.

    Use one-line comments to toggle tasks.
    """
    project_root_path = Path(__file__).resolve().parents[2]
    assignment_pipeline = AssignmentPipeline.from_project_root(project_root_path=project_root_path)

    # Comment or uncomment these lines to run single tasks.
    assignment_pipeline.run_clustering()
    assignment_pipeline.run_anomaly_detection()
    # assignment_pipeline.run_full()


if __name__ == "__main__":
    main()
