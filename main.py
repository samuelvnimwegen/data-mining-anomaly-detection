"""Repository-level entrypoint for Assignment 3 tasks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_ROOT_PATH / "src"

if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))

from data_mining_assignment.core import AssignmentPipeline  # noqa: E402


def main() -> None:
    """Runs selected assignment tasks.

    Use one-line comments to toggle tasks.
    """
    assignment_pipeline = AssignmentPipeline.from_project_root(project_root_path=PROJECT_ROOT_PATH)

    # Comment or uncomment these lines to run single tasks.
    assignment_pipeline.run_clustering()
    assignment_pipeline.run_anomaly_detection()
    assignment_pipeline.run_bag_of_words_export()
    # assignment_pipeline.run_full()


if __name__ == "__main__":
    main()
