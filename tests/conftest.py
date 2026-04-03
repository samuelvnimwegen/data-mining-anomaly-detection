"""Pytest setup for Assignment 3 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT_PATH / "src"

if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))
