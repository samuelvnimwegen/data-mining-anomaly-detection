"""Splits articles.csv by sentence overlap with original txt articles."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SplitSummary:
    """Stores the split result counts.

    Args:
        total_rows: Total rows read from articles.csv.
        matched_rows: Rows with at least one matching sentence.
        non_matched_rows: Rows with zero matching sentences.
    """

    total_rows: int
    matched_rows: int
    non_matched_rows: int


def normalize_text(raw_text: str) -> str:
    """Normalizes text for robust sentence comparison.

    Args:
        raw_text: Text before cleanup.

    Returns:
        str: Lowercased text with compact spaces.
    """
    lowered_text: str = raw_text.lower().strip()
    return re.sub(r"\s+", " ", lowered_text)


def extract_normalized_sentences(raw_text: str) -> list[str]:
    """Extracts normalized sentence candidates.

    Args:
        raw_text: Source text to split.

    Returns:
        list[str]: Clean sentence list.
    """
    sentence_parts: list[str] = re.split(r"(?<=[.!?])\s+|\n+", raw_text)
    normalized_sentences: list[str] = []

    for current_part in sentence_parts:
        current_normalized_part: str = normalize_text(current_part)
        # Keep only useful sentences.
        if len(current_normalized_part) < 20:
            continue
        if len(current_normalized_part.split()) < 4:
            continue
        normalized_sentences.append(current_normalized_part)

    return normalized_sentences


def build_reference_sentence_set(original_data_directory_path: Path) -> set[str]:
    """Builds a set of normalized sentences from original txt files.

    Args:
        original_data_directory_path: Directory that holds the source txt files.

    Returns:
        set[str]: Unique normalized sentences.
    """
    reference_sentence_set: set[str] = set()

    for current_text_file_path in sorted(original_data_directory_path.glob("*.txt")):
        with current_text_file_path.open("r", encoding="utf-8", errors="ignore") as current_file_handle:
            current_file_text: str = current_file_handle.read()
        reference_sentence_set.update(extract_normalized_sentences(current_file_text))

    return reference_sentence_set


def detect_article_text_column_name(field_names: list[str], sample_row: dict[str, str]) -> str:
    """Finds the best text column in articles.csv.

    Args:
        field_names: Header names from articles.csv.
        sample_row: First non-empty row for fallback checks.

    Returns:
        str: Name of the column that most likely stores article text.

    Raises:
        ValueError: If no valid text column can be found.
    """
    lower_to_original_name_map: dict[str, str] = {name.lower(): name for name in field_names}
    likely_names_in_priority_order: tuple[str, ...] = (
        "text",
        "article",
        "content",
        "body",
        "message",
        "document",
    )

    for current_likely_name in likely_names_in_priority_order:
        if current_likely_name in lower_to_original_name_map:
            return lower_to_original_name_map[current_likely_name]

    # Pick the longest text-like value if header is unknown.
    best_column_name: str = ""
    best_text_length: int = -1
    for current_column_name in field_names:
        current_value: str = sample_row.get(current_column_name, "")
        if len(current_value) > best_text_length:
            best_text_length = len(current_value)
            best_column_name = current_column_name

    if not best_column_name:
        raise ValueError("Could not detect a text column in articles.csv")

    return best_column_name


def row_has_reference_sentence(article_text: str, reference_sentence_set: set[str]) -> bool:
    """Checks if one article sentence exists in original txt sentences.

    Args:
        article_text: Full article text from one row.
        reference_sentence_set: Sentence set extracted from original txt files.

    Returns:
        bool: True if at least one sentence matches.
    """
    for current_sentence in extract_normalized_sentences(article_text):
        if current_sentence in reference_sentence_set:
            return True
    return False


def load_article_rows(article_csv_path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    """Loads article rows and detects the text column name.

    Args:
        article_csv_path: Path to the articles.csv file.

    Returns:
        tuple[list[str], list[dict[str, str]], str]: Header, rows, and detected text column.
    """
    with article_csv_path.open("r", encoding="utf-8", newline="") as csv_file_handle:
        csv_reader: csv.DictReader[str] = csv.DictReader(csv_file_handle)
        field_names: list[str] = list(csv_reader.fieldnames or [])
        article_rows: list[dict[str, str]] = list(csv_reader)

    if not field_names:
        raise ValueError(f"No header found in CSV: {article_csv_path}")
    if not article_rows:
        raise ValueError(f"No rows found in CSV: {article_csv_path}")

    sample_row: dict[str, str] = article_rows[0]
    detected_text_column_name: str = detect_article_text_column_name(field_names=field_names, sample_row=sample_row)
    return field_names, article_rows, detected_text_column_name


def write_csv_rows(output_csv_path: Path, field_names: list[str], rows: Iterable[dict[str, str]]) -> None:
    """Writes rows to CSV while preserving the original schema.

    Args:
        output_csv_path: Destination CSV path.
        field_names: Output header order.
        rows: Rows to write.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file_handle:
        csv_writer: csv.DictWriter[str] = csv.DictWriter(csv_file_handle, fieldnames=field_names)
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def split_articles(
    article_csv_path: Path,
    original_data_directory_path: Path,
    in_articles_output_path: Path,
    not_in_articles_output_path: Path,
) -> SplitSummary:
    """Splits article rows by sentence overlap with original txt files.

    Args:
        article_csv_path: Source articles CSV path.
        original_data_directory_path: Directory with reference txt files.
        in_articles_output_path: Output path for matching rows.
        not_in_articles_output_path: Output path for non-matching rows.

    Returns:
        SplitSummary: Counts for the split operation.
    """
    field_names, article_rows, text_column_name = load_article_rows(article_csv_path)
    reference_sentence_set: set[str] = build_reference_sentence_set(original_data_directory_path)

    in_articles_rows: list[dict[str, str]] = []
    not_in_articles_rows: list[dict[str, str]] = []

    for current_row in article_rows:
        current_article_text: str = current_row.get(text_column_name, "")
        if row_has_reference_sentence(current_article_text, reference_sentence_set):
            in_articles_rows.append(current_row)
        else:
            not_in_articles_rows.append(current_row)

    write_csv_rows(in_articles_output_path, field_names, in_articles_rows)
    write_csv_rows(not_in_articles_output_path, field_names, not_in_articles_rows)

    return SplitSummary(
        total_rows=len(article_rows),
        matched_rows=len(in_articles_rows),
        non_matched_rows=len(not_in_articles_rows),
    )


def parse_cli_arguments() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed CLI values.
    """
    repository_root_directory_path: Path = Path(__file__).resolve().parents[2]
    default_articles_csv_path: Path = repository_root_directory_path / "data" / "raw" / "articles.csv"
    default_original_data_directory_path: Path = repository_root_directory_path / "data" / "original"
    default_split_directory_path: Path = default_original_data_directory_path / "split"

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Split articles.csv by overlap with original txt files."
    )
    parser.add_argument("--articles-csv", type=Path, default=default_articles_csv_path)
    parser.add_argument("--original-dir", type=Path, default=default_original_data_directory_path)
    parser.add_argument(
        "--in-output",
        type=Path,
        default=default_split_directory_path / "in_articles.csv",
    )
    parser.add_argument(
        "--not-in-output",
        type=Path,
        default=default_split_directory_path / "not_in_articles.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the split command with default assignment paths."""
    csv.field_size_limit(50_000_000)
    cli_arguments: argparse.Namespace = parse_cli_arguments()

    split_summary: SplitSummary = split_articles(
        article_csv_path=cli_arguments.articles_csv,
        original_data_directory_path=cli_arguments.original_dir,
        in_articles_output_path=cli_arguments.in_output,
        not_in_articles_output_path=cli_arguments.not_in_output,
    )

    print(f"Total rows: {split_summary.total_rows}")
    print(f"Rows with sentence match: {split_summary.matched_rows}")
    print(f"Rows with zero sentence match: {split_summary.non_matched_rows}")
    print(f"Wrote: {cli_arguments.in_output}")
    print(f"Wrote: {cli_arguments.not_in_output}")


if __name__ == "__main__":
    main()

