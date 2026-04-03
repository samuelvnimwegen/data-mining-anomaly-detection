"""Input and output helpers for text mining tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ArticleDataset:
    """Loads and validates the article dataset.

    Args:
        input_csv_path: Path to the source CSV.
        document_id_column_name: Optional column name for document ids.
        text_column_name: Optional column name for document text.
    """

    def __init__(
        self,
        input_csv_path: Path,
        document_id_column_name: str | None = None,
        text_column_name: str | None = None,
    ) -> None:
        """Sets all dataset loader settings.

        Args:
            input_csv_path: Path to the source CSV.
            document_id_column_name: Optional column name for document ids.
            text_column_name: Optional column name for document text.
        """
        self.input_csv_path = input_csv_path
        self.document_id_column_name = document_id_column_name
        self.text_column_name = text_column_name

    def load_articles(self) -> pd.DataFrame:
        """Loads articles and returns a clean two-column DataFrame.

        Returns:
            pd.DataFrame: DataFrame with `doc_id` and `text` columns.

        Raises:
            FileNotFoundError: If the input CSV path does not exist.
            ValueError: If a text column cannot be found.
        """
        if not self.input_csv_path.exists():
            raise FileNotFoundError(f"Input file was not found: {self.input_csv_path}")

        loaded_data_frame = pd.read_csv(self.input_csv_path)
        selected_document_id_column_name, selected_text_column_name = self._pick_columns(loaded_data_frame)

        normalized_data_frame = loaded_data_frame[[selected_document_id_column_name, selected_text_column_name]].copy()
        normalized_data_frame.columns = ["doc_id", "text"]
        normalized_data_frame["doc_id"] = normalized_data_frame["doc_id"].astype(str)
        normalized_data_frame["text"] = normalized_data_frame["text"].fillna("").astype(str)
        return normalized_data_frame

    def _pick_columns(self, loaded_data_frame: pd.DataFrame) -> tuple[str, str]:
        """Finds the id and text columns in the source data.

        Args:
            loaded_data_frame: Raw DataFrame loaded from CSV.

        Returns:
            tuple[str, str]: A tuple with (document_id_column, text_column).

        Raises:
            ValueError: If a text column cannot be detected.
        """
        if self.document_id_column_name and self.text_column_name:
            return self.document_id_column_name, self.text_column_name

        normalized_column_name_map = {column_name.lower(): column_name for column_name in loaded_data_frame.columns}

        detected_document_id_column_name = (
            self.document_id_column_name
            or normalized_column_name_map.get("doc_id")
            or normalized_column_name_map.get("document_id")
            or loaded_data_frame.columns[0]
        )

        detected_text_column_name = self.text_column_name
        if not detected_text_column_name:
            for candidate_column_name in ("text", "content", "article", "document", "body"):
                if candidate_column_name in normalized_column_name_map:
                    detected_text_column_name = normalized_column_name_map[candidate_column_name]
                    break

        if not detected_text_column_name:
            if len(loaded_data_frame.columns) >= 2:
                detected_text_column_name = loaded_data_frame.columns[1]
            else:
                raise ValueError("Could not detect a text column in the input dataset.")

        return str(detected_document_id_column_name), str(detected_text_column_name)


def save_clusters(cluster_data_frame: pd.DataFrame, output_csv_path: Path) -> None:
    """Saves cluster labels to CSV.

    Args:
        cluster_data_frame: DataFrame with cluster labels.
        output_csv_path: Destination CSV path.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_data_frame[["doc_id", "label"]].to_csv(output_csv_path, index=False)


def save_anomalies(anomaly_data_frame: pd.DataFrame, output_csv_path: Path) -> None:
    """Saves anomaly ranking to CSV.

    Args:
        anomaly_data_frame: DataFrame with anomaly ranking rows.
        output_csv_path: Destination CSV path.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_data_frame[["anomaly", "doc_id"]].to_csv(output_csv_path, index=False)
