"""Tests for structural anomaly feature extraction."""

from __future__ import annotations

from preprocessing import StructuralFeatureExtractor


def test_structural_feature_extractor_returns_expected_shape() -> None:
    """Checks extractor output shape and feature count."""
    extractor = StructuralFeatureExtractor()
    matrix = extractor.transform(["Simple sentence.", "Another one with 123 and <b>tag</b> and a@b.com"])

    assert matrix.shape == (2, len(extractor.get_feature_names()))


def test_structural_feature_extractor_detects_urls_and_html() -> None:
    """Checks URL and HTML counts react to structured noise."""
    extractor = StructuralFeatureExtractor()
    matrix = extractor.transform(["<div>hello</div> http://example.com"])
    feature_names = extractor.get_feature_names()

    url_index = feature_names.index("url_count")
    html_index = feature_names.index("html_tag_count")

    assert matrix[0, url_index] >= 1.0
    assert matrix[0, html_index] >= 1.0

