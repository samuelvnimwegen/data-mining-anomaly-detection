"""Tests for advanced text normalization."""

from __future__ import annotations

from preprocessing import NormalizationConfig, TextNormalizer


def test_normalization_removes_html_stopwords_and_digit_codes() -> None:
    """Checks clustering normalization removes noisy patterns."""
    normalizer = TextNormalizer(
        clustering_config=NormalizationConfig(
            remove_html_tags=True,
            remove_urls=True,
            remove_emails=True,
            remove_non_alphanumeric=True,
            remove_digits=True,
            use_lemmatization=False,
            use_stemming_fallback=False,
            extra_stop_words={"promo"},
            preserve_structure_markers=False,
        )
    )

    raw_text = "<div>Promo OFFER 2024!!! Contact us: fake@shop.com and visit https://x.com now.</div>"
    normalized_text = normalizer.normalize_text(raw_text, normalizer.clustering_config)

    assert "<div>" not in normalized_text
    assert "promo" not in normalized_text
    assert "2024" not in normalized_text
    assert "https" not in normalized_text
    assert "fake" not in normalized_text
    assert "shop" not in normalized_text


def test_anomaly_variant_keeps_structure_markers() -> None:
    """Checks anomaly normalization keeps punctuation markers."""
    normalizer = TextNormalizer(
        anomaly_config=NormalizationConfig(
            remove_html_tags=True,
            remove_urls=False,
            remove_emails=False,
            remove_non_alphanumeric=False,
            remove_digits=False,
            use_lemmatization=False,
            use_stemming_fallback=False,
            preserve_structure_markers=True,
        )
    )

    raw_text = "Alert!!! Buy-now ### code-991"
    normalized_text = normalizer.normalize_text(raw_text, normalizer.anomaly_config)

    assert "!" in normalized_text
    assert "#" in normalized_text


def test_normalizer_returns_two_text_views() -> None:
    """Checks two normalized views are generated for the same input."""
    normalizer = TextNormalizer()
    text_bundle = normalizer.normalize_for_both_tasks(["<b>Cars</b> are FAST!!!"])

    assert len(text_bundle.clustering_texts) == 1
    assert len(text_bundle.anomaly_texts) == 1
    assert text_bundle.clustering_texts[0] != text_bundle.anomaly_texts[0]
