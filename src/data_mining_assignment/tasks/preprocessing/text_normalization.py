"""Advanced text normalization used by clustering and anomaly detection.

This module provides a configurable normalization pipeline that produces two
parallel text views from the same raw input. The two views are useful for two
separate downstream tasks:

- Clustering (semantic view): Aggressively cleaned text suitable for TF-IDF and
  other semantic vectorizers. This view focuses on words and reduces structural
  noise so cluster algorithms group by topic.

- Anomaly detection (structural view): A view that preserves punctuation,
  HTML remnants and mixed tokens so structural irregularities are available as
  features for outlier detection.

The pipeline is deterministic and light-weight so it can be run easily in
notebooks and unit tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass(slots=True)
class NormalizationConfig:
    """Stores all text normalization settings for a single pipeline.

    Use this dataclass to enable or disable specific cleaning steps. Each
    attribute controls a single aspect of the pipeline so behaviours are easy
    to compare and test.

    Attributes:
        remove_html_tags: Removes HTML tags with regex so markup does not pollute tokens.
        remove_urls: Removes web links that do not contribute to topics.
        remove_emails: Removes email addresses to avoid leakage and rare tokens.
        remove_non_alphanumeric: Removes punctuation and symbols when set.
        remove_digits: Removes tokens containing digits (IDs, refs) when set.
        use_lemmatization: Use WordNet lemmatizer to produce dictionary lemmas.
        use_stemming_fallback: Use Porter stemmer when lemmatizer is unavailable.
        extra_stop_words: Additional stop words to add to the base stoplist.
        preserve_structure_markers: Keep punctuation/symbol tokens for anomaly features.
    """

    remove_html_tags: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_non_alphanumeric: bool = True
    remove_digits: bool = True
    use_lemmatization: bool = True
    use_stemming_fallback: bool = False
    extra_stop_words: set[str] = field(default_factory=set)
    preserve_structure_markers: bool = False


@dataclass(slots=True)
class NormalizedTextBundle:
    """Container for the two normalized text lists.

    The two lists are aligned with the original document order so IDs match
    across preprocessing and downstream stages.

    Attributes:
        clustering_texts: Text cleaned for semantic grouping and vectorization.
        anomaly_texts: Text cleaned for anomaly detection with structure kept.
    """

    clustering_texts: list[str]
    anomaly_texts: list[str]


class TextNormalizer:
    """Applies advanced normalization for text mining.

    The class provides two pipelines running on the same input. Default
    configurations are chosen to be sensible for common text-mining tasks, but
    they are fully configurable via the NormalizationConfig dataclass.

    The class also exposes the basic morphology tools used in normalization.
    """

    def __init__(
        self,
        clustering_config: NormalizationConfig | None = None,
        anomaly_config: NormalizationConfig | None = None,
    ) -> None:
        """Create a normalizer with default or provided configurations.

        If no configuration is passed, the method will set two defaults:
        - `clustering_config` removes digits and punctuation and adds web tokens
          to the stop word set.
        - `anomaly_config` preserves punctuation and digits so structural
          features remain available for detection.

        Args:
            clustering_config: Optional custom rules for the clustering pipeline.
            anomaly_config: Optional custom rules for the anomaly pipeline.
        """
        # Default clustering config: aggressive cleaning for topic discovery.
        self.clustering_config = clustering_config or NormalizationConfig(
            preserve_structure_markers=False,
            remove_non_alphanumeric=True,
            remove_digits=True,
            use_stemming_fallback=False,
            extra_stop_words={"nbsp", "http", "https", "www"},
        )

        # Default anomaly config: preserve structural markers for outlier features.
        self.anomaly_config = anomaly_config or NormalizationConfig(
            preserve_structure_markers=True,
            remove_non_alphanumeric=False,
            remove_digits=False,
            use_lemmatization=False,
            use_stemming_fallback=False,
            extra_stop_words=set(),
        )

        # Prepare morphology helpers used later in _normalize_token.
        # WordNetLemmatizer requires NLTK wordnet data to be present.
        self.wordnet_lemmatizer = WordNetLemmatizer()

        # PorterStemmer is a lightweight fallback when lemmatizer is not used.
        self.porter_stemmer = PorterStemmer()

        # Use scikit-learn's English stop words as a reliable base list.
        self.base_stop_words = set(ENGLISH_STOP_WORDS)

    def normalize_for_both_tasks(self, raw_texts: list[str]) -> NormalizedTextBundle:
        """Run both normalization pipelines on a list of raw texts.

        This helper keeps the input order and returns both normalized views so
        downstream components can pick the appropriate variant.

        Args:
            raw_texts: List of raw document strings.

        Returns:
            NormalizedTextBundle: Dataclass with clustering and anomaly lists.
        """
        # Apply the clustering pipeline to each raw text.
        clustering_texts = [
            self.normalize_text(single_text, self.clustering_config) for single_text in raw_texts
        ]

        # Apply the anomaly pipeline to each raw text.
        anomaly_texts = [self.normalize_text(single_text, self.anomaly_config) for single_text in raw_texts]

        # Return both views together in a stable structure.
        return NormalizedTextBundle(clustering_texts=clustering_texts, anomaly_texts=anomaly_texts)

    def normalize_text(self, raw_text: str, normalization_config: NormalizationConfig) -> str:
        """Normalize a single raw text using the provided configuration.

        The implementation applies a clear sequence of text cleaning steps. The
        sequence is designed to minimize accidental token merging and to keep
        structural markers when required by the config.

        Args:
            raw_text: The raw document string to normalize.
            normalization_config: The configuration that controls which steps run.

        Returns:
            str: The normalized single-line string ready for vectorizers.
        """
        # Lowercase the whole document to canonicalize casing.
        cleaned_text = raw_text.lower()

        # Strip HTML tags when requested. Replace with space to keep token
        # boundaries intact.
        if normalization_config.remove_html_tags:
            cleaned_text = re.sub(r"<[^>]+>", " ", cleaned_text)

        # Remove web links like http://... and www.... to avoid noisy tokens.
        if normalization_config.remove_urls:
            cleaned_text = re.sub(r"https?://\S+|www\.\S+", " ", cleaned_text)

        # Remove email-like tokens to avoid address leakage and rare tokens.
        if normalization_config.remove_emails:
            cleaned_text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", cleaned_text)

        # Remove tokens that contain digits when that cleaning is enabled.
        # This targets IDs like REF-8821 and station44 which often do not add
        # semantic value for clustering.
        if normalization_config.remove_digits:
            cleaned_text = re.sub(r"\b\w*\d\w*\b", " ", cleaned_text)

        # If we are allowed to remove non-alphanumeric characters, do it now.
        # We keep letters and whitespace only which simplifies tokenization.
        if normalization_config.remove_non_alphanumeric and not normalization_config.preserve_structure_markers:
            cleaned_text = re.sub(r"[^a-z\s]", " ", cleaned_text)

        # Tokenize into word-like tokens. Allow apostrophes to keep contractions.
        token_list = re.findall(r"[a-z']+", cleaned_text)

        # If preserving structure markers, also collect symbols (like '%', '#')
        # and append them as tokens so they are represented in the output.
        if normalization_config.preserve_structure_markers:
            structural_token_list = re.findall(r"[^\s\w]", cleaned_text)
            token_list.extend(structural_token_list)

        # Build the stop word set: base English stop words plus any extras.
        stop_word_set = self.base_stop_words | normalization_config.extra_stop_words

        # Filter out stop words to reduce noise and dimensionality.
        filtered_token_list = [single_token for single_token in token_list if single_token not in stop_word_set]

        # Apply morphological normalization to each token (lemmatize or stem).
        normalized_token_list = [
            self._normalize_token(single_token, normalization_config) for single_token in filtered_token_list
        ]

        # Remove any empty tokens and join into a final space-separated string.
        normalized_token_list = [single_token for single_token in normalized_token_list if single_token]
        return " ".join(normalized_token_list)

    def _normalize_token(self, raw_token: str, normalization_config: NormalizationConfig) -> str:
        """Normalize a single token using lemmatization or stemming rules.

        This helper preserves common structural punctuation tokens so that
        anomaly detectors can use them as binary-like features.

        The function first checks for structural tokens, then attempts
        lemmatization. If WordNet data is missing and a fallback is allowed,
        it uses Porter stemming. If no normalization is requested the token
        is returned unchanged.

        Args:
            raw_token: The token string to normalize.
            normalization_config: Config that controls lemmatization/stemming.

        Returns:
            str: The normalized token or the original token if no change applied.
        """
        # Preserve a list of punctuation symbols as tokens for anomaly detection.
        if raw_token in {"!", "?", ".", ",", ":", ";", "#", "$", "%", "&", "*", "@"}:
            return raw_token

        # If lemmatization is enabled prefer it for linguistic correctness.
        if normalization_config.use_lemmatization:
            try:
                # Lemmatize without POS tagging for speed; this handles many cases.
                return self.wordnet_lemmatizer.lemmatize(raw_token)
            except LookupError:
                # WordNet data might not be installed in some environments.
                if normalization_config.use_stemming_fallback:
                    # Use Porter stemmer as a deterministic fallback.
                    return self.porter_stemmer.stem(raw_token)
                # If no fallback is allowed, return the token unchanged.
                return raw_token

        # If lemmatization is not used but stemming fallback is enabled, stem.
        if normalization_config.use_stemming_fallback:
            return self.porter_stemmer.stem(raw_token)

        # Otherwise return the original token unchanged.
        return raw_token
