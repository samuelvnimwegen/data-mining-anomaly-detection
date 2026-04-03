"""Advanced text normalization used by clustering and anomaly detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass(slots=True)
class NormalizationConfig:
    """Stores all text normalization settings.

    Attributes:
        remove_html_tags: Removes HTML tags with regex.
        remove_urls: Removes web links from text.
        remove_emails: Removes email like tokens.
        remove_non_alphanumeric: Removes punctuation and symbols.
        remove_digits: Removes numbers and mixed id codes.
        use_lemmatization: Uses WordNet lemmatization.
        use_stemming_fallback: Uses stemming if lemmatization data is missing.
        extra_stop_words: Extra stop words from corpus patterns.
        preserve_structure_markers: Keeps punctuation and symbols for anomaly view.
    """

    remove_html_tags: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_non_alphanumeric: bool = True
    remove_digits: bool = True
    use_lemmatization: bool = True
    use_stemming_fallback: bool = True
    extra_stop_words: set[str] = field(default_factory=set)
    preserve_structure_markers: bool = False


@dataclass(slots=True)
class NormalizedTextBundle:
    """Stores normalized text variants for different tasks.

    Attributes:
        clustering_texts: Text cleaned for semantic grouping.
        anomaly_texts: Text cleaned but with structure markers.
    """

    clustering_texts: list[str]
    anomaly_texts: list[str]


class TextNormalizer:
    """Applies advanced normalization for text mining.

    Args:
        clustering_config: Rules used for clustering text.
        anomaly_config: Rules used for anomaly text.
    """

    def __init__(
        self,
        clustering_config: NormalizationConfig | None = None,
        anomaly_config: NormalizationConfig | None = None,
    ) -> None:
        """Sets all normalizer settings.

        Args:
            clustering_config: Rules used for clustering text.
            anomaly_config: Rules used for anomaly text.
        """
        self.clustering_config = clustering_config or NormalizationConfig(
            preserve_structure_markers=False,
            remove_non_alphanumeric=True,
            remove_digits=True,
            extra_stop_words={"nbsp", "http", "https", "www"},
        )
        self.anomaly_config = anomaly_config or NormalizationConfig(
            preserve_structure_markers=True,
            remove_non_alphanumeric=False,
            remove_digits=False,
            use_lemmatization=False,
            extra_stop_words=set(),
        )
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.porter_stemmer = PorterStemmer()
        self.base_stop_words = set(ENGLISH_STOP_WORDS)

    def normalize_for_both_tasks(self, raw_texts: list[str]) -> NormalizedTextBundle:
        """Normalizes one text list into clustering and anomaly variants.

        Args:
            raw_texts: Raw input texts.

        Returns:
            NormalizedTextBundle: Two normalized text lists.
        """
        clustering_texts = [self.normalize_text(single_text, self.clustering_config) for single_text in raw_texts]
        anomaly_texts = [self.normalize_text(single_text, self.anomaly_config) for single_text in raw_texts]
        return NormalizedTextBundle(clustering_texts=clustering_texts, anomaly_texts=anomaly_texts)

    def normalize_text(self, raw_text: str, normalization_config: NormalizationConfig) -> str:
        """Normalizes one text string with selected settings.

        Args:
            raw_text: Raw input text.
            normalization_config: Rules for one normalization path.

        Returns:
            str: Cleaned text string.
        """
        cleaned_text = raw_text.lower()

        if normalization_config.remove_html_tags:
            cleaned_text = re.sub(r"<[^>]+>", " ", cleaned_text)

        if normalization_config.remove_urls:
            cleaned_text = re.sub(r"https?://\S+|www\.\S+", " ", cleaned_text)

        if normalization_config.remove_emails:
            cleaned_text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", cleaned_text)

        if normalization_config.remove_digits:
            cleaned_text = re.sub(r"\b\w*\d\w*\b", " ", cleaned_text)

        if normalization_config.remove_non_alphanumeric and not normalization_config.preserve_structure_markers:
            cleaned_text = re.sub(r"[^a-z\s]", " ", cleaned_text)

        token_list = re.findall(r"[a-z']+", cleaned_text)
        if normalization_config.preserve_structure_markers:
            structural_token_list = re.findall(r"[^\s\w]", cleaned_text)
            token_list.extend(structural_token_list)

        stop_word_set = self.base_stop_words | normalization_config.extra_stop_words
        filtered_token_list = [single_token for single_token in token_list if single_token not in stop_word_set]

        normalized_token_list = [
            self._normalize_token(single_token, normalization_config) for single_token in filtered_token_list
        ]
        normalized_token_list = [single_token for single_token in normalized_token_list if single_token]
        return " ".join(normalized_token_list)

    def _normalize_token(self, raw_token: str, normalization_config: NormalizationConfig) -> str:
        """Normalizes one token with lemmatization or stemming.

        Args:
            raw_token: Token before morphology step.
            normalization_config: Rules for one normalization path.

        Returns:
            str: Normalized token.
        """
        if raw_token in {"!", "?", ".", ",", ":", ";", "#", "$", "%", "&", "*", "@"}:
            return raw_token

        if normalization_config.use_lemmatization:
            try:
                return self.wordnet_lemmatizer.lemmatize(raw_token)
            except LookupError:
                # This keeps tests stable when NLTK corpora are missing.
                if normalization_config.use_stemming_fallback:
                    return self.porter_stemmer.stem(raw_token)
                return raw_token

        if normalization_config.use_stemming_fallback:
            return self.porter_stemmer.stem(raw_token)

        return raw_token
