from __future__ import annotations

from rwrt.config import PipelineConfig
from rwrt.index import EmbeddingIndex
from rwrt.query import select_query_words
from rwrt.types import Candidate
from rwrt.vocabulary import VocabularyStore


class BiEncoderRetriever:
    """Retrieve semantically similar unknown words via bi-encoder + FAISS."""

    def __init__(
        self,
        config: PipelineConfig,
        vocabulary: VocabularyStore,
        index: EmbeddingIndex,
    ) -> None:
        self._config = config
        self._vocabulary = vocabulary
        self._index = index

    def retrieve(
        self,
        known_words: set[str],
        k: int | None = None,
        *,
        topic_keyword: str | None = None,
    ) -> list[Candidate]:
        k = k if k is not None else self._config.retrieve_k
        known = self._vocabulary.filter_known(known_words)
        if not known:
            raise ValueError(
                "None of the known words appear in the vocabulary. "
                "Check spelling or min_weight cutoff."
            )

        self._index.ensure_loaded()

        effective_topic = topic_keyword or self._config.topic_keyword
        if self._config.query_strategy == "topic":
            query_vec = self._build_query_vector([], topic_keyword=effective_topic)
        else:
            query_words = select_query_words(
                known,
                max_words=self._config.max_query_words,
                vocabulary=self._vocabulary,
                selection=self._config.query_word_selection,
                topic_keyword=effective_topic,
                index=self._index,
            )
            query_vec = self._build_query_vector(query_words, topic_keyword=effective_topic)

        # Oversample so filtering known words still yields enough candidates.
        oversample = max(k * 3, k + len(known))
        hits = self._index.search(query_vec, min(oversample, len(self._index.words)))

        candidates: list[Candidate] = []
        for word, score in hits:
            if word in known:
                continue
            candidates.append(
                Candidate(
                    word=word,
                    bi_score=score,
                    frequency=self._vocabulary.frequency(word),
                )
            )
            if len(candidates) >= k:
                break

        return candidates

    def _build_query_vector(
        self,
        query_words: list[str],
        *,
        topic_keyword: str | None = None,
    ):
        strategy = self._config.query_strategy
        if strategy == "mean_embedding":
            return self._index.mean_embedding(query_words)
        if strategy == "weighted_mean":
            weights = {
                w: float(self._vocabulary.frequency(w) or 1) for w in query_words
            }
            return self._index.weighted_mean_embedding(query_words, weights)
        if strategy == "inverse_weighted_mean":
            weights = {
                w: 1.0 / float(self._vocabulary.frequency(w) or 1) for w in query_words
            }
            return self._index.weighted_mean_embedding(query_words, weights)
        if strategy == "topic":
            keyword = topic_keyword
            if not keyword or not keyword.strip():
                raise ValueError("topic_keyword is required when query_strategy='topic'")
            return self._index.encode_text(keyword.strip())
        raise ValueError(f"Unsupported query_strategy: {strategy!r}")
