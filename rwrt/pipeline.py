from __future__ import annotations

import math
from rwrt.config import PipelineConfig
from rwrt.index import EmbeddingIndex
from rwrt.learner import LearnerProfile
from rwrt.rerank import CrossEncoderReranker, WeightedFeatureReranker
from rwrt.retrieve import BiEncoderRetriever
from rwrt.types import Candidate
from rwrt.vocabulary import VocabularyStore


def _known_set(known_words: set[str] | list[str] | LearnerProfile) -> set[str]:
    if isinstance(known_words, LearnerProfile):
        return set(known_words.known)
    if isinstance(known_words, set):
        return known_words
    return set(known_words)


class RecommendationPipeline:
    """Orchestrate bi-encoder retrieval and optional cross-encoder reranking."""

    def __init__(
        self,
        config: PipelineConfig,
        vocabulary: VocabularyStore,
        *,
        index: EmbeddingIndex | None = None,
    ) -> None:
        self._config = config
        self._vocabulary = vocabulary
        self._index = index or EmbeddingIndex(config, vocabulary)
        self._retriever = BiEncoderRetriever(config, vocabulary, self._index)
        if config.reranker_type == "cross_encoder":
            self._reranker = CrossEncoderReranker(config, vocabulary=vocabulary, index=self._index)
        else:
            self._reranker = WeightedFeatureReranker(config, vocabulary=vocabulary, index=self._index)

    @property
    def index(self) -> EmbeddingIndex:
        return self._index

    def recommend(
        self,
        known_words: set[str] | list[str] | LearnerProfile,
        *,
        retrieve_k: int | None = None,
        return_n: int | None = None,
        use_cross_encoder: bool | None = None,
        topic_keyword: str | None = None,
    ) -> list[Candidate]:
        known = _known_set(known_words)
        retrieve_k = retrieve_k if retrieve_k is not None else self._config.retrieve_k
        return_n = return_n if return_n is not None else self._config.return_n
        use_cross = (
            use_cross_encoder
            if use_cross_encoder is not None
            else self._config.use_cross_encoder
        )

        candidates = self._retriever.retrieve(
            known, k=retrieve_k, topic_keyword=topic_keyword
        )

        # This is an optional parameter that can be used to boost the score of high-frequency words.
        if self._config.frequency_boost > 0:
            candidates = self._apply_frequency_boost(candidates)

        if use_cross:
            return self._reranker.rerank(
                known, candidates, n=return_n, topic_keyword=topic_keyword
            )

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[:return_n]

    def _apply_frequency_boost(self, candidates: list[Candidate]) -> list[Candidate]:
        boost = self._config.frequency_boost
        boosted: list[Candidate] = []
        for c in candidates:
            freq = c.frequency or 1
            bi = c.bi_score if c.bi_score is not None else 0.0
            score = bi + boost * math.log1p(freq)
            boosted.append(
                Candidate(
                    word=c.word,
                    bi_score=score,
                    cross_score=c.cross_score,
                    frequency=c.frequency,
                )
            )
        boosted.sort(key=lambda x: x.bi_score or 0.0, reverse=True)
        return boosted
