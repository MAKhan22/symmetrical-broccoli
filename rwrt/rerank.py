from __future__ import annotations

from typing import TYPE_CHECKING

from rwrt.config import PipelineConfig
from rwrt.encoders import get_cross_encoder
from rwrt.query import select_query_words
from rwrt.types import Candidate

if TYPE_CHECKING:
    from rwrt.index import EmbeddingIndex
    from rwrt.vocabulary import VocabularyStore


class CrossEncoderReranker:
    """Re-rank bi-encoder candidates with a cross-encoder."""

    def __init__(
        self,
        config: PipelineConfig,
        vocabulary: VocabularyStore | None = None,
        index: EmbeddingIndex | None = None,
    ) -> None:
        self._config = config
        self._vocabulary = vocabulary
        self._index = index
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = get_cross_encoder(self._config.cross_model, self._config.device)
        return self._model

    def format_query(
        self,
        known_words: set[str],
        *,
        topic_keyword: str | None = None,
    ) -> str:
        selected = select_query_words(
            known_words,
            max_words=self._config.max_query_words,
            vocabulary=self._vocabulary,
            selection=self._config.query_word_selection,
            topic_keyword=topic_keyword or self._config.topic_keyword,
            index=self._index,
        )
        topic = topic_keyword or self._config.topic_keyword
        if self._config.query_word_selection == "topic" and topic:
            prefix = f"Konu: {topic.strip()}. Bildiğim Türkçe kelimeler: "
        else:
            prefix = "Bildiğim Türkçe kelimeler: "
        return prefix + ", ".join(sorted(selected))

    def rerank(
        self,
        known_words: set[str],
        candidates: list[Candidate],
        n: int | None = None,
        *,
        topic_keyword: str | None = None,
    ) -> list[Candidate]:
        n = n if n is not None else self._config.return_n
        if not candidates:
            return []

        query = self.format_query(known_words, topic_keyword=topic_keyword)
        model = self._get_model()
        pairs = [(query, c.word) for c in candidates]
        scores = model.predict(
            pairs,
            batch_size=self._config.cross_batch_size,
            show_progress_bar=False,
        )

        scored = [
            Candidate(
                word=c.word,
                bi_score=c.bi_score,
                cross_score=float(score),
                frequency=c.frequency,
            )
            for c, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda c: c.final_score, reverse=True)
        return scored[:n]
