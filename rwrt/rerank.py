from __future__ import annotations

from rwrt.config import PipelineConfig
from rwrt.encoders import get_cross_encoder
from rwrt.types import Candidate


class CrossEncoderReranker:
    """Re-rank bi-encoder candidates with a cross-encoder."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = get_cross_encoder(self._config.cross_model, self._config.device)
        return self._model

    def format_query(self, known_words: set[str]) -> str:
        ordered = sorted(known_words)
        if len(ordered) > self._config.max_query_words:
            ordered = ordered[-self._config.max_query_words :]
        return "Bildiğim Türkçe kelimeler: " + ", ".join(ordered)

    def rerank(
        self,
        known_words: set[str],
        candidates: list[Candidate],
        n: int | None = None,
    ) -> list[Candidate]:
        n = n if n is not None else self._config.return_n
        if not candidates:
            return []

        query = self.format_query(known_words)
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
