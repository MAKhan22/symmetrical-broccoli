from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

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


class WeightedFeatureReranker:
    """Multi-objective weighted feature scorer.

    score(w) = w_s * S  +  w_f * F  +  w_m * M  -  w_d * D  -  w_e * E

    S = semantic similarity (bi-encoder score)
    F = frequency utility (normalized log-frequency)
    M = morphological root overlap with known words
    D = diversity penalty (MMR: max similarity to already-selected candidates)
    E = difficulty penalty (extreme frequency drops from known-word average)
    """

    def __init__(
        self,
        config: PipelineConfig,
        vocabulary: VocabularyStore | None = None,
        index: EmbeddingIndex | None = None,
    ) -> None:
        self._config = config
        self._vocabulary = vocabulary
        self._index = index

    @staticmethod
    def _common_prefix_ratio(w1: str, w2: str) -> float:
        min_len = min(len(w1), len(w2))
        if min_len < 3:
            return 0.0
        common = 0
        for a, b in zip(w1.lower(), w2.lower()):
            if a == b:
                common += 1
            else:
                break
        return common / min_len

    def _morph_overlap(self, word: str, known_words: set[str]) -> float:
        best = 0.0
        for known in known_words:
            ratio = self._common_prefix_ratio(word, known)
            if ratio > best:
                best = ratio
        return best

    def _embedding_sim(self, word1: str, word2: str) -> float:
        if self._index is None:
            return 0.0
        try:
            v1 = self._index.embedding_for(word1)
            v2 = self._index.embedding_for(word2)
            return float(np.dot(v1, v2))
        except (KeyError, ValueError):
            return 0.0

    def _difficulty_penalty(
        self,
        freq: int,
        avg_log_freq: float,
        std_log_freq: float,
    ) -> float:
        if freq <= 0:
            return 1.0
        log_freq = math.log(freq)
        if std_log_freq <= 0:
            return 0.0
        z = (avg_log_freq - log_freq) / std_log_freq
        return max(0.0, min(1.0, z / 3.0))

    def rerank(
        self,
        known_words: set[str],
        candidates: list[Candidate],
        n: int | None = None,
        *,
        topic_keyword: str | None = None,
    ) -> list[Candidate]:
        _ = topic_keyword  # unused, kept for API compatibility
        n = n if n is not None else self._config.return_n
        if not candidates:
            return []

        cfg = self._config

        # Known-word frequency stats for difficulty penalty
        known_freqs: list[int] = []
        if self._vocabulary:
            known_freqs = [
                self._vocabulary.frequency(w) or 1
                for w in known_words
                if self._vocabulary.contains(w)
            ]
        if known_freqs:
            log_freqs = [math.log(f) for f in known_freqs]
            avg_log_freq = sum(log_freqs) / len(log_freqs)
            variance = sum((lf - avg_log_freq) ** 2 for lf in log_freqs) / len(log_freqs)
            std_log_freq = math.sqrt(variance) if variance > 0 else 0.0
        else:
            avg_log_freq = 0.0
            std_log_freq = 0.0

        # Normalisation cap for frequency feature
        max_freq_in_pool = max((c.frequency or 1) for c in candidates) if candidates else 1
        max_log_freq = math.log1p(max_freq_in_pool)

        # Step 1: score all candidates with base formula (no diversity)
        scored: list[tuple[Candidate, float]] = []
        for c in candidates:
            S = c.bi_score if c.bi_score is not None else 0.0
            F = (
                math.log1p(c.frequency or 1) / max_log_freq
                if max_log_freq > 0
                else 0.0
            )
            M = self._morph_overlap(c.word, known_words)
            E = self._difficulty_penalty(c.frequency or 1, avg_log_freq, std_log_freq)

            base = (
                cfg.weight_semantic * S
                + cfg.weight_frequency * F
                + cfg.weight_morphology * M
                - cfg.weight_difficulty * E
            )
            scored.append((c, base))

        # Step 2: MMR selection (diversity via iterative greedy)
        n_result = min(n, len(scored))
        if n_result == 0:
            return []

        selected: list[Candidate] = []
        remaining = scored[:]

        for _ in range(n_result):
            best_idx = 0
            best_final = -float("inf")

            for i, (c, base) in enumerate(remaining):
                if selected:
                    sims = [self._embedding_sim(c.word, s.word) for s in selected]
                    div_penalty = max(sims)
                else:
                    div_penalty = 0.0
                final = base - cfg.weight_diversity * div_penalty
                if final > best_final:
                    best_final = final
                    best_idx = i

            selected.append(remaining.pop(best_idx)[0])

        return selected
