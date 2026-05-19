from __future__ import annotations

from rwrt.config import PipelineConfig
from rwrt.index import EmbeddingIndex
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
    ) -> list[Candidate]:
        k = k if k is not None else self._config.retrieve_k
        known = self._vocabulary.filter_known(known_words)
        if not known:
            raise ValueError(
                "None of the known words appear in the vocabulary. "
                "Check spelling or min_weight cutoff."
            )

        self._index.ensure_loaded()

        #TODO: try different query methods and check performance
        if self._config.query_strategy == "mean_embedding":
            query_vec = self._index.mean_embedding(sorted(known))
        else:
            raise ValueError(f"Unsupported query_strategy: {self._config.query_strategy}")

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
