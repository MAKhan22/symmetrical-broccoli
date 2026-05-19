from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    """A vocabulary item scored by the retrieval / reranking pipeline."""

    word: str
    bi_score: float | None = None
    cross_score: float | None = None
    frequency: int | None = None

    @property
    def final_score(self) -> float:
        if self.cross_score is not None:
            return self.cross_score
        if self.bi_score is not None:
            return self.bi_score
        return float("-inf")
