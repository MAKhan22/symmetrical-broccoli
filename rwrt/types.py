from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    """A vocabulary item scored by the retrieval / reranking pipeline."""

    word: str
    bi_score: float | None = None
    feature_score: float | None = None
    cross_score: float | None = None
    frequency: int | None = None

    @property
    def final_score(self) -> float:
        if self.cross_score is not None:
            return self.cross_score
        if self.feature_score is not None:
            return self.feature_score
        if self.bi_score is not None:
            return self.bi_score
        return float("-inf")


def candidate_score_fields(
    candidate: Candidate, *, precision: int = 4
) -> tuple[str, str, str, str]:
    """Return (bi, wf, cross, freq) display strings."""
    fmt = f".{precision}f"

    def _score(value: float | None) -> str:
        return format(value, fmt) if value is not None else "—"

    freq = str(candidate.frequency) if candidate.frequency is not None else "—"
    return (
        _score(candidate.bi_score),
        _score(candidate.feature_score),
        _score(candidate.cross_score),
        freq,
    )


def format_candidate_scores(candidate: Candidate, *, precision: int = 4) -> str:
    """Format bi / weighted-feature / cross-encoder scores for CLI output."""
    bi, wf, cross, freq = candidate_score_fields(candidate, precision=precision)
    return f"bi={bi}  wf={wf}  cross={cross}  freq={freq}"
