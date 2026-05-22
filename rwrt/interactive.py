from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rwrt.learner import LearnerProfile
    from rwrt.pipeline import RecommendationPipeline


def run_learn_session(
    profile: LearnerProfile,
    pipe: RecommendationPipeline,
    *,
    return_n: int,
    on_save: Callable[[], None] | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> int:
    """Interactive pick-and-learn loop. Returns a process exit code."""
    while True:
        try:
            results = pipe.recommend(profile, return_n=return_n)
        except ValueError as exc:
            print_fn(f"Error: {exc}", file=sys.stderr)
            return 1

        if not results:
            print_fn("No suggestions available.")
            return 0

        print_fn(f"\nKnown words: {len(profile)}")
        for i, candidate in enumerate(results, 1):
            bi = f"{candidate.bi_score:.4f}" if candidate.bi_score is not None else "—"
            cross = (
                f"{candidate.cross_score:.4f}"
                if candidate.cross_score is not None
                else "—"
            )
            freq = candidate.frequency if candidate.frequency is not None else "—"
            print_fn(
                f"  {i}. {candidate.word:20}  bi={bi}  cross={cross}  freq={freq}"
            )

        prompt = f"Pick 1-{len(results)} to learn, [r]efresh, [q]uit: "
        choice = input_fn(prompt).strip().lower()

        if choice in ("q", "quit"):
            break
        if choice in ("r", "refresh", ""):
            continue

        try:
            index = int(choice)
        except ValueError:
            print_fn("Invalid input — enter a number, r, or q.")
            continue

        if index < 1 or index > len(results):
            print_fn(f"Enter a number between 1 and {len(results)}.")
            continue

        word = results[index - 1].word
        if profile.learn(word):
            print_fn(f"Learned: {word}")
            if on_save is not None:
                on_save()
        else:
            print_fn(f"Could not add {word!r}.")

    return 0
