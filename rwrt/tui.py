from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rwrt.types import candidate_score_fields

if TYPE_CHECKING:
    from rwrt.learner import LearnerProfile
    from rwrt.llm import OpenRouterClient
    from rwrt.pipeline import RecommendationPipeline
    from rwrt.types import Candidate


@dataclass(frozen=True)
class SuggestionChoice:
    """Parsed suggestion-screen input."""

    mark_known: frozenset[int] = frozenset()
    study: int | None = None


def parse_suggestion_input(raw: str, *, max_n: int) -> SuggestionChoice | str | None:
    """Parse TUI suggestion input.

    Returns:
        ``SuggestionChoice`` for numeric input (e.g. ``1k, 2k, 4``),
        ``"refresh"`` / ``"quit"`` for commands,
        ``None`` for empty refresh,
        or raises ``ValueError`` for invalid input.
    """
    text = raw.strip().lower()
    if text in ("q", "quit"):
        return "quit"
    if text in ("r", "refresh"):
        return "refresh"
    if not text:
        return None

    tokens = re.findall(r"\d+k?", text)
    if not tokens:
        raise ValueError("Enter numbers like 4 or 1k, 2k, 4")

    mark_known: set[int] = set()
    study: list[int] = []

    for token in tokens:
        if token.endswith("k"):
            index = int(token[:-1])
            mark_known.add(index)
        else:
            study.append(int(token))

    invalid = {i for i in mark_known | set(study) if i < 1 or i > max_n}
    if invalid:
        bad = ", ".join(str(i) for i in sorted(invalid))
        raise ValueError(f"Out of range (1–{max_n}): {bad}")

    study_index = study[-1] if study else None
    if len(study) > 1:
        raise ValueError("Pick only one word to study (without k)")

    return SuggestionChoice(mark_known=frozenset(mark_known), study=study_index)


def run_chat_session(
    profile: LearnerProfile,
    pipe: RecommendationPipeline,
    llm: OpenRouterClient,
    *,
    return_n: int,
    topic_keyword: str | None = None,
    on_save: Callable[[], None] | None = None,
    console: Console | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    """Recommendation TUI with LLM word explanations and follow-up chat."""
    console = console or Console()
    prompt_input = input_fn or console.input

    console.print(
        Panel(
            Text.from_markup(
                "[bold]rwrt chat[/] — pick a word to study, or mark words as known with "
                "[bold]k[/] (e.g. [bold]1k, 2k, 4[/] marks 1–2 known and opens 4)."
            ),
            border_style="blue",
        )
    )

    while True:
        try:
            results = pipe.recommend(
                profile,
                return_n=return_n,
                topic_keyword=topic_keyword,
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/] {exc}", file=sys.stderr)
            return 1

        if not results:
            console.print("[yellow]No suggestions available.[/]")
            return 0

        _print_suggestions(console, profile, results)

        choice = prompt_input(
            f"\nPick [bold]1–{len(results)}[/] to study, "
            f"[bold]Nk[/] to mark known (e.g. [bold]1k, 2k, 4[/]), "
            f"[bold]r[/] refresh, [bold]q[/] quit: "
        ).strip()

        try:
            parsed = parse_suggestion_input(choice, max_n=len(results))
        except ValueError as exc:
            console.print(f"[yellow]{exc}[/]")
            continue

        if parsed == "quit":
            console.print("[dim]Goodbye.[/]")
            break
        if parsed in (None, "refresh"):
            continue

        assert isinstance(parsed, SuggestionChoice)

        if parsed.mark_known:
            _mark_known(console, profile, results, parsed.mark_known, on_save=on_save)

        if parsed.study is not None:
            word = results[parsed.study - 1].word
            if not _study_word(
                console,
                llm,
                profile,
                word,
                prompt_input=prompt_input,
                on_save=on_save,
            ):
                continue

    return 0


def _mark_known(
    console: Console,
    profile: LearnerProfile,
    results: list[Candidate],
    indices: frozenset[int],
    *,
    on_save: Callable[[], None] | None,
) -> None:
    added: list[str] = []
    for index in sorted(indices):
        word = results[index - 1].word
        if profile.learn(word):
            added.append(word)

    if added:
        console.print(
            "[green]Marked known:[/] "
            + ", ".join(f"[bold]{w}[/]" for w in added)
        )
        if on_save is not None:
            on_save()
    else:
        console.print("[yellow]No new words marked (already known or invalid).[/]")


def _print_suggestions(
    console: Console,
    profile: LearnerProfile,
    results: list[Candidate],
) -> None:
    table = Table(title=f"Suggestions  ·  {len(profile)} known words", show_lines=False)
    table.add_column("#", style="cyan", justify="right", width=4)
    table.add_column("Word", style="bold")
    table.add_column("bi", justify="right")
    table.add_column("wf", justify="right")
    table.add_column("cross", justify="right")
    table.add_column("freq", justify="right")

    for i, candidate in enumerate(results, 1):
        bi, wf, cross, freq = candidate_score_fields(candidate, precision=3)
        table.add_row(str(i), candidate.word, bi, wf, cross, freq)

    console.print(table)


def _study_word(
    console: Console,
    llm: OpenRouterClient,
    profile: LearnerProfile,
    word: str,
    *,
    prompt_input: Callable[[str], str],
    on_save: Callable[[], None] | None,
) -> bool:
    console.print()
    with console.status(f"[bold green]Asking about[/] [cyan]{word}[/]…"):
        try:
            explanation = llm.explain_word(word, known_words=profile.known)
        except RuntimeError as exc:
            console.print(f"[red]LLM error:[/] {exc}")
            return False

    console.print(Panel(Markdown(explanation), title=f"[bold]{word}[/]", border_style="green"))

    history: list[dict[str, str]] = [
        {
            "role": "assistant",
            "content": explanation,
        }
    ]

    while True:
        question = prompt_input(
            "\nAsk a follow-up about this word (Enter to mark learned, [bold]b[/] back): "
        ).strip()

        if question.lower() in ("b", "back"):
            console.print("[dim]Back to suggestions.[/]")
            return False

        if not question:
            break

        with console.status("[bold green]Thinking…[/]"):
            try:
                answer = llm.follow_up(word, history, question)
            except RuntimeError as exc:
                console.print(f"[red]LLM error:[/] {exc}")
                continue

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        console.print(Panel(Markdown(answer), title="Tutor", border_style="magenta"))

    if profile.learn(word):
        console.print(f"[green]Added to profile:[/] [bold]{word}[/]")
        if on_save is not None:
            on_save()
    else:
        console.print(f"[yellow]Could not add {word!r} to profile.[/]")

    return True
