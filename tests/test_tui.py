from unittest.mock import MagicMock

import pytest
from rich.console import Console

from rwrt.learner import LearnerProfile
from rwrt.llm import LLMConfig, OpenRouterClient
from rwrt.tui import SuggestionChoice, parse_suggestion_input, run_chat_session
from rwrt.types import Candidate
from rwrt.vocabulary import VocabularyStore


def test_parse_suggestion_input_mixed_known_and_study() -> None:
    parsed = parse_suggestion_input("1k, 2k, 3k, 4", max_n=5)
    assert parsed == SuggestionChoice(mark_known=frozenset({1, 2, 3}), study=4)


def test_parse_suggestion_input_known_only() -> None:
    assert parse_suggestion_input("1k,2k,3k", max_n=5) == SuggestionChoice(
        mark_known=frozenset({1, 2, 3}), study=None
    )


def test_parse_suggestion_input_study_only() -> None:
    assert parse_suggestion_input("4", max_n=5) == SuggestionChoice(
        mark_known=frozenset(), study=4
    )


def test_parse_suggestion_input_rejects_multiple_study() -> None:
    with pytest.raises(ValueError, match="only one word to study"):
        parse_suggestion_input("2, 4", max_n=5)


def test_parse_suggestion_input_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="Out of range"):
        parse_suggestion_input("6k", max_n=5)


def test_chat_session_learns_after_explanation(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir"])
    pipe = MagicMock()
    pipe.recommend.return_value = [
        Candidate(word="ev", bi_score=0.5, frequency=50),
    ]

    llm_cfg = LLMConfig(api_key="test", base_url="https://example.com/api/v1")
    llm = OpenRouterClient(llm_cfg)
    llm.explain_word = MagicMock(return_value="**English meaning**: house")
    llm.follow_up = MagicMock()

    saved: list[bool] = []
    inputs = iter(["1", "", "q"])
    console = Console(file=open("/dev/null", "w"), force_terminal=True, width=80)

    code = run_chat_session(
        profile,
        pipe,
        llm,
        return_n=1,
        on_save=lambda: saved.append(True),
        console=console,
        input_fn=lambda _: next(inputs),
    )

    assert code == 0
    assert "ev" in profile.known
    assert saved == [True]
    llm.explain_word.assert_called_once()


def test_chat_session_back_skips_learn(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir"])
    pipe = MagicMock()
    pipe.recommend.return_value = [
        Candidate(word="ev", bi_score=0.5, frequency=50),
    ]

    llm_cfg = LLMConfig(api_key="test", base_url="https://example.com/api/v1")
    llm = OpenRouterClient(llm_cfg)
    llm.explain_word = MagicMock(return_value="Explanation")
    llm.follow_up = MagicMock()

    inputs = iter(["1", "b", "q"])
    console = Console(file=open("/dev/null", "w"), force_terminal=True, width=80)

    code = run_chat_session(
        profile,
        pipe,
        llm,
        return_n=1,
        console=console,
        input_fn=lambda _: next(inputs),
    )

    assert code == 0
    assert "ev" not in profile.known


def test_chat_session_mark_known_and_study(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir"])
    pipe = MagicMock()
    pipe.recommend.return_value = [
        Candidate(word="ev", bi_score=0.5, frequency=50),
        Candidate(word="kitap", bi_score=0.4, frequency=10),
        Candidate(word="nadir", bi_score=0.3, frequency=1),
    ]

    llm_cfg = LLMConfig(api_key="test", base_url="https://example.com/api/v1")
    llm = OpenRouterClient(llm_cfg)
    llm.explain_word = MagicMock(return_value="Explanation for nadir")
    llm.follow_up = MagicMock()

    saved: list[bool] = []
    inputs = iter(["1k, 2k, 3", "", "q"])
    console = Console(file=open("/dev/null", "w"), force_terminal=True, width=80)

    code = run_chat_session(
        profile,
        pipe,
        llm,
        return_n=3,
        on_save=lambda: saved.append(True),
        console=console,
        input_fn=lambda _: next(inputs),
    )

    assert code == 0
    assert "ev" in profile.known
    assert "kitap" in profile.known
    assert "nadir" in profile.known
    assert saved == [True, True]
    llm.explain_word.assert_called_once()
    assert llm.explain_word.call_args[0][0] == "nadir"
