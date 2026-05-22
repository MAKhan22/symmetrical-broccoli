from unittest.mock import MagicMock

from rwrt.interactive import run_learn_session
from rwrt.learner import LearnerProfile
from rwrt.types import Candidate
from rwrt.vocabulary import VocabularyStore


def test_learn_session_learns_selected_word(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir"])
    pipe = MagicMock()
    pipe.recommend.return_value = [
        Candidate(word="ev", bi_score=0.5, frequency=50),
        Candidate(word="kitap", bi_score=0.4, frequency=10),
    ]
    saved: list[bool] = []
    inputs = iter(["1", "q"])

    code = run_learn_session(
        profile,
        pipe,
        return_n=2,
        on_save=lambda: saved.append(True),
        input_fn=lambda _: next(inputs),
        print_fn=lambda *args, **kwargs: None,
    )

    assert code == 0
    assert "ev" in profile.known
    assert saved == [True]
    assert pipe.recommend.call_count == 2


def test_learn_session_refresh_skips_save(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir"])
    pipe = MagicMock()
    pipe.recommend.return_value = [Candidate(word="ev", bi_score=0.5, frequency=50)]
    saved: list[bool] = []
    inputs = iter(["r", "q"])

    code = run_learn_session(
        profile,
        pipe,
        return_n=1,
        on_save=lambda: saved.append(True),
        input_fn=lambda _: next(inputs),
        print_fn=lambda *args, **kwargs: None,
    )

    assert code == 0
    assert saved == []
    assert pipe.recommend.call_count == 2


def test_learn_session_returns_error_on_recommend_failure(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir"])
    pipe = MagicMock()
    pipe.recommend.side_effect = ValueError("boom")

    code = run_learn_session(
        profile,
        pipe,
        return_n=1,
        input_fn=lambda _: "q",
        print_fn=lambda *args, **kwargs: None,
    )

    assert code == 1
