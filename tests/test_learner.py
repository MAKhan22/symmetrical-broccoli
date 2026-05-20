import json
from pathlib import Path

import pytest

from rwrt.config import PipelineConfig
from rwrt.learner import LearnerProfile
from rwrt.vocabulary import VocabularyStore


def test_add_and_remove(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab)
    assert profile.add("bir")
    assert profile.add("missing") is False
    assert "bir" in profile
    assert len(profile) == 1
    assert profile.remove("bir")
    assert len(profile) == 0


def test_add_many_reports_rejected(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab)
    added, rejected = profile.add_many(["bir", "ev", "nope"])
    assert added == 2
    assert rejected == ["nope"]


def test_learn_alias(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab)
    profile.learn("kitap")
    assert "kitap" in profile


def test_save_and_load(tmp_path: Path, vocab: VocabularyStore) -> None:
    path = tmp_path / "learner.json"
    profile = LearnerProfile(vocabulary=vocab, path=path)
    profile.add_many(["bir", "ev"])
    profile.save()

    loaded = LearnerProfile.load(path, vocabulary=vocab)
    assert loaded.known == frozenset({"bir", "ev"})


def test_load_or_create_missing(tmp_path: Path, vocab: VocabularyStore) -> None:
    path = tmp_path / "new.json"
    profile = LearnerProfile.load_or_create(path, vocabulary=vocab)
    assert len(profile) == 0
    profile.add("bir")
    profile.save(path)
    assert path.is_file()


def test_prune_invalid(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(known=["bir", "ghost"], vocabulary=None)
    profile.bind_vocabulary(vocab)
    removed = profile.prune_invalid(vocab)
    assert removed == 1
    assert profile.known == frozenset({"bir"})


def test_validate_against(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(known=["bir", "ghost"], vocabulary=vocab)
    in_vocab, oov = profile.validate_against(vocab)
    assert in_vocab == {"bir"}
    assert oov == {"ghost"}


def test_clear_and_update(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir"])
    profile.clear()
    assert len(profile) == 0
    profile.update(["ev", "kitap"])
    assert profile.known == frozenset({"ev", "kitap"})


def test_remove_many(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir", "ev", "kitap"])
    removed = profile.remove_many(["bir", "missing", "ev"])
    assert removed == 2
    assert profile.known == frozenset({"kitap"})


def test_load_legacy_list_format(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text('["bir", "ev"]', encoding="utf-8")
    profile = LearnerProfile.load(path)
    assert profile.known == frozenset({"bir", "ev"})


def test_add_rejects_whitespace(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab)
    assert profile.add("   ") is False
    assert len(profile) == 0


def test_save_requires_path_when_unset(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab)
    profile.add("bir")
    with pytest.raises(ValueError, match="No path"):
        profile.save()
