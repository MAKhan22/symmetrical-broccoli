from pathlib import Path

import pytest

from rwrt.config import PipelineConfig
from rwrt.vocabulary import VocabularyStore


def test_vocabulary_respects_min_weight(tiny_db: Path) -> None:
    cfg = PipelineConfig(db_path=tiny_db, min_weight=10)
    vocab = VocabularyStore(cfg)
    vocab.load()

    assert set(vocab.words()) == {"bir", "ev", "kitap"}
    assert vocab.frequency("bir") == 100
    assert not vocab.contains("nadir")


def test_filter_known(tiny_db: Path) -> None:
    cfg = PipelineConfig(db_path=tiny_db, min_weight=1)
    vocab = VocabularyStore(cfg)
    vocab.load()

    known = vocab.filter_known({"bir", "missing", "nadir"})
    assert known == {"bir", "nadir"}


def test_top_n(tiny_db: Path) -> None:
    cfg = PipelineConfig(db_path=tiny_db, min_weight=1)
    vocab = VocabularyStore(cfg)
    vocab.load()

    assert vocab.top_n(2) == ["bir", "ev"]


def test_unknown_words_excludes_known(tiny_db: Path) -> None:
    cfg = PipelineConfig(db_path=tiny_db, min_weight=1)
    vocab = VocabularyStore(cfg)
    vocab.load()

    unknown = vocab.unknown_words({"bir", "ev"})
    assert "bir" not in unknown
    assert "ev" not in unknown
    assert "kitap" in unknown


def test_max_vocab_size_caps_list(tiny_db: Path) -> None:
    cfg = PipelineConfig(db_path=tiny_db, min_weight=1, max_vocab_size=2)
    vocab = VocabularyStore(cfg)
    vocab.load()

    assert vocab.words() == ["bir", "ev"]


def test_load_idempotent(tiny_db: Path) -> None:
    cfg = PipelineConfig(db_path=tiny_db, min_weight=1)
    vocab = VocabularyStore(cfg)
    vocab.load()
    vocab.load()
    assert len(vocab) == 4


def test_missing_database_raises(tmp_path: Path) -> None:
    cfg = PipelineConfig(db_path=tmp_path / "missing.sqlite")
    vocab = VocabularyStore(cfg)
    with pytest.raises(FileNotFoundError):
        vocab.load()
