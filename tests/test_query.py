import random
from unittest.mock import patch

import pytest

from rwrt.index import EmbeddingIndex
from rwrt.query import select_query_words
from rwrt.vocabulary import VocabularyStore


def test_select_query_words_returns_all_when_under_cap(vocab: VocabularyStore) -> None:
    selected = select_query_words({"bir", "ev"}, max_words=64, vocabulary=vocab)
    assert selected == ["bir", "ev"]


def test_diverse_selection_avoids_only_top_frequency(vocab: VocabularyStore) -> None:
    rng = random.Random(0)
    selected = select_query_words(
        {"bir", "ev", "kitap", "nadir"},
        max_words=2,
        vocabulary=vocab,
        selection="diverse",
        rng=rng,
    )
    assert "bir" not in selected
    assert len(selected) == 2


def test_diverse_selection_uses_middle_and_lower_bands(vocab: VocabularyStore) -> None:
    rng = random.Random(1)
    selected = select_query_words(
        {"bir", "ev", "kitap", "nadir"},
        max_words=3,
        vocabulary=vocab,
        selection="diverse",
        rng=rng,
    )
    assert "bir" not in selected
    assert "ev" in selected
    assert len(selected) == 3


def test_diverse_selection_without_vocabulary() -> None:
    rng = random.Random(0)
    selected = select_query_words(
        {"a", "b", "c", "d", "e", "f"},
        max_words=3,
        vocabulary=None,
        selection="diverse",
        rng=rng,
    )
    assert len(selected) == 3


def test_select_query_words_invalid_cap_returns_all() -> None:
    selected = select_query_words({"bir", "ev"}, max_words=0, vocabulary=None)
    assert selected == ["bir", "ev"]


def test_topic_selection_picks_semantically_close_known_words(
    mini_faiss_index: EmbeddingIndex,
) -> None:
    with patch.object(
        mini_faiss_index,
        "encode_text",
        return_value=mini_faiss_index.embedding_for("ev"),
    ):
        selected = select_query_words(
            {"bir", "ev", "kitap", "nadir"},
            max_words=2,
            selection="topic",
            topic_keyword="ev",
            index=mini_faiss_index,
        )
    assert "ev" in selected
    assert len(selected) == 2


def test_topic_selection_requires_keyword(mini_faiss_index: EmbeddingIndex) -> None:
    with pytest.raises(ValueError, match="topic_keyword is required"):
        select_query_words(
            {"bir", "ev"},
            max_words=2,
            selection="topic",
            topic_keyword=None,
            index=mini_faiss_index,
        )


def test_topic_selection_requires_index() -> None:
    with pytest.raises(ValueError, match="EmbeddingIndex is required"):
        select_query_words(
            {"bir", "ev"},
            max_words=2,
            selection="topic",
            topic_keyword="food",
            index=None,
        )
