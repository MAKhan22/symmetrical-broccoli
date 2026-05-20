from pathlib import Path

import numpy as np
import pytest

from rwrt.config import PipelineConfig
from rwrt.index import EmbeddingIndex
from rwrt.vocabulary import VocabularyStore


def test_is_built_false_when_missing(cfg: PipelineConfig, vocab: VocabularyStore) -> None:
    index = EmbeddingIndex(cfg, vocab)
    assert not index.is_built


def test_load_raises_when_not_built(cfg: PipelineConfig, vocab: VocabularyStore) -> None:
    index = EmbeddingIndex(cfg, vocab)
    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        index.load()


def test_mean_embedding_normalized(mini_faiss_index: EmbeddingIndex) -> None:
    vec = mini_faiss_index.mean_embedding(["bir", "ev"])
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5


def test_mean_embedding_empty_raises(mini_faiss_index: EmbeddingIndex) -> None:
    with pytest.raises(ValueError, match="empty"):
        mini_faiss_index.mean_embedding([])


def test_embedding_for_unknown_word(mini_faiss_index: EmbeddingIndex) -> None:
    with pytest.raises(KeyError, match="not in index"):
        mini_faiss_index.embedding_for("yok")


def test_search_returns_scored_words(mini_faiss_index: EmbeddingIndex) -> None:
    query = mini_faiss_index.mean_embedding(["bir"])
    hits = mini_faiss_index.search(query, k=3)
    assert len(hits) == 3
    words = {w for w, _ in hits}
    assert "bir" in words
    assert all(isinstance(score, float) for _, score in hits)
