from unittest.mock import MagicMock

import pytest

from rwrt.config import PipelineConfig
from rwrt.rerank import CrossEncoderReranker
from rwrt.types import Candidate
from rwrt.vocabulary import VocabularyStore


def test_format_query_sorted() -> None:
    cfg = PipelineConfig(max_query_words=64)
    reranker = CrossEncoderReranker(cfg)
    q = reranker.format_query({"ev", "bir"})
    assert q == "Bildiğim Türkçe kelimeler: bir, ev"


def test_format_query_truncates_with_diverse_selection(vocab: VocabularyStore) -> None:
    cfg = PipelineConfig(max_query_words=2)
    reranker = CrossEncoderReranker(cfg, vocabulary=vocab)
    q = reranker.format_query({"bir", "ev", "kitap", "nadir"})
    assert "bir" not in q
    assert q.count(",") == 1


def test_rerank_empty_candidates() -> None:
    reranker = CrossEncoderReranker(PipelineConfig())
    assert reranker.rerank({"bir"}, [], n=5) == []


def test_rerank_uses_model_scores() -> None:
    cfg = PipelineConfig(return_n=2)
    reranker = CrossEncoderReranker(cfg)
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.9, 0.5]
    reranker._model = mock_model

    candidates = [
        Candidate(word="ev", bi_score=0.8),
        Candidate(word="kitap", bi_score=0.7),
        Candidate(word="nadir", bi_score=0.6),
    ]
    ranked = reranker.rerank({"bir"}, candidates, n=2)

    assert [c.word for c in ranked] == ["kitap", "nadir"]
    assert ranked[0].cross_score == 0.9
    mock_model.predict.assert_called_once()
