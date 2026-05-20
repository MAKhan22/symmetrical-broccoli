from unittest.mock import MagicMock

import pytest

from rwrt.config import PipelineConfig
from rwrt.index import EmbeddingIndex
from rwrt.learner import LearnerProfile
from rwrt.pipeline import RecommendationPipeline, _known_set
from rwrt.types import Candidate
from rwrt.vocabulary import VocabularyStore


def test_known_set_from_profile(vocab: VocabularyStore) -> None:
    profile = LearnerProfile(vocabulary=vocab, known=["bir", "ev"])
    assert _known_set(profile) == {"bir", "ev"}


def test_frequency_boost_reorders_by_freq(cfg: PipelineConfig, vocab: VocabularyStore) -> None:
    cfg.frequency_boost = 1.0
    pipe = RecommendationPipeline(cfg, vocab)
    candidates = [
        Candidate(word="nadir", bi_score=0.99, frequency=1),
        Candidate(word="bir", bi_score=0.10, frequency=100),
    ]
    boosted = pipe._apply_frequency_boost(candidates)
    assert boosted[0].word == "bir"


def test_recommend_bi_only(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    cfg.use_cross_encoder = False
    pipe = RecommendationPipeline(cfg, vocab, index=mini_faiss_index)
    results = pipe.recommend({"bir"}, return_n=2, use_cross_encoder=False)

    assert len(results) == 2
    assert "bir" not in {c.word for c in results}


def test_recommend_with_mocked_reranker(cfg: PipelineConfig, vocab: VocabularyStore) -> None:
    pipe = RecommendationPipeline(cfg, vocab)
    pipe._retriever.retrieve = MagicMock(
        return_value=[
            Candidate(word="kitap", bi_score=0.5, frequency=10),
            Candidate(word="ev", bi_score=0.4, frequency=50),
        ]
    )
    pipe._reranker.rerank = MagicMock(
        return_value=[Candidate(word="ev", bi_score=0.4, cross_score=0.99, frequency=50)]
    )

    results = pipe.recommend({"bir"}, return_n=1, use_cross_encoder=True)
    assert results[0].word == "ev"
    pipe._reranker.rerank.assert_called_once()
