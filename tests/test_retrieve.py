import pytest
from unittest.mock import patch

from rwrt.config import PipelineConfig
from rwrt.index import EmbeddingIndex
from rwrt.retrieve import BiEncoderRetriever
from rwrt.vocabulary import VocabularyStore

def test_retrieve_excludes_known_words(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    results = retriever.retrieve({"bir"}, k=5)

    assert len(results) <= 5
    assert all(c.word != "bir" for c in results)
    assert all(c.bi_score is not None for c in results)
    assert all(c.frequency is not None for c in results)


def test_retrieve_no_known_in_vocab_raises(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    with pytest.raises(ValueError, match="None of the known words"):
        retriever.retrieve({"nonexistent"})


def test_retrieve_respects_k(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    results = retriever.retrieve({"bir", "ev"}, k=2)
    assert len(results) == 2


def test_retrieve_weighted_mean_strategy(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    cfg.query_strategy = "weighted_mean"
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    results = retriever.retrieve({"bir"}, k=3)
    assert len(results) == 3
    assert all(c.word != "bir" for c in results)


def test_retrieve_inverse_weighted_mean_upweights_rare_words(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    cfg.query_strategy = "inverse_weighted_mean"
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    with patch.object(
        mini_faiss_index,
        "weighted_mean_embedding",
        wraps=mini_faiss_index.weighted_mean_embedding,
    ) as mock_weighted:
        retriever.retrieve({"bir", "nadir"}, k=2)
    weights = mock_weighted.call_args[0][1]
    assert weights["nadir"] > weights["bir"]


def test_retrieve_topic_strategy_uses_keyword_embedding(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    cfg.query_strategy = "topic"
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    topic_vec = mini_faiss_index.embedding_for("ev")
    with patch.object(
        mini_faiss_index,
        "encode_text",
        return_value=topic_vec,
    ) as mock_encode:
        with patch.object(mini_faiss_index, "mean_embedding") as mock_mean:
            retriever.retrieve({"bir"}, k=2, topic_keyword="yemek")
    mock_encode.assert_called_once_with("yemek")
    mock_mean.assert_not_called()


def test_retrieve_topic_strategy_requires_keyword(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    cfg.query_strategy = "topic"
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    with pytest.raises(ValueError, match="topic_keyword is required"):
        retriever.retrieve({"bir"}, k=2)


def test_retrieve_subsamples_known_words_for_query(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    cfg.max_query_words = 2
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    with patch.object(
        mini_faiss_index,
        "mean_embedding",
        wraps=mini_faiss_index.mean_embedding,
    ) as mock_mean:
        retriever.retrieve({"bir", "ev", "kitap", "nadir"}, k=2)
    query_words = mock_mean.call_args[0][0]
    assert "bir" not in query_words
    assert len(query_words) == 2


def test_retrieve_unsupported_strategy_raises(
    cfg: PipelineConfig, vocab: VocabularyStore, mini_faiss_index: EmbeddingIndex
) -> None:
    cfg.query_strategy = "unsupported"  # type: ignore[assignment]
    retriever = BiEncoderRetriever(cfg, vocab, mini_faiss_index)
    with pytest.raises(ValueError, match="Unsupported query_strategy"):
        retriever.retrieve({"bir"}, k=2)
