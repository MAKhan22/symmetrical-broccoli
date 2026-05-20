import pytest

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
