import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytest

from rwrt.config import PipelineConfig
from rwrt.index import EmbeddingIndex
from rwrt.vocabulary import VocabularyStore


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL UNIQUE,
                weight INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO nodes (word, weight) VALUES
                ('bir', 100), ('ev', 50), ('kitap', 10), ('nadir', 1);
            """
        )
    return db


@pytest.fixture
def cfg(tiny_db: Path, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        db_path=tiny_db,
        index_dir=tmp_path / "faiss",
        min_weight=1,
        retrieve_k=10,
        return_n=3,
        use_cross_encoder=False,
    ).resolve(tmp_path)


@pytest.fixture
def vocab(cfg: PipelineConfig) -> VocabularyStore:
    store = VocabularyStore(cfg)
    store.load()
    return store


@pytest.fixture
def mini_faiss_index(cfg: PipelineConfig, vocab: VocabularyStore) -> EmbeddingIndex:
    """Small pre-built FAISS index aligned with *vocab* (no ML models)."""
    words = vocab.words()
    dim = 16
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((len(words), dim)).astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(cfg.faiss_path))
    cfg.words_path.write_text("\n".join(words), encoding="utf-8")
    np.save(cfg.embeddings_path, embeddings)

    emb_index = EmbeddingIndex(cfg, vocab)
    emb_index.load()
    return emb_index
