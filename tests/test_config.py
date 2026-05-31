from pathlib import Path

from rwrt.config import DEFAULT_INDEX_DIR, PipelineConfig, model_slug


def test_model_slug() -> None:
    assert model_slug("intfloat/multilingual-e5-small") == "intfloat_multilingual_e5_small"


def test_resolve_creates_index_dir(tmp_path: Path, tiny_db: Path) -> None:
    cfg = PipelineConfig(db_path=tiny_db, index_dir=tmp_path / "idx")
    cfg.resolve(tmp_path)

    assert cfg.db_path == (tmp_path / tiny_db.name).resolve()
    assert cfg.index_dir == (tmp_path / "idx").resolve()
    assert cfg.index_dir.is_dir()


def test_artifact_paths_under_index_dir(tmp_path: Path) -> None:
    cfg = PipelineConfig(index_dir=tmp_path / "faiss").resolve(tmp_path)

    assert cfg.faiss_path == tmp_path / "faiss" / "index.faiss"
    assert cfg.words_path == tmp_path / "faiss" / "words.txt"
    assert cfg.embeddings_path == tmp_path / "faiss" / "embeddings.npy"
    assert cfg.meta_path == tmp_path / "faiss" / "meta.json"


def test_resolve_uses_per_model_subdir_when_present(tmp_path: Path) -> None:
    root = tmp_path / "project"
    faiss_root = root / DEFAULT_INDEX_DIR
    model_dir = faiss_root / model_slug(PipelineConfig().bi_model)
    model_dir.mkdir(parents=True)
    (model_dir / "index.faiss").write_bytes(b"stub")
    (model_dir / "meta.json").write_text(
        '{"max_vocab_size": 50000}', encoding="utf-8"
    )

    cfg = PipelineConfig(index_dir=DEFAULT_INDEX_DIR).resolve(root)

    assert cfg.index_dir == model_dir.resolve()
    assert cfg.max_vocab_size == 50000
