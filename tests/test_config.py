from pathlib import Path

from rwrt.config import PipelineConfig


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
