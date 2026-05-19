from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class PipelineConfig:
    """Configuration for the rwrt recommendation pipeline."""

    db_path: Path = Path("primary_graph.sqlite")
    index_dir: Path = Path("data/faiss")

    min_weight: int = 10
    max_vocab_size: int | None = None

    bi_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    cross_model: str = "dbmdz/bert-base-turkish-cased"

    device: str = field(default_factory=_default_device)
    encode_batch_size: int = 256

    retrieve_k: int = 200
    return_n: int = 5
    use_cross_encoder: bool = True

    query_strategy: Literal["mean_embedding"] = "mean_embedding"
    frequency_boost: float = 0.0

    max_query_words: int = 64
    cross_batch_size: int = 32

    def resolve(self, base: Path | None = None) -> PipelineConfig:
        """Resolve relative paths against *base* (defaults to cwd)."""
        root = base or Path.cwd()
        self.db_path = (root / self.db_path).resolve()
        self.index_dir = (root / self.index_dir).resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def words_path(self) -> Path:
        return self.index_dir / "words.txt"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"
