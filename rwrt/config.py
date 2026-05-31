from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RerankerType = Literal["weighted_feature", "cross_encoder"]

DEFAULT_INDEX_DIR = Path("data/faiss")


def model_slug(model_name: str) -> str:
    """Filesystem-safe name for a HuggingFace model id (matches alpha.ipynb)."""
    return model_name.replace("/", "_").replace("-", "_")


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
    index_dir: Path = DEFAULT_INDEX_DIR

    min_weight: int = 10
    max_vocab_size: int | None = None

    bi_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    cross_model: str = "dbmdz/bert-base-turkish-cased"

    device: str = field(default_factory=_default_device)
    encode_batch_size: int = 256

    retrieve_k: int = 200
    return_n: int = 5
    use_cross_encoder: bool = True
    reranker_type: RerankerType = "weighted_feature"

    query_strategy: Literal[
        "mean_embedding",
        "weighted_mean",
        "inverse_weighted_mean",
        "topic",
    ] = "mean_embedding"
    query_word_selection: Literal["diverse", "topic"] = "diverse"
    topic_keyword: str | None = None
    frequency_boost: float = 0.0

    max_query_words: int = 64
    cross_batch_size: int = 32

    # Weights for weighted_feature reranker
    weight_semantic: float = 0.35
    weight_frequency: float = 0.35
    weight_morphology: float = 0.10
    weight_diversity: float = 0.15
    weight_difficulty: float = 0.05

    def resolve(self, base: Path | None = None) -> PipelineConfig:
        """Resolve relative paths against *base* (defaults to cwd).

        Index layout:
        - If ``index_dir/index.faiss`` exists, use that directory.
        - Else if ``index_dir/<model_slug>/index.faiss`` exists, use the subdir
          (layout produced by alpha.ipynb and default ``rwrt-build-index``).
        - Else if ``index_dir`` is the default ``data/faiss``, target the
          per-model subdir for builds.
        """
        root = base or Path.cwd()
        self.db_path = (root / self.db_path).resolve()

        requested = (root / self.index_dir).resolve()
        requested.mkdir(parents=True, exist_ok=True)

        flat_index = requested / "index.faiss"
        model_dir = requested / model_slug(self.bi_model)
        model_index = model_dir / "index.faiss"
        default_faiss_root = (root / DEFAULT_INDEX_DIR).resolve()

        if flat_index.is_file():
            self.index_dir = requested
        elif model_index.is_file():
            self.index_dir = model_dir
        elif requested == default_faiss_root:
            self.index_dir = model_dir
            self.index_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.index_dir = requested

        self._sync_index_meta()
        return self

    def _sync_index_meta(self) -> None:
        """Align unset vocab caps with an existing index's meta.json."""
        if self.max_vocab_size is not None or not self.meta_path.is_file():
            return
        import json

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("max_vocab_size") is not None:
            self.max_vocab_size = meta["max_vocab_size"]

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def words_path(self) -> Path:
        return self.index_dir / "words.txt"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    @property
    def meta_path(self) -> Path:
        return self.index_dir / "meta.json"
