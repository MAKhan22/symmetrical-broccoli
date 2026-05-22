from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from rwrt.config import PipelineConfig
from rwrt.encoders import get_bi_encoder
from rwrt.vocabulary import VocabularyStore


class EmbeddingIndex:
    """FAISS index over bi-encoder embeddings of the vocabulary."""

    def __init__(
        self,
        config: PipelineConfig,
        vocabulary: VocabularyStore,
    ) -> None:
        self._config = config
        self._vocabulary = vocabulary
        self._index: faiss.Index | None = None
        self._embeddings: np.ndarray | None = None
        self._words: list[str] = []
        self._word_to_idx: dict[str, int] = {}

    @property
    def words(self) -> list[str]:
        return self._words

    @property
    def is_built(self) -> bool:
        return self._config.faiss_path.is_file() and self._config.words_path.is_file()

    def build(self, *, save_embeddings: bool = True) -> None:
        """Embed all vocabulary words and persist a FAISS index to disk."""
        self._vocabulary.ensure_loaded()
        words = self._vocabulary.words()
        if not words:
            raise ValueError("Vocabulary is empty; check min_weight / database path.")

        model = get_bi_encoder(self._config.bi_model, self._config.device)
        embeddings = model.encode(
            words,
            batch_size=self._config.encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self._config.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._config.faiss_path))
        self._config.words_path.write_text("\n".join(words), encoding="utf-8")

        #TODO: try different models and check performance
        meta = {
            "bi_model": self._config.bi_model,
            "min_weight": self._config.min_weight,
            "max_vocab_size": self._config.max_vocab_size,
            "num_vectors": len(words),
            "dimension": embeddings.shape[1],
        }
        (self._config.meta_path).write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

        if save_embeddings:
            np.save(self._config.embeddings_path, embeddings)

        self._attach(words, embeddings, index)

    def load(self) -> None:
        if not self.is_built:
            raise FileNotFoundError(
                f"FAISS index not found under {self._config.index_dir}. "
                "Run `rwrt-build-index` or EmbeddingIndex.build() first."
            )

        words = self._config.words_path.read_text(encoding="utf-8").splitlines()
        index = faiss.read_index(str(self._config.faiss_path))

        embeddings: np.ndarray | None = None
        if self._config.embeddings_path.is_file():
            embeddings = np.load(self._config.embeddings_path)

        self._validate_meta(expected_vectors=len(words))
        self._attach(words, embeddings, index)

    def ensure_loaded(self) -> None:
        if self._index is None:
            self.load()

    def _validate_meta(self, *, expected_vectors: int) -> None:
        meta_path = self._config.meta_path
        if not meta_path.is_file():
            return

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mismatches: list[str] = []

        if meta.get("bi_model") != self._config.bi_model:
            mismatches.append(
                f"bi_model: index={meta.get('bi_model')!r} "
                f"config={self._config.bi_model!r}"
            )
        if meta.get("min_weight") != self._config.min_weight:
            mismatches.append(
                f"min_weight: index={meta.get('min_weight')!r} "
                f"config={self._config.min_weight!r}"
            )
        if meta.get("max_vocab_size") != self._config.max_vocab_size:
            mismatches.append(
                f"max_vocab_size: index={meta.get('max_vocab_size')!r} "
                f"config={self._config.max_vocab_size!r}"
            )
        if meta.get("num_vectors") != expected_vectors:
            mismatches.append(
                f"num_vectors: meta={meta.get('num_vectors')!r} "
                f"words.txt={expected_vectors!r}"
            )

        if mismatches:
            detail = "\n  ".join(mismatches)
            raise ValueError(
                f"Index metadata does not match current configuration:\n  {detail}\n"
                "Rebuild with `rwrt-build-index`."
            )

    def _attach(
        self,
        words: list[str],
        embeddings: np.ndarray | None,
        index: faiss.Index,
    ) -> None:
        self._words = words
        self._word_to_idx = {w: i for i, w in enumerate(words)}
        self._index = index
        self._embeddings = embeddings

    def embedding_for(self, word: str) -> np.ndarray:
        self.ensure_loaded()
        assert self._index is not None
        idx = self._word_to_idx.get(word)
        if idx is None:
            raise KeyError(f"Word not in index: {word!r}")
        if self._embeddings is not None:
            return self._embeddings[idx].copy()
        return self._index.reconstruct(idx)

    def encode_text(self, text: str) -> np.ndarray:
        """Embed free text with the bi-encoder (e.g. a topic keyword)."""
        model = get_bi_encoder(self._config.bi_model, self._config.device)
        vec = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def mean_embedding(self, words: list[str]) -> np.ndarray:
        """Calculate the mean embedding of a list of words."""
        if not words:
            raise ValueError("Cannot build query embedding from an empty known-word set.")
        vectors = np.stack([self.embedding_for(w) for w in words], axis=0)
        mean = vectors.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean.astype(np.float32)

    def weighted_mean_embedding(
        self,
        words: list[str],
        weights: dict[str, float],
    ) -> np.ndarray:
        """Frequency-weighted mean embedding, L2-normalized."""
        if not words:
            raise ValueError("Cannot build query embedding from an empty known-word set.")
        vectors = np.stack([self.embedding_for(w) for w in words], axis=0)
        w = np.array([weights.get(word, 1.0) for word in words], dtype=np.float32)
        total = w.sum()
        if total <= 0:
            return self.mean_embedding(words)
        w = w / total
        mean = (vectors * w[:, np.newaxis]).sum(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean.astype(np.float32)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        self.ensure_loaded()
        assert self._index is not None

        query = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, min(k, len(self._words)))

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            results.append((self._words[idx], float(score)))
        return results
