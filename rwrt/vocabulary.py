from __future__ import annotations

import sqlite3
from pathlib import Path

from rwrt.config import PipelineConfig


class VocabularyStore:
    """Word-frequency vocabulary backed by the primary graph SQLite database."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._db_path = Path(config.db_path)
        self._words: list[str] = []
        self._frequencies: dict[str, int] = {}
        self._loaded = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def load(self) -> None:
        if self._loaded:
            return
        if not self._db_path.is_file():
            raise FileNotFoundError(f"Vocabulary database not found: {self._db_path}")

        query = """
            SELECT word, weight
            FROM nodes
            WHERE weight >= ?
            ORDER BY weight DESC
        """
        params: list[object] = [self._config.min_weight]
        if self._config.max_vocab_size is not None:
            query += " LIMIT ?"
            params.append(self._config.max_vocab_size)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        self._words = [word for word, _ in rows]
        self._frequencies = {word: int(weight) for word, weight in rows}
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def __len__(self) -> int:
        self.ensure_loaded()
        return len(self._words)

    def words(self) -> list[str]:
        self.ensure_loaded()
        return list(self._words)

    def frequency(self, word: str) -> int | None:
        self.ensure_loaded()
        return self._frequencies.get(word)

    def contains(self, word: str) -> bool:
        self.ensure_loaded()
        return word in self._frequencies

    def top_n(self, n: int) -> list[str]:
        self.ensure_loaded()
        return self._words[:n]

    def filter_known(self, known: set[str]) -> set[str]:
        self.ensure_loaded()
        return {w for w in known if w in self._frequencies}

    def unknown_words(self, known: set[str]) -> list[str]:
        self.ensure_loaded()
        return [w for w in self._words if w not in known]
