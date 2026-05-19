from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rwrt.vocabulary import VocabularyStore


class LearnerProfile:
    """Per-learner set of known words with optional persistence and vocabulary validation."""

    def __init__(
        self,
        *,
        vocabulary: VocabularyStore | None = None,
        path: Path | None = None,
        known: Iterable[str] | None = None,
    ) -> None:
        self._vocabulary = vocabulary
        self._path = Path(path) if path is not None else None
        self._known: set[str] = set()
        if known is not None:
            self.add_many(known, validate=False)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def known(self) -> frozenset[str]:
        return frozenset(self._known)

    def __len__(self) -> int:
        return len(self._known)

    def __contains__(self, word: str) -> bool:
        return word in self._known

    def bind_vocabulary(self, vocabulary: VocabularyStore) -> None:
        self._vocabulary = vocabulary

    def _is_valid(self, word: str) -> bool:
        if self._vocabulary is None:
            return True
        self._vocabulary.ensure_loaded()
        return self._vocabulary.contains(word)

    def add(self, word: str, *, validate: bool = True) -> bool:
        """Add a word. Returns False if rejected (e.g. not in corpus when validating)."""
        word = word.strip()
        if not word:
            return False
        if validate and not self._is_valid(word):
            return False
        self._known.add(word)
        return True

    def learn(self, word: str, *, validate: bool = True) -> bool:
        """Mark a word as known (alias for :meth:`add`)."""
        return self.add(word, validate=validate)

    def remove(self, word: str) -> bool:
        """Remove a word from the known set. Returns True if it was present."""
        word = word.strip()
        if word not in self._known:
            return False
        self._known.remove(word)
        return True

    def add_many(
        self,
        words: Iterable[str],
        *,
        validate: bool = True,
    ) -> tuple[int, list[str]]:
        """Add many words. Returns (count added, list of rejected words)."""
        added = 0
        rejected: list[str] = []
        for word in words:
            w = word.strip() if isinstance(word, str) else str(word).strip()
            if not w:
                continue
            if self.add(w, validate=validate):
                added += 1
            else:
                rejected.append(w)
        return added, rejected

    def remove_many(self, words: Iterable[str]) -> int:
        """Remove many words. Returns how many were removed."""
        removed = 0
        for word in words:
            if self.remove(word.strip() if isinstance(word, str) else str(word).strip()):
                removed += 1
        return removed

    def clear(self) -> None:
        self._known.clear()

    def update(self, words: Iterable[str], *, validate: bool = True) -> tuple[int, list[str]]:
        """Replace the known set with *words*."""
        self.clear()
        return self.add_many(words, validate=validate)

    def validate_against(self, vocabulary: VocabularyStore) -> tuple[set[str], set[str]]:
        """Split known words into (in_vocab, out_of_vocab)."""
        vocabulary.ensure_loaded()
        in_vocab = vocabulary.filter_known(self._known)
        out_of_vocab = self._known - in_vocab
        return in_vocab, out_of_vocab

    def prune_invalid(self, vocabulary: VocabularyStore) -> int:
        """Drop known words not in *vocabulary*. Returns count removed."""
        _, invalid = self.validate_against(vocabulary)
        for word in invalid:
            self._known.discard(word)
        return len(invalid)

    def save(self, path: Path | None = None) -> Path:
        """Persist known words to JSON. Returns the path written."""
        dest = Path(path) if path is not None else self._path
        if dest is None:
            raise ValueError("No path given and profile has no default path.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "known": sorted(self._known),
        }
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._path = dest
        return dest

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        vocabulary: VocabularyStore | None = None,
        prune_invalid: bool = False,
    ) -> LearnerProfile:
        """Load a profile from JSON."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Learner profile not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            known = data
        else:
            known = data.get("known", [])

        profile = cls(vocabulary=vocabulary, path=path, known=known)
        if prune_invalid and vocabulary is not None:
            profile.prune_invalid(vocabulary)
        return profile

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        vocabulary: VocabularyStore | None = None,
    ) -> LearnerProfile:
        path = Path(path)
        if path.is_file():
            return cls.load(path, vocabulary=vocabulary)
        return cls(vocabulary=vocabulary, path=path)
