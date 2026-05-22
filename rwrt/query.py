from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from rwrt.index import EmbeddingIndex
    from rwrt.vocabulary import VocabularyStore

QueryWordSelection = Literal["diverse", "topic"]


def select_query_words(
    known: set[str],
    *,
    max_words: int,
    vocabulary: VocabularyStore | None = None,
    selection: QueryWordSelection = "diverse",
    topic_keyword: str | None = None,
    index: EmbeddingIndex | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """Pick known words to anchor bi-encoder / cross-encoder queries.

    ``diverse`` (default) mixes middle- and lower-frequency known words plus a
    small random sample so the query tracks recent learning instead of the
    learner's first high-frequency words.

    ``topic`` keeps known words whose embeddings are closest to *topic_keyword*.
    """
    if selection == "topic":
        if not topic_keyword or not topic_keyword.strip():
            raise ValueError("topic_keyword is required when query_word_selection='topic'")
        if index is None:
            raise ValueError("EmbeddingIndex is required for topic-guided query word selection")

    if max_words <= 0 or len(known) <= max_words:
        return sorted(known)

    if selection == "topic":
        return _topic_selection(
            known,
            max_words=max_words,
            topic_keyword=topic_keyword,
            index=index,
        )

    return _diverse_selection(
        known,
        max_words=max_words,
        vocabulary=vocabulary,
        rng=rng or random.Random(),
    )


def _diverse_selection(
    known: set[str],
    *,
    max_words: int,
    vocabulary: VocabularyStore | None,
    rng: random.Random,
) -> list[str]:
    if vocabulary is not None:
        vocabulary.ensure_loaded()
        ranked = sorted(
            known,
            key=lambda w: vocabulary.frequency(w) or 0,
            reverse=True,
        )
    else:
        ranked = sorted(known)

    n = len(ranked)
    third = max(1, n // 3)
    upper = ranked[:third]
    middle = ranked[third : 2 * third] if n > third else []
    lower = ranked[2 * third :] if n > 2 * third else ranked[-third:]

    # Cap high-frequency words; prioritize middle + lower bands and exploration.
    n_upper = min(len(upper), max(0, max_words // 5))
    remaining = max_words - n_upper
    n_middle = min(len(middle), remaining // 2)
    remaining -= n_middle
    n_lower = min(len(lower), remaining)
    remaining -= n_lower
    n_random = remaining

    selected: list[str] = []
    selected.extend(_sample(rng, upper, n_upper))
    selected.extend(_sample(rng, middle, n_middle))
    selected.extend(_sample(rng, lower, n_lower))

    pool = [w for w in ranked if w not in selected]
    selected.extend(_sample(rng, pool, n_random))

    return sorted(selected)


def _topic_selection(
    known: set[str],
    *,
    max_words: int,
    topic_keyword: str | None,
    index: EmbeddingIndex | None,
) -> list[str]:
    assert topic_keyword is not None and topic_keyword.strip()
    assert index is not None

    index.ensure_loaded()
    topic_vec = index.encode_text(topic_keyword.strip())

    scored: list[tuple[str, float]] = []
    for word in known:
        vec = index.embedding_for(word)
        scored.append((word, float(np.dot(vec, topic_vec))))

    scored.sort(key=lambda item: item[1], reverse=True)
    return sorted(word for word, _ in scored[:max_words])


def _sample(rng: random.Random, pool: list[str], count: int) -> list[str]:
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        return list(pool)
    return rng.sample(pool, count)
