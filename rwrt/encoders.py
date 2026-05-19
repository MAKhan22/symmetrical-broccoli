from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

_bi_cache: dict[tuple[str, str], SentenceTransformer] = {}
_cross_cache: dict[tuple[str, str], CrossEncoder] = {}


def get_bi_encoder(model_name: str, device: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    key = (model_name, device)
    if key not in _bi_cache:
        _bi_cache[key] = SentenceTransformer(model_name, device=device)
    return _bi_cache[key]


def get_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    key = (model_name, device)
    if key not in _cross_cache:
        _cross_cache[key] = CrossEncoder(model_name, device=device)
    return _cross_cache[key]


def clear_encoder_cache() -> None:
    """Release cached models (useful in tests)."""
    _bi_cache.clear()
    _cross_cache.clear()
