from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

# Keep CLI output clean: HF Hub token nags and transformers load reports go to stderr.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_bi_cache: dict[tuple[str, str], SentenceTransformer] = {}
_cross_cache: dict[tuple[str, str], CrossEncoder] = {}


@contextmanager
def _quiet_model_load():
    """Suppress progress bars and weight-load reports during model init."""
    for name in ("transformers", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def get_bi_encoder(model_name: str, device: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    key = (model_name, device)
    if key not in _bi_cache:
        with _quiet_model_load():
            _bi_cache[key] = SentenceTransformer(model_name, device=device)
    return _bi_cache[key]


def get_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    key = (model_name, device)
    if key not in _cross_cache:
        with _quiet_model_load():
            _cross_cache[key] = CrossEncoder(model_name, device=device)
    return _cross_cache[key]


def clear_encoder_cache() -> None:
    """Release cached models (useful in tests)."""
    _bi_cache.clear()
    _cross_cache.clear()
