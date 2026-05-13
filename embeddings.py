"""Multilingual embeddings via FastEmbed (ONNX). Default: mpnet (Amharic-friendly)."""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding

warnings.filterwarnings(
    "ignore",
    message=".*mean pooling instead of CLS embedding.*",
    category=UserWarning,
)

# Strong multilingual retrieval; ~1 GB one-time ONNX download
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def env_model_name() -> str:
    return os.environ.get("RAG_EMBED_MODEL", DEFAULT_MODEL).strip()


def resolved_model(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return env_model_name()


def _e5_prefixes(name: str) -> bool:
    return "e5" in name.lower()


def _as_passage(text: str, name: str) -> str:
    t = (text or "").strip()[:8000]
    if _e5_prefixes(name):
        return f"passage: {t}"
    return t


def _as_query(q: str, name: str) -> str:
    t = (q or "").strip()[:8000]
    if _e5_prefixes(name):
        return f"query: {t}"
    return t


@lru_cache(maxsize=8)
def load_embedder(name: str) -> TextEmbedding:
    return TextEmbedding(model_name=name)


def _stack(embed_iter: Iterable[np.ndarray], dim_fallback_name: str) -> np.ndarray:
    arrs = [np.asarray(e, dtype=np.float32) for e in embed_iter]
    if not arrs:
        dim = TextEmbedding.get_embedding_size(dim_fallback_name)
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack(arrs)


def encode_passages(
    texts: list[str],
    batch_size: int = 64,
    model: str | None = None,
) -> np.ndarray:
    name = resolved_model(model)
    m = load_embedder(name)
    docs = [_as_passage(t, name) for t in texts]
    emb = _stack(m.embed(docs, batch_size=batch_size), dim_fallback_name=name)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return emb / norms


def encode_query(q: str, model: str | None = None) -> np.ndarray:
    name = resolved_model(model)
    m = load_embedder(name)
    emb = _stack(m.embed([_as_query(q, name)], batch_size=1), dim_fallback_name=name)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return emb / norms
