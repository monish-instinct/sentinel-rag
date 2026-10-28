"""Lazy, thread-safe loading of the embedding and NLI models."""
from __future__ import annotations

import threading

import numpy as np

from .config import EMBED_MODEL, NLI_MODEL

_lock = threading.Lock()
_embedder = None
_nli = None
_nli_labels: list[str] | None = None


def embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        return _embedder


def tokenizer():
    return embedder().tokenizer


def max_seq_length() -> int:
    return int(embedder().max_seq_length)


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, so a dot product equals cosine similarity."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    vecs = embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def nli():
    global _nli, _nli_labels
    with _lock:
        if _nli is None:
            from sentence_transformers import CrossEncoder

            _nli = CrossEncoder(NLI_MODEL, device="cpu")
            id2label = _nli.model.config.id2label
            _nli_labels = [id2label[i].lower() for i in range(len(id2label))]
        return _nli


def nli_probs(pairs: list[tuple[str, str]]) -> list[dict]:
    """Return {'entailment', 'contradiction', 'neutral'} probabilities per (premise, hypothesis)."""
    if not pairs:
        return []
    model = nli()
    logits = np.asarray(model.predict(pairs, batch_size=32, show_progress_bar=False), dtype="float64")
    if logits.ndim == 1:
        logits = logits[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return [{lab: float(p[i]) for i, lab in enumerate(_nli_labels)} for p in probs]


def warm_up():
    embed(["warm up"])
    nli_probs([("The sky is blue.", "The sky has a colour.")])
