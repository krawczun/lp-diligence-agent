"""Cross-encoder reranking for retrieval.

The vector search in :mod:`vectorstore` is a *bi-encoder* retrieval: the query
and every chunk are embedded independently, and ranking is cosine similarity
between those independent vectors. That is fast (the corpus is embedded once,
ahead of time) but the model never sees the query and the chunk together, so it
cannot reason about how they actually relate.

A *cross-encoder* does the opposite trade. It takes the (query, chunk) pair as
a single input and scores the pair directly, which is far more accurate, but it
has to run once per candidate at query time, so it cannot scale across a whole
corpus.

The standard pattern, and the one used here, is retrieve-then-rerank: use the
cheap bi-encoder to pull a wide candidate set, then spend the cross-encoder
budget only on those candidates and keep the best few.

The model is a local sentence-transformers cross-encoder, so reranking adds
latency but no API cost.

Measured on this corpus (CPU-only torch, 4 threads, warm model, mean of three
queries, ``max_length=256``, ``batch_size=16``). Baseline vector search is
~22 ms; the figures below are the reranked total:

===========================  ==========  ===========
model                        candidates  added
===========================  ==========  ===========
ms-marco-MiniLM-L-6-v2       25          ~2085 ms
ms-marco-MiniLM-L-2-v2       25          ~704 ms
ms-marco-MiniLM-L-2-v2       15          ~430 ms
===========================  ==========  ===========

The cost is dominated by sequence length and candidate count, both of which
are configurable. On GPU this collapses to tens of milliseconds; the numbers
above are the honest CPU-only story.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, Sequence

from . import config

log = logging.getLogger(__name__)


class _Scorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


_MODEL: Optional[_Scorer] = None
_LOAD_FAILED = False


def _load_model() -> Optional[_Scorer]:
    """Lazily load the cross-encoder.

    Returns ``None`` if sentence-transformers is unavailable or the model
    cannot be loaded, so that reranking degrades to a no-op rather than
    taking down retrieval.
    """
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        log.warning(
            "sentence-transformers not installed; reranking disabled. "
            "Install with: pip install -e '.[embeddings]'"
        )
        _LOAD_FAILED = True
        return None
    try:
        _MODEL = CrossEncoder(config.RERANK_MODEL, max_length=config.RERANK_MAX_LENGTH)
    except Exception:
        log.exception("Failed to load cross-encoder %s; reranking disabled", config.RERANK_MODEL)
        _LOAD_FAILED = True
        return None
    return _MODEL


def is_available() -> bool:
    """True if the cross-encoder can be loaded (used by eval reporting)."""
    return _load_model() is not None


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """Reorder ``hits`` by cross-encoder relevance and keep the best ``top_k``.

    Each hit gains two keys:

    ``rerank_score``
        The raw cross-encoder logit for the (query, chunk) pair. Higher is more
        relevant. These are not probabilities and are not comparable across
        different queries.
    ``vector_rank``
        The chunk's 0-based position *before* reranking, kept so the eval can
        report how far reranking actually moved things.

    If the model is unavailable the hits are returned unchanged (truncated to
    ``top_k``), so callers never need to handle a reranking-specific failure.
    """
    if not hits:
        return []

    model = _load_model()
    if model is None:
        return hits[:top_k]

    pairs = [(query, h.get(text_key) or "") for h in hits]
    try:
        scores = model.predict(pairs, batch_size=config.RERANK_BATCH_SIZE)
    except Exception:
        log.exception("Cross-encoder scoring failed; returning vector order")
        return hits[:top_k]

    for rank, (hit, score) in enumerate(zip(hits, scores)):
        hit["vector_rank"] = rank
        hit["rerank_score"] = float(score)

    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k]


__all__ = ("rerank", "is_available")
