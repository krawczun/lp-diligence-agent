"""Hybrid retrieval: fuse dense (vector) and sparse (BM25) rankings.

Dense and sparse retrieval fail in different, complementary ways.

Dense embedding search understands *meaning*. It will match "how did the fund
perform against its benchmark" to a chunk about relative returns even with no
shared vocabulary. What it is bad at is exactness: tickers, fund names, dates,
and specific figures get smeared into the surrounding semantics, so a query
mentioning "2Q17" can rank a chunk about a different quarter just as highly.

BM25 keyword search is the mirror image. It nails the literal token and has no
idea that "returns" and "performance" are related.

Fusing them covers both. The problem is that their scores are not comparable:
cosine distance and bm25() live on different scales with different signs, so
any weighted sum of the raw scores is arbitrary. Reciprocal Rank Fusion sides
steps this entirely by discarding the scores and using only *rank position*,
which is directly comparable across any number of retrievers.

RRF, from Cormack et al. (2009):

    score(d) = sum over retrievers r of  1 / (k + rank_r(d))

``k`` (conventionally 60) damps the influence of the very top ranks so that a
document ranked #1 by one retriever does not automatically beat a document
ranked #2 and #3 by both. Agreement across retrievers is what RRF rewards.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

# Cormack et al.'s default. Larger flattens the curve (rank position matters
# less, so broad agreement dominates); smaller sharpens it (a single #1 hit
# carries more weight).
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[dict[str, Any]]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """Fuse several ranked hit lists into one, best first.

    ``rankings`` is a sequence of ranked lists, each already ordered best-first
    by its own retriever. ``weights`` optionally scales each retriever's
    contribution (defaults to equal weighting).

    Each returned record carries the fusion bookkeeping:

    ``rrf_score``
        The summed reciprocal-rank score. Comparable only within one fusion.
    ``rrf_ranks``
        ``{retriever_index: rank}`` showing where each retriever placed it,
        which is what makes the fusion auditable rather than magic.

    The record itself is taken from the first retriever that returned it, so
    text and metadata come through unchanged.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights has length {len(weights)}, expected {len(rankings)}")

    fused: dict[str, dict[str, Any]] = {}

    for retriever_idx, ranking in enumerate(rankings):
        weight = weights[retriever_idx]
        for rank, hit in enumerate(ranking):
            doc_id = hit.get(id_key)
            if doc_id is None:
                continue
            entry = fused.get(doc_id)
            if entry is None:
                # Copy so fusion bookkeeping never mutates the caller's dicts.
                entry = dict(hit)
                entry["rrf_score"] = 0.0
                entry["rrf_ranks"] = {}
                fused[doc_id] = entry
            entry["rrf_score"] += weight / (k + rank + 1)
            entry["rrf_ranks"][retriever_idx] = rank

    return sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)


__all__ = ("reciprocal_rank_fusion", "DEFAULT_RRF_K")
