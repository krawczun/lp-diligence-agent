"""Retrieval helpers: query → top-k chunks formatted for the LLM prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import config
from .embeddings import Embedder
from .vectorstore import VectorStore


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    entity: str
    period: str
    section: str
    page_start: int | None
    page_end: int | None
    excerpt: str
    score: float
    # Set only when reranking ran. ``rerank_score`` is the cross-encoder logit
    # for this (query, chunk) pair; ``vector_rank`` is where the chunk sat in
    # the bi-encoder ordering beforehand, which is what makes the movement
    # measurable.
    rerank_score: float | None = None
    vector_rank: int | None = None

    def label(self) -> str:
        """Short citation label like ``[PSERS 2Q17 Performance p.12]``."""
        page_str = ""
        if self.page_start is not None:
            if self.page_end and self.page_end != self.page_start:
                page_str = f" p.{self.page_start}-{self.page_end}"
            else:
                page_str = f" p.{self.page_start}"
        return f"[{self.entity} {self.period} {self.section}{page_str}]"


def retrieve(
    query: str,
    *,
    doc_id: Optional[str] = None,
    doc_ids: Optional[list[str]] = None,
    k: Optional[int] = None,
    embedder: Optional[Embedder] = None,
    store: Optional[VectorStore] = None,
    rerank: Optional[bool] = None,
    hybrid: Optional[bool] = None,
) -> list[Citation]:
    """Vector-search the corpus and return citations.

    If ``embedder``/``store`` are not provided, a default pair is constructed
    using ``config``. Callers running many queries should pass shared instances
    to avoid model-reload cost.

    Two optional stages sit on top of vector search, each defaulting to its
    ``config`` flag and each independently switchable for A/B runs:

    ``hybrid``
        Also run BM25 keyword search and fuse the two rankings with reciprocal
        rank fusion (:mod:`lp_diligence.hybrid`). Covers the exact-token cases
        embeddings smear: tickers, periods, figures.
    ``rerank``
        Re-score the candidate set with a cross-encoder that sees each
        (query, chunk) pair together (:mod:`lp_diligence.reranking`).

    With both on the order is retrieve -> fuse -> rerank, so the most accurate
    signal gets the final say.
    """
    embedder = embedder or Embedder()
    own_store = False
    if store is None:
        store = VectorStore(config.VECTOR_DB_PATH, dim=embedder.dim)
        own_store = True

    qvec = embedder.embed_query(query)
    if qvec is None:
        if own_store:
            store.close()
        return []

    top_k = k or config.RETRIEVAL_K
    use_rerank = config.RERANK_ENABLED if rerank is None else rerank
    use_hybrid = config.HYBRID_ENABLED if hybrid is None else hybrid

    # Both stages want a wider candidate pool than the final k: hybrid needs
    # room for the two rankings to disagree, reranking needs candidates to
    # choose between. Take the widest requirement that applies.
    search_k = top_k
    if use_hybrid:
        search_k = max(search_k, config.HYBRID_CANDIDATES)
    if use_rerank:
        search_k = max(search_k, config.RERANK_CANDIDATES)

    hits = store.search(
        qvec,
        k=search_k,
        filter_doc_id=doc_id,
        filter_doc_ids=doc_ids,
    )

    if use_hybrid:
        from .hybrid import reciprocal_rank_fusion

        keyword_hits = store.search_keyword(
            query,
            k=search_k,
            filter_doc_id=doc_id,
            filter_doc_ids=doc_ids,
        )
        # With no lexical matches, fusion would just re-emit the dense ranking,
        # so skip it and keep the vector order untouched.
        if keyword_hits:
            hits = reciprocal_rank_fusion(
                [hits, keyword_hits],
                k=config.HYBRID_RRF_K,
                weights=[config.HYBRID_DENSE_WEIGHT, config.HYBRID_SPARSE_WEIGHT],
            )

    if own_store:
        store.close()

    if use_rerank:
        # Rerank runs last: it is the most accurate signal, so it gets the
        # final say over whatever candidate set the earlier stages assembled.
        from .reranking import rerank as _rerank

        hits = _rerank(query, hits, top_k=top_k)
    else:
        hits = hits[:top_k]

    citations: list[Citation] = []
    for h in hits:
        citations.append(
            Citation(
                chunk_id=h["chunk_id"],
                doc_id=h["doc_id"],
                entity=h["entity"],
                period=h["period"],
                section=h["section"],
                page_start=h.get("page_start"),
                page_end=h.get("page_end"),
                excerpt=h["text"],
                score=h["score"],
                rerank_score=h.get("rerank_score"),
                vector_rank=h.get("vector_rank"),
            )
        )
    return citations


def format_context(citations: list[Citation], *, max_chars: int = 12000) -> str:
    """Render citations into a prompt-ready context block.

    Each chunk is prefixed with its citation label so the LLM can echo back
    the same label when synthesizing the answer.
    """
    blocks: list[str] = []
    total = 0
    for c in citations:
        block = f"{c.label()}\n{c.excerpt.strip()}\n"
        if total + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        total += len(block)
    return "\n---\n".join(blocks)


__all__ = ("Citation", "retrieve", "format_context")
