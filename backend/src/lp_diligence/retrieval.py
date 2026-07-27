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
) -> list[Citation]:
    """Vector-search the corpus and return citations.

    If ``embedder``/``store`` are not provided, a default pair is constructed
    using ``config``. Callers running many queries should pass shared instances
    to avoid model-reload cost.

    When reranking is on (``rerank``, defaulting to ``config.RERANK_ENABLED``),
    the vector search is widened to ``config.RERANK_CANDIDATES`` and a local
    cross-encoder re-scores those candidates down to the requested ``k``. See
    :mod:`lp_diligence.reranking` for why that ordering matters.
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

    # With reranking on, pull a wider candidate set so the cross-encoder has
    # something to actually choose from; the bi-encoder's job becomes recall,
    # and precision is the reranker's problem.
    search_k = max(top_k, config.RERANK_CANDIDATES) if use_rerank else top_k

    hits = store.search(
        qvec,
        k=search_k,
        filter_doc_id=doc_id,
        filter_doc_ids=doc_ids,
    )

    if own_store:
        store.close()

    if use_rerank:
        from .reranking import rerank as _rerank

        hits = _rerank(query, hits, top_k=top_k)

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
