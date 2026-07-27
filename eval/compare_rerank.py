"""Retrieval-only A/B of cross-encoder reranking.

``run_eval.py`` measures end-to-end answer quality and needs a live Anthropic
key for the judge. This script isolates the *retrieval* half, so it runs fully
offline against the local corpus and the local cross-encoder.

It reports, per question and in aggregate:

churn
    How many of the top-k chunks reranking replaced.
promoted
    How many surviving chunks came from outside the vector top-k, i.e. results
    the bi-encoder alone would never have shown the model.
deep_rank
    The deepest vector rank that survived into the reranked top-k. A large
    value is the argument for reranking: that chunk was retrievable but not
    rankable without it.
latency
    Warm per-query cost of each path.

Usage::

    python eval/compare_rerank.py
    python eval/compare_rerank.py --candidates 15 --model cross-encoder/ms-marco-MiniLM-L-2-v2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend" / "src"))

GOLDEN_SET = _REPO_ROOT / "eval" / "golden_set" / "golden_set.jsonl"


def _load_questions(limit: int | None) -> list[str]:
    if not GOLDEN_SET.exists():
        raise SystemExit(f"golden set not found at {GOLDEN_SET}")
    questions: list[str] = []
    for line in GOLDEN_SET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        q = row.get("question") or row.get("q")
        if q:
            questions.append(q)
    return questions[:limit] if limit else questions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=None, help="Override RERANK_CANDIDATES.")
    ap.add_argument("--model", type=str, default=None, help="Override the cross-encoder model.")
    ap.add_argument("--max-length", type=int, default=None, help="Override RERANK_MAX_LENGTH.")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    args = ap.parse_args()

    # Config is read at import time, so set overrides before importing.
    if args.model:
        os.environ["LP_DILIGENCE_RERANK_MODEL"] = args.model
    if args.candidates:
        os.environ["LP_DILIGENCE_RERANK_CANDIDATES"] = str(args.candidates)
    if args.max_length:
        os.environ["LP_DILIGENCE_RERANK_MAX_LENGTH"] = str(args.max_length)

    from lp_diligence import config, embeddings, reranking, retrieval, vectorstore

    questions = _load_questions(args.limit)
    if not questions:
        raise SystemExit("no questions loaded")

    embedder = embeddings.Embedder()
    store = vectorstore.VectorStore(config.VECTOR_DB_PATH, dim=embedder.dim)

    if not reranking.is_available():
        store.close()
        raise SystemExit(
            "cross-encoder unavailable. Install with: pip install -e 'backend[embeddings]'"
        )

    # Warm both paths so the timings below exclude one-off model load.
    retrieval.retrieve(questions[0], embedder=embedder, store=store, rerank=False)
    retrieval.retrieve(questions[0], embedder=embedder, store=store, rerank=True)

    print(
        f"corpus={config.VECTOR_DB_PATH.name}  k={config.RETRIEVAL_K}  "
        f"candidates={config.RERANK_CANDIDATES}  "
        f"model={config.RERANK_MODEL.split('/')[-1]}  max_len={config.RERANK_MAX_LENGTH}"
    )
    print(f"questions={len(questions)}\n")
    print(f"{'#':>3}  {'churn':>5}  {'promoted':>8}  {'deep':>4}  {'base ms':>7}  {'rr ms':>6}")
    print("-" * 46)

    churns: list[int] = []
    promotions: list[int] = []
    deepest: list[int] = []
    base_ms: list[float] = []
    rr_ms: list[float] = []

    for i, q in enumerate(questions, 1):
        t = time.perf_counter()
        base = retrieval.retrieve(q, embedder=embedder, store=store, rerank=False)
        b_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        rr = retrieval.retrieve(q, embedder=embedder, store=store, rerank=True)
        r_ms = (time.perf_counter() - t) * 1000

        base_ids = {c.chunk_id for c in base}
        churn = sum(1 for c in rr if c.chunk_id not in base_ids)
        promoted = sum(1 for c in rr if c.vector_rank is not None and c.vector_rank >= len(base))
        deep = max((c.vector_rank or 0) for c in rr) if rr else 0

        churns.append(churn)
        promotions.append(promoted)
        deepest.append(deep)
        base_ms.append(b_ms)
        rr_ms.append(r_ms)

        print(f"{i:>3}  {churn:>5}  {promoted:>8}  {deep:>4}  {b_ms:>7.0f}  {r_ms:>6.0f}")

    k = config.RETRIEVAL_K
    print("-" * 46)
    print(
        f"\nmean churn        {stats.mean(churns):.1f} of {k} chunks replaced "
        f"({stats.mean(churns) / k:.0%})"
    )
    print(f"mean promoted     {stats.mean(promotions):.1f} from outside the vector top-{k}")
    print(f"deepest survivor  rank {max(deepest)} (of {config.RERANK_CANDIDATES} candidates)")
    print(f"\nlatency  baseline {stats.mean(base_ms):.0f} ms -> reranked {stats.mean(rr_ms):.0f} ms")
    print(f"         added    {stats.mean(rr_ms) - stats.mean(base_ms):.0f} ms per query")

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
