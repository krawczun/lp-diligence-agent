"""Env-driven constants for the LP Diligence Agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve repo root: this file lives at <repo>/backend/src/lp_diligence/config.py,
# so parents[4] = <repo>. Sanity-check by looking for a known sibling and walking
# up if needed so behavior survives editable installs from odd cwds.
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "corpus").exists():
            return parent
    return here.parents[4] if len(here.parents) > 4 else here.parents[-1]


_REPO_ROOT = _find_repo_root()
load_dotenv(_REPO_ROOT / ".env", override=False)

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL: str = os.environ.get("LP_DILIGENCE_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL: str = os.environ.get("LP_DILIGENCE_JUDGE_MODEL", "claude-haiku-4-5-20251001")

EMBEDDING_PROVIDER: str = os.environ.get("LP_DILIGENCE_EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

CACHE_DIR: Path = Path(os.environ.get("LP_DILIGENCE_CACHE_DIR") or (_REPO_ROOT / "data" / "cache"))
CORPUS_DIR: Path = _REPO_ROOT / "data" / "corpus"
VECTOR_DB_PATH: Path = CACHE_DIR / "vectors.sqlite"

REQUEST_TIMEOUT: int = 60

# Chunking
TARGET_TOKENS: int = 500
OVERLAP_TOKENS: int = 60
RETRIEVAL_K: int = 8

# Reranking
# Retrieve a wide candidate set with the cheap bi-encoder, then let a local
# cross-encoder re-score those candidates and keep the best RETRIEVAL_K.
#
# OFF by default, and that is a measured decision rather than an oversight.
# Over the 20-question golden set (2026-07-28) reranking did beat the plain
# vector baseline, but it lost to hybrid search on every quality metric while
# costing ~2 s per query instead of ~6 ms. Stacking it on top of hybrid was
# worse than hybrid alone (context recall 0.61 vs 0.73): RRF ranks by
# cross-retriever agreement, and re-scoring the fused set on pairwise
# relevance discards exactly that signal.
#
# Kept in the codebase because it is the right tool for a corpus where
# recall is good and ordering is bad, and because the GPU path changes the
# latency calculus entirely. LP_DILIGENCE_RERANK=1 turns it on.
RERANK_ENABLED: bool = os.environ.get("LP_DILIGENCE_RERANK", "").strip().lower() in {"1", "true", "yes", "on"}
RERANK_MODEL: str = os.environ.get(
    "LP_DILIGENCE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_CANDIDATES: int = int(os.environ.get("LP_DILIGENCE_RERANK_CANDIDATES", "25"))
# Cross-encoder cost is roughly linear in candidates and in sequence length.
# On CPU the 512-token window is the dominant term, and chunks here target
# ~500 tokens, so truncating to 256 buys most of the speedup at little cost
# in ranking quality: the signal that matters is usually early in the chunk.
RERANK_MAX_LENGTH: int = int(os.environ.get("LP_DILIGENCE_RERANK_MAX_LENGTH", "256"))
RERANK_BATCH_SIZE: int = int(os.environ.get("LP_DILIGENCE_RERANK_BATCH_SIZE", "16"))

# Hybrid search
# Fuse dense vector search with BM25 keyword search over the FTS5 index.
#
# ON by default. Measured 2026-07-28 over the 20-question golden set, hybrid
# was the best configuration on every quality metric (faithfulness 0.91 vs
# 0.84 baseline, context precision 0.71 vs 0.59, refusal correctness 0.80 vs
# 0.75) at a cost of ~6 ms per query. Set LP_DILIGENCE_HYBRID=0 to disable.
HYBRID_ENABLED: bool = os.environ.get("LP_DILIGENCE_HYBRID", "1").strip().lower() in {"1", "true", "yes", "on"}
# Candidates pulled from each retriever before fusion.
HYBRID_CANDIDATES: int = int(os.environ.get("LP_DILIGENCE_HYBRID_CANDIDATES", "25"))
# RRF damping constant; see lp_diligence.hybrid for what it controls.
HYBRID_RRF_K: int = int(os.environ.get("LP_DILIGENCE_HYBRID_RRF_K", "60"))
# Relative weight of the two retrievers at fusion time (dense, sparse).
HYBRID_DENSE_WEIGHT: float = float(os.environ.get("LP_DILIGENCE_HYBRID_DENSE_WEIGHT", "1.0"))
HYBRID_SPARSE_WEIGHT: float = float(os.environ.get("LP_DILIGENCE_HYBRID_SPARSE_WEIGHT", "1.0"))


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
