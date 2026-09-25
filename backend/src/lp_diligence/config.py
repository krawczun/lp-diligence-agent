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
# Sonnet 5 since 2026-09-25: the judge should be at least as capable as the answers it grades.
JUDGE_MODEL: str = os.environ.get("LP_DILIGENCE_JUDGE_MODEL", "claude-sonnet-5")

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
# Over the 20-question golden set (run 2026-07-28, re-graded 2026-09-25 with a
# judge that sees the full context), reranking bought no measurable quality:
# faithfulness, recall and precision all land within 0.03 of hybrid and of
# plain vector search. It costs ~2 s per query against ~6 ms for hybrid, so
# on this corpus it doesn't earn its latency. (An earlier note here claimed
# stacking it on hybrid cut recall from 0.73 to 0.61; that was an artifact of
# a judge that saw only part of the context. See README, "Correction".)
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
# ON by default. Over the 20-question golden set (re-graded 2026-09-25), hybrid
# is at least as good as every other configuration on judged quality (all
# within 0.03) and leads slightly on refusal correctness (0.80 vs 0.75) and
# keyword match (0.85 vs 0.75), at ~6 ms per query. Those leads are one or two
# questions, so the case for it is "no worse, nearly free", not "clearly
# better". Set LP_DILIGENCE_HYBRID=0 to disable.
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
