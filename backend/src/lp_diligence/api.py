"""FastAPI server backing the Next.js demo UI.

Routes:
  GET  /api/documents              -> list of available documents
  GET  /api/checklist/items        -> the 9-item checklist
  POST /api/checklist/run          -> run the checklist against one document
  POST /api/ask                    -> RAG Q&A over one document

Rate limited at the route level via a simple per-IP token bucket so the public
demo doesn't get drained by a script, plus a global daily ceiling as a cost
backstop. Tune ``RATE_LIMIT_*`` in env if needed.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config
from .checklist import CHECKLIST_ITEMS, run_checklist, answer_item, _build_client
from .documents import list_documents
from .embeddings import Embedder
from .retrieval import format_context, retrieve
from .vectorstore import VectorStore

logger = logging.getLogger("lp_diligence.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


RATE_LIMIT_WINDOW_S = int(os.environ.get("RATE_LIMIT_WINDOW_S", "3600"))
RATE_LIMIT_MAX_RUNS = int(os.environ.get("RATE_LIMIT_MAX_RUNS", "10"))
# Ceiling across ALL visitors per rolling day. The per-IP bucket can be dodged
# by rotating addresses; this cannot, so it bounds the worst-case API bill.
RATE_LIMIT_GLOBAL_DAILY = int(os.environ.get("RATE_LIMIT_GLOBAL_DAILY", "200"))

# Upper bounds on client-supplied inputs. k sets how many ~500-token chunks go
# into every prompt, so an unbounded k is an unbounded input-token bill.
MAX_K = 20
MAX_QUESTION_CHARS = 500

_buckets: dict[str, deque[float]] = defaultdict(deque)
_global_bucket: deque[float] = deque()

# Requests reach this server through the Next.js rewrite (and Cloudflare in
# front of that), so the socket peer is the local proxy for every visitor.
# Forwarded headers are only trusted when the peer is that local proxy;
# a direct caller could otherwise pick any IP it likes.
_TRUSTED_PROXIES = {"127.0.0.1", "::1", "localhost"}


def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    if peer not in _TRUSTED_PROXIES:
        return peer
    cf_ip = request.headers.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or peer


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    while _global_bucket and now - _global_bucket[0] > 86400:
        _global_bucket.popleft()
    if len(_global_bucket) >= RATE_LIMIT_GLOBAL_DAILY:
        raise HTTPException(
            status_code=429,
            detail="The demo has reached its daily usage limit. Please try again tomorrow.",
        )
    bucket = _buckets[ip]
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_S:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX_RUNS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {RATE_LIMIT_MAX_RUNS} runs per {RATE_LIMIT_WINDOW_S // 60} minutes. Try again later.",
        )
    bucket.append(now)
    _global_bucket.append(now)
    # Drop idle IPs so the dict doesn't grow for the life of the process.
    if len(_buckets) > 10_000:
        for key in [k for k, b in _buckets.items() if not b or now - b[-1] > RATE_LIMIT_WINDOW_S]:
            del _buckets[key]


# Shared resources, populated at startup so first-request latency doesn't
# include the ~3-5 second sentence-transformers model load. Without this, the
# first checklist call can exceed proxy timeouts in dev.
_embedder: Optional[Embedder] = None
_anthropic_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _anthropic_client
    logger.info("Pre-warming embedder...")
    t0 = time.time()
    _embedder = Embedder()
    _embedder.embed_query("warmup")
    logger.info("Embedder ready in %.1fs", time.time() - t0)
    _anthropic_client = _build_client()
    logger.info("Anthropic client ready")
    yield


def get_embedder() -> Embedder:
    if _embedder is None:
        raise RuntimeError("Embedder not initialized")
    return _embedder


def get_client():
    if _anthropic_client is None:
        raise RuntimeError("Anthropic client not initialized")
    return _anthropic_client


app = FastAPI(title="LP Diligence Agent", lifespan=lifespan)

# The browser always calls same-origin /api/* through the Next.js rewrite, so no
# cross-origin access is needed. Set CORS_ALLOW_ORIGINS only if a separately
# hosted frontend has to call this server directly.
_cors_origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Full detail goes to the server log only; the client gets a generic message
    # so file paths and upstream API errors aren't published to visitors.
    logger.error("Unhandled exception on %s %s:\n%s", request.method, request.url.path, traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})


class ChecklistRunRequest(BaseModel):
    doc_id: str = Field(max_length=200)
    k: Optional[int] = Field(default=None, ge=1, le=MAX_K)


class AskRequest(BaseModel):
    doc_id: str = Field(max_length=200)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    k: Optional[int] = Field(default=None, ge=1, le=MAX_K)


@app.get("/api/documents")
def get_documents() -> dict:
    return {"documents": list_documents()}


@app.get("/api/checklist/items")
def get_checklist_items() -> dict:
    return {"items": [{"id": i, "question": q} for i, q in CHECKLIST_ITEMS]}


@app.post("/api/checklist/run")
def post_checklist_run(req: ChecklistRunRequest, request: Request) -> dict:
    _check_rate_limit(_client_ip(request))
    embedder = get_embedder()
    client = get_client()
    store = VectorStore(config.VECTOR_DB_PATH, dim=embedder.dim)
    answers = []
    try:
        for item_id, question in CHECKLIST_ITEMS:
            answers.append(
                answer_item(
                    item_id,
                    question,
                    doc_id=req.doc_id,
                    client=client,
                    embedder=embedder,
                    store=store,
                    k=req.k or config.RETRIEVAL_K,
                )
            )
    finally:
        store.close()
    return {"doc_id": req.doc_id, "answers": [asdict(a) for a in answers]}


@app.post("/api/ask")
def post_ask(req: AskRequest, request: Request) -> dict:
    _check_rate_limit(_client_ip(request))
    embedder = get_embedder()
    client = get_client()
    store = VectorStore(config.VECTOR_DB_PATH, dim=embedder.dim)
    try:
        ans = answer_item(
            "adhoc",
            req.question,
            doc_id=req.doc_id,
            client=client,
            embedder=embedder,
            store=store,
            k=req.k or config.RETRIEVAL_K,
        )
    finally:
        store.close()
    return {"doc_id": req.doc_id, "answer": asdict(ans)}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}
