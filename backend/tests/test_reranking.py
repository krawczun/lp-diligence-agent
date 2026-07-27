"""Tests for cross-encoder reranking.

These avoid loading the real model: the point is the wiring and the failure
behaviour, both of which must hold whether or not sentence-transformers is
installed on the machine running the tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lp_diligence import reranking  # noqa: E402


class _StubModel:
    """Scores pairs by a caller-supplied lookup on the chunk text."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=None):  # noqa: ANN001
        self.calls.extend(pairs)
        return [self.scores.get(text, 0.0) for _, text in pairs]


def _hits(*texts: str) -> list[dict]:
    return [{"chunk_id": f"c{i}", "text": t} for i, t in enumerate(texts)]


@pytest.fixture(autouse=True)
def _reset_model_cache(monkeypatch):
    """Keep module-level model caching from leaking across tests."""
    monkeypatch.setattr(reranking, "_MODEL", None)
    monkeypatch.setattr(reranking, "_LOAD_FAILED", False)


def test_reorders_by_cross_encoder_score(monkeypatch):
    stub = _StubModel({"low": -5.0, "high": 9.0, "mid": 1.0})
    monkeypatch.setattr(reranking, "_load_model", lambda: stub)

    out = reranking.rerank("q", _hits("low", "high", "mid"), top_k=3)

    assert [h["text"] for h in out] == ["high", "mid", "low"]


def test_truncates_to_top_k(monkeypatch):
    stub = _StubModel({"a": 1.0, "b": 2.0, "c": 3.0})
    monkeypatch.setattr(reranking, "_load_model", lambda: stub)

    out = reranking.rerank("q", _hits("a", "b", "c"), top_k=2)

    assert [h["text"] for h in out] == ["c", "b"]


def test_annotates_score_and_prior_rank(monkeypatch):
    stub = _StubModel({"first": 0.0, "second": 8.0})
    monkeypatch.setattr(reranking, "_load_model", lambda: stub)

    out = reranking.rerank("q", _hits("first", "second"), top_k=2)

    promoted = out[0]
    assert promoted["text"] == "second"
    assert promoted["rerank_score"] == 8.0
    # It sat at index 1 in the vector ordering, which is what makes the
    # movement reportable in the eval.
    assert promoted["vector_rank"] == 1


def test_passes_query_with_each_chunk(monkeypatch):
    """A cross-encoder must see the pair; scoring the chunk alone is the bug."""
    stub = _StubModel({})
    monkeypatch.setattr(reranking, "_load_model", lambda: stub)

    reranking.rerank("my query", _hits("x", "y"), top_k=2)

    assert stub.calls == [("my query", "x"), ("my query", "y")]


def test_degrades_to_vector_order_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(reranking, "_load_model", lambda: None)

    out = reranking.rerank("q", _hits("a", "b", "c"), top_k=2)

    assert [h["text"] for h in out] == ["a", "b"]
    assert "rerank_score" not in out[0]


def test_degrades_to_vector_order_when_scoring_raises(monkeypatch):
    class _Boom:
        def predict(self, pairs, batch_size=None):  # noqa: ANN001
            raise RuntimeError("model exploded")

    monkeypatch.setattr(reranking, "_load_model", lambda: _Boom())

    out = reranking.rerank("q", _hits("a", "b"), top_k=2)

    assert [h["text"] for h in out] == ["a", "b"]


def test_empty_hits_short_circuits(monkeypatch):
    def _should_not_load():
        raise AssertionError("model must not load for an empty candidate set")

    monkeypatch.setattr(reranking, "_load_model", _should_not_load)

    assert reranking.rerank("q", [], top_k=8) == []
