"""Tests for the public demo's cost guards: input bounds, client IP, rate limits."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lp_diligence import api  # noqa: E402


def _request(peer: str, headers: dict[str, str] | None = None):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers or {})


@pytest.fixture(autouse=True)
def _fresh_buckets(monkeypatch):
    api._buckets.clear()
    api._global_bucket.clear()
    yield
    api._buckets.clear()
    api._global_bucket.clear()


class TestInputBounds:
    def test_k_above_cap_is_rejected(self):
        with pytest.raises(ValidationError):
            api.AskRequest(doc_id="d", question="q", k=api.MAX_K + 1)

    def test_k_below_one_is_rejected(self):
        with pytest.raises(ValidationError):
            api.ChecklistRunRequest(doc_id="d", k=0)

    def test_k_at_cap_and_omitted_are_accepted(self):
        assert api.ChecklistRunRequest(doc_id="d", k=api.MAX_K).k == api.MAX_K
        assert api.ChecklistRunRequest(doc_id="d").k is None

    def test_overlong_question_is_rejected(self):
        with pytest.raises(ValidationError):
            api.AskRequest(doc_id="d", question="x" * (api.MAX_QUESTION_CHARS + 1))

    def test_empty_question_is_rejected(self):
        with pytest.raises(ValidationError):
            api.AskRequest(doc_id="d", question="")


class TestClientIp:
    def test_forwarded_headers_trusted_from_local_proxy(self):
        req = _request("127.0.0.1", {"cf-connecting-ip": "203.0.113.7"})
        assert api._client_ip(req) == "203.0.113.7"

    def test_x_forwarded_for_uses_first_hop(self):
        req = _request("::1", {"x-forwarded-for": "198.51.100.4, 10.0.0.1"})
        assert api._client_ip(req) == "198.51.100.4"

    def test_forwarded_headers_ignored_from_direct_caller(self):
        """A caller that isn't the local proxy can't pick its own bucket."""
        req = _request("192.0.2.50", {"cf-connecting-ip": "203.0.113.7"})
        assert api._client_ip(req) == "192.0.2.50"

    def test_local_proxy_without_headers_falls_back_to_peer(self):
        assert api._client_ip(_request("127.0.0.1")) == "127.0.0.1"


class TestRateLimit:
    def test_per_ip_limit_is_per_visitor(self, monkeypatch):
        monkeypatch.setattr(api, "RATE_LIMIT_MAX_RUNS", 2)
        api._check_rate_limit("a")
        api._check_rate_limit("a")
        with pytest.raises(HTTPException) as exc:
            api._check_rate_limit("a")
        assert exc.value.status_code == 429
        # A different visitor is unaffected.
        api._check_rate_limit("b")

    def test_global_daily_ceiling_applies_across_ips(self, monkeypatch):
        monkeypatch.setattr(api, "RATE_LIMIT_GLOBAL_DAILY", 3)
        for ip in ("a", "b", "c"):
            api._check_rate_limit(ip)
        with pytest.raises(HTTPException) as exc:
            api._check_rate_limit("d")
        assert exc.value.status_code == 429

    def test_rejected_request_does_not_consume_global_budget(self, monkeypatch):
        monkeypatch.setattr(api, "RATE_LIMIT_MAX_RUNS", 1)
        monkeypatch.setattr(api, "RATE_LIMIT_GLOBAL_DAILY", 2)
        api._check_rate_limit("a")
        with pytest.raises(HTTPException):
            api._check_rate_limit("a")
        api._check_rate_limit("b")
        assert len(api._global_bucket) == 2


class TestKeywordIndexBackfill:
    """Startup repairs a store that predates hybrid search."""

    @staticmethod
    def _store(path):
        from lp_diligence.vectorstore import VectorStore
        return VectorStore(path, dim=4)

    def _seed_without_fts(self, path):
        store = self._store(path)
        store.add_chunks([{
            "chunk_id": "c1", "doc_id": "d", "entity": "e", "period": "p",
            "section": "s", "chunk_idx": 0, "page_start": 1, "page_end": 1,
            "text": "management fee of 1.25 percent", "embedding": [0.1, 0.2, 0.3, 0.4],
        }])
        store.conn.execute("DELETE FROM chunk_fts")
        store.conn.commit()
        assert store.fts_count() == 0
        store.close()

    def test_empty_index_is_backfilled(self, tmp_path, monkeypatch):
        db = tmp_path / "v.sqlite"
        self._seed_without_fts(db)
        monkeypatch.setattr(api.config, "VECTOR_DB_PATH", db)
        monkeypatch.setattr(api.config, "HYBRID_ENABLED", True)
        api._ensure_keyword_index(4)
        store = self._store(db)
        assert store.fts_count() == 1
        store.close()

    def test_skipped_when_hybrid_disabled(self, tmp_path, monkeypatch):
        db = tmp_path / "v.sqlite"
        self._seed_without_fts(db)
        monkeypatch.setattr(api.config, "VECTOR_DB_PATH", db)
        monkeypatch.setattr(api.config, "HYBRID_ENABLED", False)
        api._ensure_keyword_index(4)
        store = self._store(db)
        assert store.fts_count() == 0
        store.close()
