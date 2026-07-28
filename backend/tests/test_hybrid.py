"""Tests for reciprocal rank fusion and the FTS query sanitizer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lp_diligence.hybrid import DEFAULT_RRF_K, reciprocal_rank_fusion  # noqa: E402
from lp_diligence.vectorstore import _fts_match_expression  # noqa: E402


def _ranking(*ids: str) -> list[dict]:
    return [{"chunk_id": i, "text": f"text-{i}"} for i in ids]


class TestReciprocalRankFusion:
    def test_agreement_beats_a_single_top_hit(self):
        """The core property: two retrievers agreeing outranks one shouting."""
        dense = _ranking("a", "b", "c")
        sparse = _ranking("x", "b", "c")

        out = reciprocal_rank_fusion([dense, sparse])

        # "b" is #2 in both; "a" and "x" are each #1 in only one list.
        assert out[0]["chunk_id"] == "b"

    def test_union_not_intersection(self):
        """Documents found by only one retriever still survive fusion."""
        out = reciprocal_rank_fusion([_ranking("a"), _ranking("b")])

        assert {h["chunk_id"] for h in out} == {"a", "b"}

    def test_records_each_retrievers_rank(self):
        out = reciprocal_rank_fusion([_ranking("a", "b"), _ranking("b", "a")])

        by_id = {h["chunk_id"]: h for h in out}
        assert by_id["a"]["rrf_ranks"] == {0: 0, 1: 1}
        assert by_id["b"]["rrf_ranks"] == {0: 1, 1: 0}

    def test_score_matches_the_rrf_formula(self):
        out = reciprocal_rank_fusion([_ranking("a")], k=60)

        assert out[0]["rrf_score"] == pytest.approx(1 / (60 + 1))

    def test_weights_shift_the_winner(self):
        dense = _ranking("d")
        sparse = _ranking("s")

        balanced = reciprocal_rank_fusion([dense, sparse], weights=[1.0, 1.0])
        dense_heavy = reciprocal_rank_fusion([dense, sparse], weights=[5.0, 1.0])

        # Tied on rank, so balanced fusion is a coin flip resolved by sort
        # stability; weighting must be decisive.
        assert {h["chunk_id"] for h in balanced} == {"d", "s"}
        assert dense_heavy[0]["chunk_id"] == "d"

    def test_agreement_wins_at_every_k_but_the_margin_narrows(self):
        """Two #2 finishes beat one #1 for any k, since 2/(k+2) > 1/(k+1).

        k does not flip that outcome; it controls how *decisively* agreement
        wins. Small k narrows the gap, large k widens it. Worth pinning down,
        because "tune k to favour top hits" is an intuitive but wrong reading
        of the formula.
        """
        dense = _ranking("a", "b")
        sparse = _ranking("x", "b")

        def margin(k: int) -> float:
            out = reciprocal_rank_fusion([dense, sparse], k=k)
            by_id = {h["chunk_id"]: h["rrf_score"] for h in out}
            assert out[0]["chunk_id"] == "b"
            return by_id["b"] - by_id["a"]

        # Relative margin, since absolute scores shrink as k grows.
        sharp = margin(1) / (1 / (1 + 1))
        flat = margin(1000) / (1 / (1000 + 1))
        assert sharp < flat

    def test_does_not_mutate_input_records(self):
        dense = _ranking("a")
        reciprocal_rank_fusion([dense])

        assert "rrf_score" not in dense[0]

    def test_rejects_weight_length_mismatch(self):
        with pytest.raises(ValueError, match="expected 2"):
            reciprocal_rank_fusion([_ranking("a"), _ranking("b")], weights=[1.0])

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_default_k_is_the_published_constant(self):
        assert DEFAULT_RRF_K == 60


class TestFtsMatchExpression:
    def test_strips_question_syntax(self):
        """A raw question is an FTS5 syntax error; terms must be quoted."""
        expr = _fts_match_expression("What was the fund's Q2 return?")

        assert "?" not in expr
        assert "'" not in expr
        assert '"return"' in expr

    def test_drops_stopwords(self):
        expr = _fts_match_expression("what is the net return")

        assert '"what"' not in expr
        assert '"the"' not in expr
        assert '"net"' in expr

    def test_keeps_period_tokens(self):
        """Period codes are exactly what the sparse half exists to catch."""
        assert '"2q17"' in _fts_match_expression("PSERS 2Q17 returns")

    def test_ors_the_terms(self):
        expr = _fts_match_expression("private equity")

        assert expr == '"private" OR "equity"'

    def test_all_stopwords_yields_empty(self):
        """No lexical signal, so the caller should fall back to dense only."""
        assert _fts_match_expression("what is the") == ""

    def test_empty_query(self):
        assert _fts_match_expression("") == ""

    def test_punctuation_only(self):
        assert _fts_match_expression("??? ...") == ""
