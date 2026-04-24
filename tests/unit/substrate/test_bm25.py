"""Tests for BM25 keyword retrieval signal (Layer 3)."""
from __future__ import annotations

import pytest

from src.substrate.retrieval import _bm25_scores, _tokenize_for_bm25
from src.substrate.schema import Claim, ClaimStatus


def _make_claim(subject: str, predicate: str, value: str) -> Claim:
    """Minimal claim for BM25 testing."""
    return Claim(
        claim_id=f"cl_{subject}_{value[:8]}",
        session_id="test",
        subject=subject,
        predicate=predicate,
        value=value,
        value_normalised=None,
        confidence=0.9,
        source_turn_id="tu_test",
        status=ClaimStatus.ACTIVE,
        created_ts=1000,
    )


class TestTokenizer:
    def test_lowercases(self) -> None:
        assert _tokenize_for_bm25("Hello World") == ["hello", "world"]

    def test_splits_on_punctuation(self) -> None:
        assert _tokenize_for_bm25("user/onset=3pm") == ["user", "onset", "3pm"]

    def test_preserves_numbers(self) -> None:
        tokens = _tokenize_for_bm25("Dr. Patel at 3pm on 2024-01-15")
        assert "3pm" in tokens
        assert "2024" in tokens
        assert "01" in tokens
        assert "15" in tokens
        assert "patel" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize_for_bm25("") == []


class TestBM25Scores:
    def test_exact_match_higher_than_unrelated(self) -> None:
        claims = [
            _make_claim("user", "onset", "Dr. Patel mentioned chest pain"),
            _make_claim("user", "onset", "a doctor mentioned something"),
        ]
        query_tokens = _tokenize_for_bm25("Dr. Patel")
        scores = _bm25_scores(query_tokens, claims)
        assert scores[0] > scores[1]

    def test_number_exact_match(self) -> None:
        claims = [
            _make_claim("user", "onset", "appointment at 3pm"),
            _make_claim("user", "onset", "went to the doctor in afternoon"),
        ]
        query_tokens = _tokenize_for_bm25("3pm")
        scores = _bm25_scores(query_tokens, claims)
        assert scores[0] > scores[1]

    def test_all_scores_nonnegative(self) -> None:
        claims = [
            _make_claim("user", "onset", "pain started yesterday"),
            _make_claim("user", "severity", "moderate to severe"),
        ]
        query_tokens = _tokenize_for_bm25("when did the pain start")
        scores = _bm25_scores(query_tokens, claims)
        assert all(s >= 0.0 for s in scores)

    def test_no_overlap_returns_zero(self) -> None:
        claims = [
            _make_claim("user", "onset", "alpha beta gamma"),
        ]
        query_tokens = _tokenize_for_bm25("xyz completely unrelated")
        scores = _bm25_scores(query_tokens, claims)
        assert scores[0] == 0.0

    def test_empty_claims_returns_empty(self) -> None:
        scores = _bm25_scores(["hello"], [])
        assert scores == []

    def test_empty_query_returns_zeros(self) -> None:
        claims = [_make_claim("user", "onset", "some value")]
        scores = _bm25_scores([], claims)
        assert scores == [0.0]

    def test_deterministic(self) -> None:
        claims = [
            _make_claim("user", "onset", "pain started 3 days ago"),
            _make_claim("user", "severity", "moderate"),
        ]
        query_tokens = _tokenize_for_bm25("pain onset 3 days")
        s1 = _bm25_scores(query_tokens, claims)
        s2 = _bm25_scores(query_tokens, claims)
        assert s1 == s2

    def test_multi_token_query_accumulates(self) -> None:
        claims = [
            _make_claim("user", "onset", "pain started 3 days ago"),
            _make_claim("user", "severity", "moderate chest discomfort"),
        ]
        single = _bm25_scores(["pain"], claims)
        multi = _bm25_scores(["pain", "3", "days"], claims)
        assert multi[0] > single[0]
