"""Unit tests for the LongMemEval benchmark-owned context repair layer."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from eval.longmemeval.adapter import LongMemEvalQuestion
from eval.longmemeval.context import (
    build_longmemeval_context,
    build_query_variants,
    format_context_bundle,
    make_retrying_longmemeval_extractor,
)
from src.substrate.claims import insert_claim, insert_turn, new_turn_id, now_ns
from src.substrate.retrieval import EmbeddingClient
from src.substrate.schema import Claim, ClaimStatus, Speaker, Turn, open_database


class _StubEmbedder(EmbeddingClient):
    def __init__(self) -> None:
        super().__init__(modal_url="__unused__", model_version="stub@test")
        self._modal_url = None
        self._local_model = object()  # type: ignore[assignment]

    def embed(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        out: list[list[float]] = []
        for text in texts:
            buckets = [0.0] * 8
            for idx, ch in enumerate(text.lower().encode("utf-8")):
                buckets[idx % 8] += float((ch % 17) + 1)
            norm = sum(v * v for v in buckets) ** 0.5 or 1.0
            out.append([v / norm for v in buckets])
        return out


@pytest.fixture(autouse=True)
def _longmemeval_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTIVE_PACK", "personal_assistant")
    from src.substrate.predicate_packs import active_pack

    active_pack.cache_clear()
    yield
    active_pack.cache_clear()


def _seed_turn(conn: sqlite3.Connection, session_id: str, text: str) -> str:
    tid = new_turn_id()
    insert_turn(
        conn,
        Turn(
            turn_id=tid,
            session_id=session_id,
            speaker=Speaker.PATIENT,
            text=text,
            ts=now_ns(),
        ),
    )
    return tid


def _seed_claim(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    subject: str,
    predicate: str,
    value: str,
) -> Claim:
    turn_id = _seed_turn(conn, session_id, f"{subject} {predicate} {value}")
    return insert_claim(
        conn,
        session_id=session_id,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=0.95,
        source_turn_id=turn_id,
    )


def _question() -> LongMemEvalQuestion:
    return LongMemEvalQuestion(
        question_id="q_001",
        question="When did the user move to Boston and what is the current job?",
        answer="2021",
        question_type="knowledge_update",
        haystack_sessions=[],
    )


class TestQueryExpansion:
    def test_query_variants_expand_deterministically(self) -> None:
        variants = build_query_variants("When did the user move to Boston?")
        assert len(variants) >= 3
        assert len(set(variants)) == len(variants)
        assert variants[0] == "When did the user move to Boston?"
        assert any("temporal memory facts about" in v for v in variants)


class TestContextBundle:
    def test_rerank_prefers_multihit_claims_over_single_high_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = open_database(":memory:")
        q = _question()
        c1 = _seed_claim(conn, q.question_id, subject="user", predicate="user_fact", value="Moved in 2020")
        c2 = _seed_claim(conn, q.question_id, subject="user", predicate="user_fact", value="Moved in 2021")

        monkeypatch.setattr(
            "eval.longmemeval.context.backfill_embeddings",
            lambda *_a, **_k: 0,
        )
        monkeypatch.setattr("eval.longmemeval.context.EmbeddingClient", _StubEmbedder)

        def _retrieval(_conn: Any, *, session_id: str, query: str, **_kwargs: Any):
            if query.startswith("When did"):
                return [(c1, 0.80), (c2, 0.80)]
            if "prior conversation facts" in query:
                return [(c2, 0.80)]
            return [(c2, 0.80)]

        monkeypatch.setattr("eval.longmemeval.context.retrieve_hybrid", _retrieval)

        bundle = build_longmemeval_context(q, conn, top_k=2, embedding_client=_StubEmbedder())
        assert bundle.evidence[0].claim_id == c2.claim_id
        assert bundle.evidence[0].query_hits >= bundle.evidence[1].query_hits
        conn.close()

    def test_superseded_claim_is_filtered_from_bundle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = open_database(":memory:")
        q = _question()
        c1 = _seed_claim(conn, q.question_id, subject="user", predicate="user_fact", value="Lives in Denver")
        c2 = _seed_claim(conn, q.question_id, subject="user", predicate="user_fact", value="Lives in Boston")
        from src.substrate.claims import set_claim_status

        set_claim_status(conn, c1.claim_id, ClaimStatus.SUPERSEDED)
        monkeypatch.setattr("eval.longmemeval.context.EmbeddingClient", _StubEmbedder)

        bundle = build_longmemeval_context(q, conn, top_k=4, embedding_client=_StubEmbedder())
        claim_ids = [e.claim_id for e in bundle.evidence]
        assert c1.claim_id not in claim_ids
        assert c2.claim_id in claim_ids
        conn.close()

    def test_stable_formatting_is_deterministic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = open_database(":memory:")
        q = _question()
        _seed_claim(conn, q.question_id, subject="user", predicate="user_fact", value="Moved in 2021")
        monkeypatch.setattr("eval.longmemeval.context.backfill_embeddings", lambda *_a, **_k: 0)
        monkeypatch.setattr("eval.longmemeval.context.EmbeddingClient", _StubEmbedder)
        monkeypatch.setattr(
            "eval.longmemeval.context.retrieve_hybrid",
            lambda _conn, **_kw: [],
        )

        bundle = build_longmemeval_context(q, conn, top_k=2, embedding_client=_StubEmbedder())
        rendered_a = bundle.format()
        rendered_b = format_context_bundle(bundle)
        assert rendered_a == rendered_b
        assert rendered_a.startswith("LONGMEMEVAL EVIDENCE BUNDLE")
        conn.close()


class TestExtractorRetry:
    def test_retry_and_backoff_on_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACTIVE_PACK", "personal_assistant")

        class _TransientError(Exception):
            def __init__(self) -> None:
                super().__init__("rate limited")
                self.status_code = 429

        class _Resp:
            choices = [SimpleNamespace(message=SimpleNamespace(content=json.dumps([{
                "subject": "user",
                "predicate": "user_fact",
                "value": "moved to Boston",
                "confidence": 0.9,
            }])))]

        calls: list[str] = []

        class _Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**_kwargs: Any) -> Any:
                        calls.append("call")
                        if len(calls) < 3:
                            raise _TransientError()
                        return _Resp()

        monkeypatch.setattr(
            "eval._openai_client.make_openai_client",
            lambda _purpose, _env=None: (_Client(), "stub-model"),
        )

        delays: list[float] = []
        extractor = make_retrying_longmemeval_extractor(
            env={"OPENAI_API_KEY": "sk-test"},
            max_attempts=4,
            base_delay_s=0.5,
            sleep_fn=delays.append,
        )

        turn = Turn(
            turn_id="tu_1",
            session_id="s1",
            speaker=Speaker.SYSTEM,
            text="[system] I moved to Boston in 2021.",
            ts=now_ns(),
        )
        claims = extractor(turn)
        assert len(calls) == 3
        assert delays == [0.5, 1.0]
        assert claims and claims[0].predicate == "user_fact"
