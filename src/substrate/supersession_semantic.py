"""Supersession Pass 2 — semantic identity via embeddings.

Per `Eng_doc.md` §5.3 Pass 2 + `docs/gt_v2_study_notes.md` §2.4 and
`docs/research_brief.md` §2.5 / §5.

Identity embedding is `subject + predicate + context`, **never** `value`
(gt_v2_study_notes §3.7 — easy-to-break invariant). If `value` is in
the embedding, "onset: 3 days" and "onset: 4 days" embed differently
and supersession can never match.

Cosine threshold: **0.88** (research_brief §2.5 — lower than GT v2's
0.92 for better recall on clinical paraphrases).

Embedder interface: `Embedder.embed(texts: list[str]) -> list[list[float]]`.
Two implementations:

- `NullEmbedder` — returns the same constant vector for every input.
  Deterministic; cosine is always 1.0. Used for tests and the CI
  default so we do not require a heavy model install. **Pass 2 will
  match everything with identical (subject, predicate, source_turn)
  context but does not distinguish semantic paraphrases** — that's
  the intended "no-op" behaviour.
- `E5Embedder` — `intfloat/e5-small-v2` via `sentence_transformers`.
  Lazy-imported so the dependency is optional at install time.

Upstream decision (wt-extraction owns the semantic integration per
`CLAUDE.md` §5.2); this file provides the substrate-side interface so
other worktrees can swap in a real embedder without touching schema.
"""
from __future__ import annotations

import math
import sqlite3
from typing import Any, Protocol, cast, runtime_checkable

import structlog

from src.substrate.claims import set_claim_status
from src.substrate.schema import Claim, ClaimStatus, EdgeType, SupersessionEdge
from src.substrate.supersession import write_supersession_edge

log = structlog.get_logger(__name__)


DEFAULT_COSINE_THRESHOLD = 0.88  # research_brief §2.5


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder interface. Implementations must be deterministic."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list-of-vectors; one vector per input text, same order."""
        ...


class NullEmbedder:
    """Constant-vector embedder. Deterministic; cosine is always 1.0.

    Used as the default so unit tests and CI don't need a heavy model.
    Real embedding is plugged in by `wt-extraction` (`CLAUDE.md` §5.2).
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._vec = [1.0 / math.sqrt(dim)] * dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


class E5Embedder:
    """`intfloat/e5-small-v2` via `sentence_transformers` (MIT).

    Attribution: see `MODEL_ATTRIBUTIONS.md` — e5-small-v2 is already
    declared, so the licensing CI gate stays green.

    Lazy import: `sentence_transformers` is heavy (pulls torch); we only
    import when the class is actually instantiated. If the import fails
    we log and propagate the `ImportError` (rules.md §8 — no silent
    failures).
    """

    MODEL_ID = "intfloat/e5-small-v2"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            log.error("substrate.e5_import_failed", error=str(exc))
            raise

        # Per the e5 paper, passage text should be prefixed with "passage: "
        # for the identity-match use case (asymmetric, close to symmetric here).
        self._model = SentenceTransformer(self.MODEL_ID)
        self._prefix = "passage: "

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._prefix + t for t in texts]
        # `encode` is overloaded such that pyright can't narrow return type.
        # Cast to a generic iterable-of-iterables for pyright strict; the
        # runtime shape is a numpy ndarray of shape (batch, dim).
        raw = cast(Any, self._model).encode(prefixed, normalize_embeddings=True)
        out: list[list[float]] = []
        for v in raw:
            out.append([float(x) for x in v])
        return out


# -------------------------------------------------------------- cosine sim ---


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# ------------------------------------------------- SemanticSupersession API ---


def identity_text(claim: Claim, context: str) -> str:
    """Build the identity embedding input — `subject predicate [context]`.

    `context` is a short surrounding-turn snippet (first ~80 chars of the
    source-turn text is a reasonable default). **Value is excluded.**
    """
    ctx = context.strip().replace("\n", " ")
    if len(ctx) > 160:
        ctx = ctx[:160]
    return f"{claim.subject} {claim.predicate} {ctx}"


class SemanticSupersession:
    """Pass-2 semantic supersession with a pluggable embedder.

    Usage:

        sem = SemanticSupersession(NullEmbedder())
        edge = sem.detect(conn, new_claim)

    With `NullEmbedder`, cosine is always 1.0, so Pass 2 will fire on any
    prior active claim that shares the same subject+predicate but has
    different value. In practice Pass 1 catches those first, so Pass 2
    with `NullEmbedder` is effectively a no-op on well-formed flows.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = DEFAULT_COSINE_THRESHOLD,
    ) -> None:
        self._embedder: Embedder = embedder or NullEmbedder()
        self._threshold = threshold

    def detect(
        self,
        conn: sqlite3.Connection,
        new_claim: Claim,
        *,
        new_turn_text: str = "",
    ) -> SupersessionEdge | None:
        """Find the nearest prior active claim in the same predicate family.

        Returns `(old_claim_id, new_claim_id, SEMANTIC_REPLACE, cosine)`
        if cosine ≥ threshold, and marks the old claim superseded; else
        `None`.

        Guard (Eng_doc.md §5.3): same-turn candidates are excluded.
        """
        # Candidate set: active claims in same session+predicate, different id+turn.
        rows = conn.execute(
            "SELECT * FROM claims WHERE session_id = ?"
            " AND predicate = ?"
            " AND status IN ('active','confirmed')"
            " AND claim_id != ?"
            " AND source_turn_id != ?",
            (
                new_claim.session_id,
                new_claim.predicate,
                new_claim.claim_id,
                new_claim.source_turn_id,
            ),
        ).fetchall()
        if not rows:
            return None

        from src.substrate.claims import row_to_claim  # local import avoids cycles

        candidates = [row_to_claim(r) for r in rows]

        # Build identity texts for new + each candidate.
        new_text = identity_text(new_claim, new_turn_text)
        cand_texts: list[str] = []
        for c in candidates:
            old_ctx = self._lookup_turn_text(conn, c.source_turn_id)
            cand_texts.append(identity_text(c, old_ctx))

        vecs = self._embedder.embed([new_text, *cand_texts])
        if len(vecs) != 1 + len(candidates):
            # Protocol violation — log and bail (no silent fallback).
            log.error(
                "substrate.semantic_embed_length_mismatch",
                expected=1 + len(candidates),
                actual=len(vecs),
            )
            return None
        new_vec = vecs[0]
        cand_vecs = vecs[1:]

        best_idx = -1
        best_score = -1.0
        for i, v in enumerate(cand_vecs):
            score = cosine(new_vec, v)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0 or best_score < self._threshold:
            log.debug(
                "substrate.semantic_no_match",
                session_id=new_claim.session_id,
                claim_id=new_claim.claim_id,
                best_score=best_score,
                threshold=self._threshold,
            )
            return None

        old_claim = candidates[best_idx]
        # If Pass 1 already superseded this candidate in the same transaction
        # (edge exists + status changed), skip — idempotent.
        if old_claim.status is ClaimStatus.SUPERSEDED:
            return None

        edge = write_supersession_edge(
            conn,
            old_claim_id=old_claim.claim_id,
            new_claim_id=new_claim.claim_id,
            edge_type=EdgeType.SEMANTIC_REPLACE,
            identity_score=best_score,
        )
        set_claim_status(conn, old_claim.claim_id, ClaimStatus.SUPERSEDED)
        log.info(
            "substrate.supersession_pass2",
            session_id=new_claim.session_id,
            old_claim_id=old_claim.claim_id,
            claim_id=new_claim.claim_id,
            cosine=best_score,
        )
        return edge

    @staticmethod
    def _lookup_turn_text(conn: sqlite3.Connection, turn_id: str) -> str:
        row = conn.execute("SELECT text FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        return row["text"] if row is not None else ""
