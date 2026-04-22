"""Query-conditional retrieval over the claim store.

Stream A sixth-piece: the substrate already covers claim extraction, typed
supersession edges, span-based provenance, a closed predicate ontology, and
active-set projections. The missing piece is *retrieval* — a query-conditional
rank head so a downstream reader can pull back only the few facts it needs
from sessions that may contain thousands of turns.

Design (locked in `reasons.md` entries for this stream):

- **Sidecar embeddings table**, not an inline column on `claims`. Lets us
  re-embed (new model, new revision) without touching the main row and keeps
  the storage cost off installs that never call retrieval.
- **bge-m3** (MIT) as the embedding model, hosted on **Modal** (profile
  `glitch112213`, see `eval/infra/modal_bge_m3.py`). When `MODAL_BGE_M3_URL`
  is unset we fall back to the local `sentence-transformers` install — this
  makes the module unit-testable without a live deploy.
- **Normalize embeddings=True** server- *and* client-side, so cosine reduces
  to a dot product and is stable across runs.
- **Filter ordering**: supersession-active set is computed first, only then
  do we embed the question and rank. A superseded claim can never reach the
  reader even when its embedding is still similar.
- **Top-k default 20** (Zep precedent; ICLR 2025 LongMemEval leaderboard
  entries cluster around 10–25).
- **Cache** embeddings under `~/.cache/substrate/embeddings/`; write-time
  `embed_and_store` failures log a WARN and leave the claim un-embedded so
  a backfill pass can pick it up later rather than blocking ingestion.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.substrate.claims import list_active_claims
from src.substrate.schema import Claim

log = structlog.get_logger(__name__)


# --- constants ---------------------------------------------------------------

# bge-m3 revision pin. Change intentionally: re-embed the whole store when
# bumping so `embedding_model_version` rows stay in step with the vectors.
BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION = "main"  # HuggingFace branch pin
BGE_M3_VERSION_TAG = f"{BGE_M3_MODEL_ID}@{BGE_M3_MODEL_REVISION}"
BGE_M3_EMBED_DIM = 1024

MODAL_URL_ENV = "MODAL_BGE_M3_URL"
CACHE_DIR_ENV = "SUBSTRATE_EMBED_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "substrate" / "embeddings"

DEFAULT_TOP_K: int = 20


# --- dataclasses -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedClaim:
    """A retrieved claim with its cosine-similarity score and the vector version.

    `embedding_model_version` is carried through so callers can detect a
    cross-version mismatch (claim embedded with an older model than the
    question encoder) and fall back / re-embed as appropriate.
    """

    claim: Claim
    similarity_score: float
    embedding_model_version: str


# --- embedding client --------------------------------------------------------


class EmbeddingClient:
    """bge-m3 embedding client.

    Tries Modal first (HTTP POST to `{MODAL_BGE_M3_URL}/embed`). Falls back
    to local `sentence-transformers` when `MODAL_BGE_M3_URL` is unset — this
    keeps unit tests runnable without a deploy.

    Lazy imports keep `substrate.retrieval` importable on installs that
    don't have `sentence_transformers` or `requests` available.
    """

    def __init__(
        self,
        *,
        modal_url: str | None = None,
        model_version: str = BGE_M3_VERSION_TAG,
    ) -> None:
        self._modal_url = (modal_url or os.environ.get(MODAL_URL_ENV, "")).strip() or None
        self._model_version = model_version
        self._local_model: Any = None  # lazy-loaded sentence-transformers instance

    @property
    def model_version(self) -> str:
        return self._model_version

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts; returns one unit-length vector per input."""
        if not texts:
            return []
        if self._modal_url is not None:
            return self._embed_modal(texts)
        return self._embed_local(texts)

    # --- Modal path ---

    def _embed_modal(self, texts: list[str]) -> list[list[float]]:
        try:
            import requests  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "requests package not importable; add `requests` to runtime deps "
                "or unset MODAL_BGE_M3_URL to use the local sentence-transformers fallback."
            ) from exc
        url = self._modal_url.rstrip("/") + "/embed"  # type: ignore[union-attr]
        resp = requests.post(url, json={"texts": texts}, timeout=60.0)
        resp.raise_for_status()
        payload = resp.json()
        vectors = payload.get("embeddings") or []
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Modal embedding endpoint returned {len(vectors)} vectors for "
                f"{len(texts)} inputs"
            )
        # Prefer server-reported version so client can detect drift.
        server_version = payload.get("model_version")
        if isinstance(server_version, str) and server_version:
            self._model_version = server_version
        return [list(map(float, v)) for v in vectors]

    # --- local fallback path ---

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "sentence_transformers is not installed; either set "
                "MODAL_BGE_M3_URL (preferred) or install sentence-transformers."
            ) from exc
        if self._local_model is None:
            # Revision pinning via `revision=...`; SentenceTransformer forwards
            # kwargs to HF hub loader.
            self._local_model = SentenceTransformer(
                BGE_M3_MODEL_ID, revision=BGE_M3_MODEL_REVISION
            )
        raw = self._local_model.encode(texts, normalize_embeddings=True)
        out: list[list[float]] = []
        for v in raw:
            out.append([float(x) for x in v])
        return out


# --- blob encoding -----------------------------------------------------------


def _encode_vector(vec: list[float]) -> bytes:
    """Pack a float vector into a length-prefixed little-endian float32 blob.

    Stable across platforms because `struct` honours the explicit `<` byte
    order. `len(vec)` is stored in the first 4 bytes so decode is self-describing.
    """
    n = len(vec)
    return struct.pack(f"<I{n}f", n, *vec)


def _decode_vector(blob: bytes) -> list[float]:
    if len(blob) < 4:
        raise ValueError("embedding blob shorter than length prefix")
    (n,) = struct.unpack_from("<I", blob, 0)
    expected = 4 + 4 * n
    if len(blob) != expected:
        raise ValueError(
            f"embedding blob length {len(blob)} does not match declared dim {n}"
        )
    return list(struct.unpack_from(f"<{n}f", blob, 4))


# --- cache ------------------------------------------------------------------


def _cache_path() -> Path:
    """Resolve the on-disk cache dir; honours $SUBSTRATE_EMBED_CACHE_DIR for tests."""
    raw = os.environ.get(CACHE_DIR_ENV, "").strip()
    path = Path(raw) if raw else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(model_version: str, text: str) -> str:
    """Deterministic filename fragment for a (version, text) pair."""
    h = hashlib.sha256()
    h.update(model_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _cache_get(model_version: str, text: str) -> list[float] | None:
    path = _cache_path() / f"{_cache_key(model_version, text)}.bin"
    if not path.is_file():
        return None
    try:
        return _decode_vector(path.read_bytes())
    except ValueError:
        # Corrupt cache entry — nuke it and re-embed.
        path.unlink(missing_ok=True)
        return None


def _cache_put(model_version: str, text: str, vec: list[float]) -> None:
    path = _cache_path() / f"{_cache_key(model_version, text)}.bin"
    path.write_bytes(_encode_vector(vec))


# --- claim → identity text --------------------------------------------------


def claim_retrieval_text(claim: Claim) -> str:
    """Canonical text fed to the embedder for a claim.

    Includes value here (unlike `identity_text` in `supersession_semantic.py`)
    because retrieval wants full-signal content — "onset 3 days" and
    "onset 4 days" SHOULD embed differently so the reader can pick whichever
    matches the question. Supersession-time identity matching wants the
    opposite and thus excludes value.
    """
    return f"{claim.subject} {claim.predicate} {claim.value}"


# --- write-time embedder -----------------------------------------------------


def embed_and_store(
    conn: sqlite3.Connection,
    claim: Claim,
    *,
    client: EmbeddingClient | None = None,
) -> None:
    """Embed `claim` and upsert its vector into the sidecar table.

    Failure mode (per `reasons.md`): we log a WARN and leave the claim
    un-embedded. A backfill pass can re-run this later without touching
    the claims table. This way a flaky embedder never blocks the ingestion
    pipeline — retrieval simply degrades by skipping the un-embedded claim.
    """
    effective = client or EmbeddingClient()
    text = claim_retrieval_text(claim)
    try:
        # Check the on-disk cache first.
        cached = _cache_get(effective.model_version, text)
        if cached is not None:
            vec = cached
        else:
            vec = effective.embed([text])[0]
            _cache_put(effective.model_version, text, vec)
    except Exception as exc:  # noqa: BLE001 — documented degradation
        log.warning(
            "substrate.embed_and_store_failed",
            claim_id=claim.claim_id,
            error=repr(exc)[:200],
        )
        return

    blob = _encode_vector(vec)
    conn.execute(
        "INSERT OR REPLACE INTO claim_embeddings "
        "(claim_id, embedding, embedding_model_version, embedded_at_unix) "
        "VALUES (?, ?, ?, ?)",
        (claim.claim_id, blob, effective.model_version, int(time.time())),
    )


def backfill_embeddings(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    client: EmbeddingClient | None = None,
) -> int:
    """Embed every active claim in `session_id` that is not already in the sidecar.

    Returns the count of newly-embedded claims. Safe to run repeatedly —
    idempotent because we only insert missing rows.
    """
    effective = client or EmbeddingClient()
    active = list_active_claims(conn, session_id)
    missing = _filter_missing_embeddings(conn, active, effective.model_version)
    if not missing:
        return 0

    texts = [claim_retrieval_text(c) for c in missing]
    # Separate cached hits from misses so we only hit the network for misses.
    uncached_texts: list[str] = []
    uncached_claims: list[Claim] = []
    resolved: dict[str, list[float]] = {}
    for c, t in zip(missing, texts, strict=True):
        cached = _cache_get(effective.model_version, t)
        if cached is not None:
            resolved[c.claim_id] = cached
        else:
            uncached_texts.append(t)
            uncached_claims.append(c)

    if uncached_texts:
        try:
            vectors = effective.embed(uncached_texts)
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "substrate.backfill_embed_failed",
                session_id=session_id,
                error=repr(exc)[:200],
            )
            return len(resolved)
        for c, t, v in zip(uncached_claims, uncached_texts, vectors, strict=True):
            _cache_put(effective.model_version, t, v)
            resolved[c.claim_id] = v

    now = int(time.time())
    for claim_id, vec in resolved.items():
        conn.execute(
            "INSERT OR REPLACE INTO claim_embeddings "
            "(claim_id, embedding, embedding_model_version, embedded_at_unix) "
            "VALUES (?, ?, ?, ?)",
            (claim_id, _encode_vector(vec), effective.model_version, now),
        )
    return len(resolved)


def _filter_missing_embeddings(
    conn: sqlite3.Connection, claims: Iterable[Claim], model_version: str
) -> list[Claim]:
    out: list[Claim] = []
    for c in claims:
        row = conn.execute(
            "SELECT embedding_model_version FROM claim_embeddings WHERE claim_id = ?",
            (c.claim_id,),
        ).fetchone()
        if row is None:
            out.append(c)
        elif row["embedding_model_version"] != model_version:
            # Model version drift — re-embed.
            out.append(c)
    return out


# --- retrieval API -----------------------------------------------------------


def _cosine_normalised(a: list[float], b: list[float]) -> float:
    """Dot product — already-normalised vectors reduce cosine to this."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cosine_fallback(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
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


def retrieve_relevant_claims(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    question: str,
    branch: str | None = None,
    k: int = DEFAULT_TOP_K,
    client: EmbeddingClient | None = None,
) -> list[RankedClaim]:
    """Return top-k active claims by cosine similarity to `question`.

    Filter order (important — superseded claims must never reach the reader):

    1. `list_active_claims` → claims with status ∈ {active, confirmed}.
    2. Optional `branch` filter via predicate_pack sub_slot mapping.
    3. Left-outer-join onto `claim_embeddings`; claims without an embedding
       row (or with a mismatched `embedding_model_version`) are skipped
       with a DEBUG log. They can be recovered via `backfill_embeddings`.
    4. Embed the question once, dot-product against candidates, sort desc,
       take top-k.

    Time-window filtering was cut pre-merge — `Claim.created_ts` is the
    substrate's wall-clock-at-ingestion timestamp, not the original session
    time, so any window anchored on real-world dates would silently exclude
    every claim during a smoke run. Re-introducing time-aware retrieval
    requires a separate schema change (a `valid_from_ts` field carrying
    the originating turn's timestamp). See reasons.md.

    Never raises on empty substrate — returns `[]`.
    """
    if not question or not question.strip():
        return []

    effective = client or EmbeddingClient()
    model_version = effective.model_version

    active = list_active_claims(conn, session_id)
    if branch is not None:
        active = _filter_by_branch(active, branch)
    if not active:
        return []

    # Pull embeddings for the candidates in one query.
    ids = tuple(c.claim_id for c in active)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT claim_id, embedding, embedding_model_version "
        f"FROM claim_embeddings WHERE claim_id IN ({placeholders})",
        ids,
    ).fetchall()
    embedding_by_id: dict[str, tuple[list[float], str]] = {}
    for row in rows:
        try:
            vec = _decode_vector(row["embedding"])
        except ValueError:
            log.warning(
                "substrate.retrieval_decode_failed", claim_id=row["claim_id"]
            )
            continue
        embedding_by_id[row["claim_id"]] = (vec, row["embedding_model_version"])

    # Embed the question. Assume the client produces unit-length vectors
    # (bge-m3 server-side + local fallback both set normalize_embeddings=True).
    q_vec = effective.embed([question])[0]

    scored: list[tuple[float, Claim, str]] = []
    for c in active:
        entry = embedding_by_id.get(c.claim_id)
        if entry is None:
            log.debug("substrate.retrieval_skip_unembedded", claim_id=c.claim_id)
            continue
        vec, version = entry
        if version != model_version:
            log.debug(
                "substrate.retrieval_version_mismatch",
                claim_id=c.claim_id,
                claim_version=version,
                query_version=model_version,
            )
            continue
        # bge-m3 outputs unit vectors — dot is cosine. Use the safe fallback
        # if a non-normalised vector sneaks in (length ~ 1 is cheap to check).
        score = _cosine_normalised(q_vec, vec)
        if not (-1.01 <= score <= 1.01):
            score = _cosine_fallback(q_vec, vec)
        scored.append((score, c, version))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    return [
        RankedClaim(claim=c, similarity_score=s, embedding_model_version=v)
        for s, c, v in top
    ]


def _filter_by_branch(claims: list[Claim], branch: str) -> list[Claim]:
    """Filter claims to the given branch's sub-slot predicates.

    The active predicate pack declares `sub_slots: dict[branch -> predicates]`;
    if the pack doesn't recognise `branch` we return the claims unchanged
    (no silent empty list — that would mask a misconfigured branch name).
    """
    try:
        from src.substrate.predicate_packs import active_pack

        slots = active_pack().sub_slots
    except Exception:  # noqa: BLE001 — pack absent / misloaded: do not filter
        log.debug("substrate.retrieval_branch_pack_unavailable", branch=branch)
        return claims
    allowed = slots.get(branch)
    if allowed is None:
        log.warning(
            "substrate.retrieval_unknown_branch",
            branch=branch,
            known_branches=sorted(slots.keys())[:10],
        )
        return claims
    return [c for c in claims if c.predicate in allowed]


__all__ = [
    "BGE_M3_EMBED_DIM",
    "BGE_M3_MODEL_ID",
    "BGE_M3_MODEL_REVISION",
    "BGE_M3_VERSION_TAG",
    "DEFAULT_TOP_K",
    "EmbeddingClient",
    "RankedClaim",
    "backfill_embeddings",
    "claim_retrieval_text",
    "embed_and_store",
    "retrieve_relevant_claims",
]
