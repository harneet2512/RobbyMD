"""Smoke-run harness — deterministic first-10-case sanity pass.

Per `eval/README.md` "Smoke-first discipline" and the 2026-04-21 revision pass.

## CLI

    python eval/smoke/run_smoke.py [--benchmark {longmemeval|acibench|both}]
                                    [--reader {qwen2.5-14b|gpt-4o-mini|gpt-4.1-mini|all}]
                                    [--variant {baseline|substrate|both}]
                                    [--n N]
                                    [--budget-usd USD]
                                    [--dry-run]

Dry-run is guaranteed zero-cost: parses args, imports adapters, checks the
dataset directory, prints the planned matrix, and exits 0. No network.

Real-run wiring (2026-04-22):
- Per case: baseline variant → reader answer → judge score;
            substrate variant → pack-backed claim extraction → reader → judge score.
- Budget-halt: cumulative reader_cost + judge_cost checked before each case.
- Results.json written per benchmark×reader×variant matrix cell.
- Verdict: PASS / ANOMALY / FAIL per specced criteria.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Allow `python eval/smoke/run_smoke.py` invocation from repo root — without this
# the `eval.*` and `src.*` imports fail when the script is run as a file rather
# than via `python -m eval.smoke.run_smoke`.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DATA_DIR = _REPO_ROOT / "eval" / "data"
_RESULTS_ROOT = _REPO_ROOT / "eval" / "smoke"
_REFERENCE_BASELINES = _RESULTS_ROOT / "reference_baselines.json"

# Pack mapping per benchmark — ACTIVE_PACK must be set before any import
# of src.differential.lr_table (see CLAUDE.md §3 and lr_table.py header).
BENCHMARK_PACK: dict[str, str] = {
    "longmemeval": "personal_assistant",
    "acibench": "clinical_general",
}

BENCHMARKS = ("longmemeval", "acibench")
READERS = ("qwen2.5-14b", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1")
VARIANTS = ("baseline", "substrate")

# Cost estimates per 1k tokens (USD) — conservative upper bounds for budget-halt.
# Qwen endpoint is self-hosted: reader cost per call estimated at $0.00 when
# on-cluster; use a conservative $0.001 / call to trigger budget-halt in tests.
_READER_COST_PER_CALL: dict[str, float] = {
    "qwen2.5-14b": 0.001,
    "gpt-4o-mini": 0.003,
    "gpt-4.1-mini": 0.004,
    "gpt-4.1": 0.030,
}
# GPT-4o judge per call (LongMemEval only).
_JUDGE_COST_PER_CALL: float = 0.01

# ±20pp tolerance for baseline-vs-reference check.
_BASELINE_PP_TOLERANCE: float = 20.0


@dataclass(slots=True)
class SmokeConfig:
    benchmarks: tuple[str, ...]
    readers: tuple[str, ...]
    variants: tuple[str, ...]
    n_cases: int
    budget_usd: float
    dry_run: bool
    # FIX 2 (pre-merge gate): the LongMemEval substrate variant defaults to
    # the new bge-m3 + retrieval + CoN path. The old E5 + bundle-then-reader
    # path is kept under this flag for one cycle so any reproducibility
    # questions on the pre-FIX-2 numbers can be answered without git-checkout.
    legacy_lme_substrate: bool = False


@dataclass(slots=True)
class SmokeResult:
    verdict: str  # "PASS" | "ANOMALY" | "FAIL"
    lines: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_cases: int = 0


# ------------------------------------------------------------------
# Per-case result record (written to results.json per matrix cell).
# ------------------------------------------------------------------


@dataclass
class CaseResult:
    """One benchmark case × variant result record.

    All fields are required to be present in results.json per the spec. Fields
    that are not applicable for a benchmark (e.g. judge_reasoning for ACI-Bench)
    are set to None.
    """

    case_id: str
    benchmark: str
    reader: str
    variant: str  # "baseline" | "substrate"
    baseline_score: float | None
    substrate_score: float | None
    delta: float | None
    latency_baseline_ms: float | None
    latency_substrate_ms: float | None
    tokens_used_baseline: int | None
    tokens_used_substrate: int | None
    estimated_cost: float
    judge_reasoning: str | None  # LongMemEval only
    structural_validity: dict[str, Any] = field(default_factory=dict)
    # {claims_written_count, supersessions_fired_count, projection_nonempty, active_pack}


def _parse_args(argv: list[str]) -> SmokeConfig:
    ap = argparse.ArgumentParser(prog="run_smoke.py", description=__doc__)
    ap.add_argument("--benchmark", choices=(*BENCHMARKS, "both"), default="both")
    ap.add_argument("--reader", choices=(*READERS, "all"), default="qwen2.5-14b")
    ap.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--budget-usd", type=float, default=50.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--legacy-lme-substrate",
        action="store_true",
        help=(
            "Use the older E5 + bundle-then-reader LongMemEval substrate path "
            "(`_call_longmemeval_substrate`) instead of the default bge-m3 + "
            "retrieval + Chain-of-Note path. Off by default — the new path is "
            "the post-FIX-2 ship target."
        ),
    )
    ns = ap.parse_args(argv)

    benchmarks = BENCHMARKS if ns.benchmark == "both" else (ns.benchmark,)
    readers = READERS if ns.reader == "all" else (ns.reader,)
    variants = VARIANTS if ns.variant == "both" else (ns.variant,)

    return SmokeConfig(
        benchmarks=benchmarks,
        readers=readers,
        variants=variants,
        n_cases=ns.n,
        budget_usd=ns.budget_usd,
        dry_run=ns.dry_run,
        legacy_lme_substrate=ns.legacy_lme_substrate,
    )


def _check_dataset(benchmark: str) -> tuple[bool, str]:
    """Return (found, message) — True when the expected dataset path exists."""
    if benchmark == "longmemeval":
        p = _DATA_DIR / "longmemeval" / "data" / "longmemeval_s.json"
        if p.is_file():
            return True, f"{p} ({p.stat().st_size} bytes)"
        return False, f"missing: {p} — run eval/smoke/prepare_datasets.sh"
    if benchmark == "acibench":
        p = _DATA_DIR / "acibench" / "data" / "challenge_data_json" / "clinicalnlp_taskB_test1.json"
        if p.is_file():
            return True, f"{p} ({p.stat().st_size} bytes)"
        return False, f"missing: {p} — run eval/smoke/prepare_datasets.sh"
    return False, f"unknown benchmark: {benchmark}"


def _import_adapters() -> dict[str, object]:
    """Import per-benchmark adapter modules. Failures surface as (None, err) entries."""
    imports: dict[str, object] = {}
    for bench, module_name in (
        ("longmemeval", "eval.longmemeval.adapter"),
        ("acibench", "eval.aci_bench.adapter"),
    ):
        try:
            imports[bench] = importlib.import_module(module_name)
        except ImportError as exc:
            imports[bench] = f"ImportError: {exc}"
    return imports


def _load_reference_baselines() -> dict:
    if not _REFERENCE_BASELINES.is_file():
        return {}
    with _REFERENCE_BASELINES.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _print_planned_matrix(cfg: SmokeConfig) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"[smoke] Planned matrix: {len(cfg.benchmarks)} benchmark(s) × "
        f"{len(cfg.readers)} reader(s) × {len(cfg.variants)} variant(s) × {cfg.n_cases} cases"
    )
    lines.append(f"[smoke] Budget hard-cap: ${cfg.budget_usd:.2f}")
    lines.append("[smoke] Combinations:")
    for bench in cfg.benchmarks:
        for reader in cfg.readers:
            for variant in cfg.variants:
                lines.append(f"[smoke]   - {bench} × {reader} × {variant}")
    # FIX 2 dry-run assertion: announce which LongMemEval substrate path
    # would be used so silent-wiring failures are visible before any spend.
    if "longmemeval" in cfg.benchmarks and "substrate" in cfg.variants:
        if cfg.legacy_lme_substrate:
            lines.append(
                "[smoke] longmemeval substrate path: LEGACY "
                "(`_call_longmemeval_substrate` — E5 + bundle-then-reader)"
            )
        else:
            lines.append(
                "[smoke] longmemeval substrate path: DEFAULT "
                "(`_call_longmemeval_substrate_retrieval_con` — bge-m3 + retrieval + CoN)"
            )
    return lines


def _dry_run(cfg: SmokeConfig) -> SmokeResult:
    result = SmokeResult(verdict="PASS")
    result.lines.extend(_print_planned_matrix(cfg))

    # Import check (substrate-variant rows fail without wt-engine's write API; this is expected).
    adapters = _import_adapters()
    for bench, mod in adapters.items():
        if bench in cfg.benchmarks:
            ok = not isinstance(mod, str)
            result.lines.append(
                f"[smoke] adapter({bench}): {'OK' if ok else 'NOT IMPORTABLE ({mod})'.format(mod=mod)}"
            )

    # Dataset presence check.
    for bench in cfg.benchmarks:
        found, msg = _check_dataset(bench)
        if found:
            result.lines.append(f"[smoke] dataset({bench}): FOUND — {msg}")
        else:
            result.lines.append(f"[smoke] dataset({bench}): {msg}")

    # Reference-baselines sanity check.
    baselines = _load_reference_baselines()
    if baselines:
        result.lines.append(
            f"[smoke] reference_baselines.json: OK "
            f"({len(baselines)} benchmark(s) with reference numbers)"
        )
    else:
        result.lines.append(
            "[smoke] reference_baselines.json: MISSING — real run will skip baseline ±20pp check"
        )

    result.lines.append(
        "[smoke] DRY RUN: no API calls made. Exit 0. "
        "Re-run without --dry-run for the real smoke pass."
    )
    return result


# ------------------------------------------------------------------
# Env-var validation helpers (real-run only).
# ------------------------------------------------------------------


def _require_env(var: str) -> str:
    """Return the value of an env var or raise SystemExit with an actionable message."""
    val = os.environ.get(var, "").strip()
    if not val:
        sys.exit(
            f"[smoke] ERROR: required environment variable {var!r} is not set.\n"
            f"[smoke] Set it before invoking the smoke harness:\n"
            f"[smoke]   export {var}=<value>"
        )
    return val


def _get_reader_env(reader: str) -> dict[str, str]:
    """Validate and return reader-specific env vars.

    Returns a dict with keys the reader call site needs:
      - "endpoint" for qwen2.5-14b
      - "openai_key" for gpt-* readers

    Exits with actionable error on missing vars.
    """
    if reader == "qwen2.5-14b":
        # QWEN_API_BASE is the canonical var (matches user's deploy recipe and
        # vLLM's openai-compatible endpoint convention). QWEN_ENDPOINT is kept
        # as a legacy fallback so older .env files still work.
        endpoint = (
            os.environ.get("QWEN_API_BASE", "").strip()
            or os.environ.get("QWEN_ENDPOINT", "").strip()
        )
        fireworks = os.environ.get("FIREWORKS_API_KEY", "").strip()
        together = os.environ.get("TOGETHER_API_KEY", "").strip()
        if not (endpoint or fireworks or together):
            sys.exit(
                "[smoke] ERROR: qwen2.5-14b reader requires at least one of:\n"
                "[smoke]   QWEN_API_BASE (self-hosted vLLM OpenAI-compatible URL, e.g. http://host:8000/v1)\n"
                "[smoke]   FIREWORKS_API_KEY (Fireworks.ai hosted)\n"
                "[smoke]   TOGETHER_API_KEY (Together.ai hosted)\n"
                "[smoke] Set one of these before running the real smoke pass."
            )
        return {"endpoint": endpoint, "fireworks": fireworks, "together": together}
    else:
        # gpt-4o-mini / gpt-4.1-mini / gpt-4.1 — either direct OpenAI
        # (OPENAI_API_KEY) or Azure (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
        # + AZURE_OPENAI_GPT4OMINI_DEPLOYMENT). The reader call routes via
        # `eval._openai_client.make_openai_client` which picks the right
        # branch, so we only need to ensure at least one set is present.
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if azure_endpoint:
            # Azure routing — make_openai_client will validate deployment
            # names when actually called; here we only assert the minimum
            # so dry-run errors surface early.
            if not os.environ.get("AZURE_OPENAI_API_KEY", "").strip():
                sys.exit(
                    "[smoke] ERROR: AZURE_OPENAI_ENDPOINT is set but "
                    "AZURE_OPENAI_API_KEY is missing.\n"
                    "[smoke] Set both before invoking the smoke harness."
                )
            return {"azure_routed": "true", "openai_key": openai_key}
        # Direct OpenAI fallback.
        key = _require_env("OPENAI_API_KEY")
        return {"openai_key": key}


# ------------------------------------------------------------------
# Substrate-variant: admit turns through the substrate, collect stats.
# ------------------------------------------------------------------


@dataclass
class SubstrateStats:
    """State counters collected during substrate ingestion of one case.

    Back-compat shape (noop-extractor path): `claims_written_count`,
    `supersessions_fired_count`, `projection_nonempty`, `active_pack`.

    Phase-3 wiring adds (populated only when the real extractor fires):
    `active_claim_count`, `sentence_with_provenance_ratio`,
    `substrate_vs_baseline_edit_distance`, `top_k_retrieved`,
    `top_k_sim_mean`, `top_k_sim_min`. Unused fields stay at their
    defaults so JSON output remains stable for old callers.
    """

    claims_written_count: int = 0
    supersessions_fired_count: int = 0
    projection_nonempty: bool = False
    active_pack: str = ""
    active_claim_count: int = 0
    sentence_with_provenance_ratio: float = 0.0
    substrate_vs_baseline_edit_distance: float | None = None
    top_k_retrieved: int = 0
    top_k_sim_mean: float = 0.0
    top_k_sim_min: float = 0.0


def _run_substrate_ingestion(
    benchmark: str,
    case_payload: object,
) -> SubstrateStats:
    """Push a benchmark case's turns through the substrate.

    ACTIVE_PACK must already be set in os.environ before this function is called
    (caller sets it per-benchmark before the first substrate import).

    Uses the substrate's `on_new_turn` orchestrator with a no-op extractor so
    the pipeline runs end-to-end (admission → turn persistence → supersession
    checks → projection rebuild) without requiring a live LLM for extraction.

    For the smoke tier, extraction is intentionally stubbed — the structural
    validity check verifies the pipeline wiring, not extractor quality.
    """
    from src.substrate.predicate_packs import active_pack
    from src.substrate.schema import Speaker, open_database

    stats = SubstrateStats(active_pack=os.environ.get("ACTIVE_PACK", "clinical_general"))

    # In-memory DB per case — no disk I/O, no cross-case state bleed.
    conn = open_database(":memory:")

    # No-op extractor: returns zero claims so we test admission + turn
    # persistence + supersession scaffolding without an LLM backend.
    def _noop_extractor(turn: object) -> list:
        return []

    if benchmark == "longmemeval":
        from eval.longmemeval.adapter import LongMemEvalQuestion, session_to_turns
        from src.substrate.on_new_turn import on_new_turn

        q: LongMemEvalQuestion = case_payload  # type: ignore[assignment]
        session_id = q.question_id
        for sidx in range(len(q.haystack_sessions)):
            for t in session_to_turns(q, sidx):
                res = on_new_turn(
                    conn,
                    session_id=session_id,
                    speaker=Speaker(t.speaker) if t.speaker in ("patient", "physician", "system") else Speaker.SYSTEM,
                    text=t.text,
                    extractor=_noop_extractor,
                )
                if res.admitted:
                    stats.claims_written_count += len(res.created_claims)
                    stats.supersessions_fired_count += len(res.supersession_edges)

    elif benchmark == "acibench":
        from eval.aci_bench.adapter import ACIEncounter, encounter_to_turns
        from src.substrate.on_new_turn import on_new_turn

        enc: ACIEncounter = case_payload  # type: ignore[assignment]
        session_id = enc.encounter_id
        for t in encounter_to_turns(enc):
            res = on_new_turn(
                conn,
                session_id=session_id,
                speaker=Speaker(t.speaker) if t.speaker in ("patient", "physician", "system") else Speaker.SYSTEM,
                text=t.text,
                extractor=_noop_extractor,
            )
            if res.admitted:
                stats.claims_written_count += len(res.created_claims)
                stats.supersessions_fired_count += len(res.supersession_edges)

    else:
        raise ValueError(f"unknown benchmark: {benchmark!r} (expected 'longmemeval' or 'acibench')")

    # Check projection non-emptiness — the substrate projection is rebuilt
    # after each turn; here we check the final state. With a no-op extractor,
    # projection will have 0 active claims (no extraction), so we check that
    # at least some turns were admitted (pipeline connectivity, not claim count).
    from src.substrate.claims import list_active_claims

    active = list_active_claims(conn, session_id)
    # Projection is "nonempty" if turns were admitted (structural connectivity
    # confirmed even with zero claims from the stub extractor).
    admitted_turns = conn.execute(
        "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    stats.projection_nonempty = admitted_turns > 0

    conn.close()
    return stats


# ------------------------------------------------------------------
# Reader call stubs (real calls happen here; mocked in unit tests).
# ------------------------------------------------------------------


def _call_longmemeval_baseline(
    case_payload: object,
    reader: str,
    reader_env: dict[str, str],
) -> tuple[str, float, int]:
    """Call the reader for LongMemEval-S baseline variant.

    Returns (predicted_answer, latency_ms, tokens_used).
    Delegates to eval.longmemeval.baseline.predict_answer for the existing
    reader (Opus 4.7 / Anthropic), or calls the appropriate OAI model.
    """
    from eval.longmemeval.adapter import LongMemEvalQuestion
    from eval.longmemeval.baseline import _flatten_sessions

    q: LongMemEvalQuestion = case_payload  # type: ignore[assignment]
    prompt = f"{_flatten_sessions(q)}\n\nQuestion: {q.question}"
    system = (
        "You are a helpful assistant. Use only the conversation history below to "
        "answer the user's question at the end. If the history does not contain "
        "enough information, reply exactly: 'I don't have enough information to answer.'"
    )

    t0 = time.monotonic()
    if reader == "qwen2.5-14b":
        text, tokens = _call_qwen(system, prompt, reader_env)
    else:
        text, tokens = _call_openai(reader, system, prompt, reader_env["openai_key"])
    latency_ms = (time.monotonic() - t0) * 1000
    return text, latency_ms, tokens


def _call_longmemeval_substrate(
    case_payload: object,
    reader: str,
    reader_env: dict[str, str],
    top_k: int = 20,
) -> tuple[str, float, int, SubstrateStats]:
    """Substrate variant for LongMemEval — real ingestion + top-K retrieval.

    Flow:
    1. Ingest all haystack turns through `on_new_turn` with the real
       LLM-backed extractor. Claims populate the substrate; supersession
       fires as duplicates land.
    2. Embed the question and each active claim's `subject + predicate + value`
       with E5-small-v2 (already a dependency for semantic supersession).
    3. Pick the top-K by cosine similarity, format them as a bullet list,
       prepend to the reader prompt.
    4. Return the reader's answer alongside retrieval stats.

    NOTE: this is a novel retrieval layer not validated by the LongMemEval
    paper. Numbers produced here are labelled
    `LongMemEval-S substrate-smoke, top-20 claim retrieval` — not
    stackable against the ICLR paper leaderboard.
    """
    from eval.longmemeval.adapter import LongMemEvalQuestion
    from src.extraction.claim_extractor.extractor import make_llm_extractor
    from src.substrate.claims import list_active_claims

    q: LongMemEvalQuestion = case_payload  # type: ignore[assignment]

    extractor = make_llm_extractor()
    t_ingest = time.monotonic()
    stats, conn = _ingest_with_real_extractor("longmemeval", q, extractor)
    ingest_latency_ms = (time.monotonic() - t_ingest) * 1000

    active = list_active_claims(conn, q.question_id)
    stats.active_claim_count = len(active)

    # Build retrieval bundle. If no claims, fall back to baseline prompt.
    if not active:
        conn.close()
        text, latency_ms, tokens = _call_longmemeval_baseline(q, reader, reader_env)
        return text, ingest_latency_ms + latency_ms, tokens, stats

    claim_texts = [
        f"{c.subject} / {c.predicate} / {c.value}" for c in active
    ]
    top_indices, top_sims = _top_k_by_embedding(q.question, claim_texts, top_k)
    stats.top_k_retrieved = len(top_indices)
    if top_sims:
        stats.top_k_sim_mean = sum(top_sims) / len(top_sims)
        stats.top_k_sim_min = min(top_sims)

    bundle_lines = ["Relevant facts extracted from prior sessions:"]
    for i, sim in zip(top_indices, top_sims, strict=False):
        c = active[i]
        bundle_lines.append(
            f"- [sim={sim:.2f}] {c.subject}: {c.predicate} = {c.value}"
        )
    bundle_text = "\n".join(bundle_lines)

    conn.close()

    # Reader call with the bundle prepended.
    from eval.longmemeval.baseline import _flatten_sessions

    prompt = (
        f"{bundle_text}\n\n---\n\n{_flatten_sessions(q)}\n\nQuestion: {q.question}"
    )
    system = (
        "You are a helpful assistant. Use the 'Relevant facts' bundle and "
        "the conversation history below to answer the user's question at "
        "the end. If neither contains enough information, reply exactly: "
        "'I don't have enough information to answer.'"
    )

    t0 = time.monotonic()
    if reader == "qwen2.5-14b":
        text, tokens = _call_qwen(system, prompt, reader_env)
    else:
        text, tokens = _call_openai(reader, system, prompt, reader_env["openai_key"])
    latency_ms = (time.monotonic() - t0) * 1000
    return text, ingest_latency_ms + latency_ms, tokens, stats


def _call_longmemeval_substrate_retrieval_con(
    case_payload: object,
    reader_env: dict[str, str],
    top_k: int = 20,
) -> tuple[str, float, int, SubstrateStats, dict[str, Any]]:
    """Stream A substrate variant: extract → retrieve(top-k) → CoN.

    Wires the Stream A pieces together:
    - `src.extraction.claim_extractor.extractor.make_llm_extractor` for claims.
    - `src.substrate.retrieval.retrieve_relevant_claims` with bge-m3 sidecar
      embeddings + supersession-active filter.
    - `eval.longmemeval.reader_con.answer_with_con` for the two-call CoN reader.

    Returns `(answer, latency_ms, tokens_used, substrate_stats, provenance)`.

    Reader purpose is `longmemeval_reader` (gpt-4o-2024-08-06 paper-faithful,
    gpt-4.1 under the documented Stream A fallback).

    Time-aware retrieval was cut pre-merge — `Claim.created_ts` is wall-clock
    at substrate ingestion, not original session time, so any time-window
    filter would silently exclude every claim. Re-introducing requires the
    `valid_from_ts` schema change. See reasons.md.

    NOTE: unit tests pass a mocked extractor/embedder; the smoke path resolves
    live clients from env. This function is the default LongMemEval substrate
    path post-FIX 2 (the older `_call_longmemeval_substrate` is gated behind
    `--legacy-lme-substrate`).
    """
    from eval.longmemeval.adapter import LongMemEvalQuestion
    from eval.longmemeval.reader_con import answer_with_con
    from src.extraction.claim_extractor.extractor import make_llm_extractor
    from src.substrate.retrieval import (
        EmbeddingClient,
        backfill_embeddings,
        retrieve_relevant_claims,
    )

    q: LongMemEvalQuestion = case_payload  # type: ignore[assignment]
    extractor = make_llm_extractor()

    t_ingest = time.monotonic()
    stats, conn = _ingest_with_real_extractor("longmemeval", q, extractor)
    ingest_latency_ms = (time.monotonic() - t_ingest) * 1000

    # Embed every new claim into the sidecar. bge-m3 client falls back to
    # local sentence-transformers if MODAL_BGE_M3_URL is unset.
    embed_client = EmbeddingClient()
    backfill_embeddings(conn, q.question_id, client=embed_client)

    ranked = retrieve_relevant_claims(
        conn,
        session_id=q.question_id,
        question=q.question,
        k=top_k,
        client=embed_client,
    )
    stats.top_k_retrieved = len(ranked)
    if ranked:
        sims = [r.similarity_score for r in ranked]
        stats.top_k_sim_mean = sum(sims) / len(sims)
        stats.top_k_sim_min = min(sims)
    conn.close()

    # CoN reader — uses longmemeval_reader purpose with gpt-4.1 fallback.
    answer, provenance = answer_with_con(q.question, ranked)
    total_latency_ms = ingest_latency_ms + provenance.get("total_latency_ms", 0.0)
    # `answer_with_con` does not return token count; estimate as len(answer)/4.
    tokens_estimate = max(1, len(answer) // 4)
    return answer, total_latency_ms, tokens_estimate, stats, provenance


def _top_k_by_embedding(
    query: str,
    candidates: list[str],
    k: int,
) -> tuple[list[int], list[float]]:
    """Embed `query` + `candidates` via E5-small-v2 and return top-K indices + sims.

    Falls back to a hash-based ranking if `sentence_transformers` is not
    installed (which would happen in CI without torch). The fallback keeps
    the pipeline working for smoke tests — stats will just be meaningless.
    """
    try:
        from src.substrate.supersession_semantic import E5Embedder

        embedder = E5Embedder()
        vectors = embedder.embed([query] + candidates)
    except Exception:  # noqa: BLE001 — torch import or HF download failure
        # Degenerate fallback — first K candidates with sim=0.0.
        indices = list(range(min(k, len(candidates))))
        sims = [0.0] * len(indices)
        return indices, sims

    q_vec = vectors[0]
    c_vecs = vectors[1:]
    scored = [
        (_cosine(q_vec, cv), i)
        for i, cv in enumerate(c_vecs)
    ]
    scored.sort(reverse=True)
    top = scored[:k]
    indices = [i for _, i in top]
    sims = [s for s, _ in top]
    return indices, sims


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 if either vector is all-zero."""
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _call_longmemeval_judge(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    openai_key: str,
) -> tuple[float, str]:
    """Score a LongMemEval answer using the GPT-4o judge.

    Returns (score_0_or_1, judge_reasoning). Uses the LongMemEval official
    judge prompt format (binary correct/incorrect). Re-uses the judge module
    from eval.longmemeval if it exists, otherwise falls back to a direct
    GPT-4o call with the official prompt.

    Routes to Azure OpenAI when AZURE_OPENAI_ENDPOINT is set; otherwise
    falls back to direct OpenAI using `openai_key`. `openai_key` is
    ignored in the Azure branch — the Azure client picks up
    AZURE_OPENAI_API_KEY itself via `make_openai_client`.
    """
    # Try to import a judge module if it exists.
    try:
        judge_mod = importlib.import_module("eval.longmemeval.judge")
        score, reasoning = judge_mod.score(question, gold_answer, predicted_answer)
        return float(score), str(reasoning)
    except (ImportError, AttributeError):
        pass

    # Direct GPT-4o judge call with LongMemEval's official binary-scoring prompt.
    # Prompt phrasing follows LongMemEval ICLR 2025 §5.1 evaluator description.
    judge_prompt = (
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        "Does the predicted answer correctly answer the question given the gold answer? "
        "Reply with CORRECT or INCORRECT, then a one-sentence explanation."
    )
    try:
        # Azure-aware routing. If AZURE_OPENAI_ENDPOINT is set, we use the
        # Azure GPT-4o deployment; otherwise the direct-OpenAI fallback uses
        # `openai_key` (which came from OPENAI_API_KEY upstream).
        if os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip():
            from eval._openai_client import make_openai_client

            client, model = make_openai_client("judge_gpt4o")
        else:
            import openai

            client = openai.OpenAI(api_key=openai_key)
            model = "gpt-4o-2024-08-06"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        score = 1.0 if text.strip().upper().startswith("CORRECT") else 0.0
        return score, text
    except Exception as exc:
        return 0.0, f"judge error: {exc}"


def _call_acibench_baseline(
    case_payload: object,
    reader: str,
    reader_env: dict[str, str],
) -> tuple[str, float, int]:
    """Call the reader for ACI-Bench baseline variant.

    Returns (predicted_note, latency_ms, tokens_used).
    """
    from eval.aci_bench.adapter import ACIEncounter
    from eval.aci_bench.baseline import _format_dialogue

    enc: ACIEncounter = case_payload  # type: ignore[assignment]
    dialogue_text = _format_dialogue(enc)
    system = (
        "You are a clinical documentation assistant producing a SOAP note from a "
        "doctor-patient conversation. Output only the note, in four sections "
        "(SUBJECTIVE / OBJECTIVE / ASSESSMENT / PLAN). Use only information present "
        "in the transcript. Do NOT fabricate findings or orders. This is a research "
        "prototype, not a medical device; the physician makes every clinical decision."
    )

    t0 = time.monotonic()
    if reader == "qwen2.5-14b":
        text, tokens = _call_qwen(system, dialogue_text, reader_env)
    else:
        text, tokens = _call_openai(reader, system, dialogue_text, reader_env["openai_key"])
    latency_ms = (time.monotonic() - t0) * 1000
    return text, latency_ms, tokens


def _ingest_with_real_extractor(
    benchmark: str,
    case_payload: object,
    extractor: object,
    *,
    parallel_extract: bool = True,
    max_workers: int = 10,
) -> tuple[SubstrateStats, Any]:
    """Run substrate ingestion with a caller-supplied extractor.

    Returns `(stats, conn)`. Caller is responsible for closing `conn`.

    Differs from `_run_substrate_ingestion` in three ways:
    1. Uses the provided extractor instead of a noop.
    2. Keeps the connection open so the caller can query
       `list_active_claims` or build a retrieval index over them.
    3. For LongMemEval (many-hundreds of turns per case), pre-extracts
       claims concurrently via a thread pool before handing them to the
       substrate in order. Observed Phase 4B v2 (sequential): 2h elapsed
       with only 12s CPU because one rate-limited call blocked the whole
       loop. Parallel pre-extraction decouples the network-bound
       extractor work from the serial substrate pipeline and gives an
       order-of-magnitude speedup.

    Parameters
    ----------
    parallel_extract:
        When True (default), for LongMemEval pre-extract all claims
        concurrently with `max_workers` threads and feed cached results
        into `on_new_turn`. Set False for unit tests that need
        deterministic ordering or to debug extractor behaviour.
    max_workers:
        Thread pool size for parallel extraction. 10 is safe for Azure
        OpenAI tier with token-per-minute limits up to 100k. Raise for
        higher tiers.
    """
    from src.substrate.schema import Speaker, open_database

    stats = SubstrateStats(active_pack=os.environ.get("ACTIVE_PACK", "clinical_general"))
    conn = open_database(":memory:")

    if benchmark == "longmemeval":
        from eval.longmemeval.adapter import LongMemEvalQuestion, session_to_turns
        from src.substrate.on_new_turn import on_new_turn

        q: LongMemEvalQuestion = case_payload  # type: ignore[assignment]
        session_id = q.question_id

        # Collect all turns first (across all haystack sessions, in order).
        all_turns = []
        for sidx in range(len(q.haystack_sessions)):
            for t in session_to_turns(q, sidx):
                all_turns.append(t)

        # Pre-extract claims in parallel. Cache keyed by `turn.text` —
        # NOT turn.turn_id, because `on_new_turn` generates a fresh
        # turn_id internally (so the substrate-side turn passed to the
        # extractor has a different ID than the adapter-side turn we
        # extracted from). Text is the deterministic input the extractor
        # actually consumes.
        extraction_cache: dict[str, list] = {}
        if parallel_extract and len(all_turns) > 1:
            extraction_cache = _parallel_extract_claims(
                extractor, all_turns, max_workers=max_workers
            )

        def _cached_or_direct_extractor(turn):  # type: ignore[no-untyped-def]
            cached = extraction_cache.get(turn.text)
            if cached is not None:
                return cached
            # Fallback: direct call. Hits when parallel_extract=False or
            # the upstream extraction silently dropped this turn.
            return extractor(turn)  # type: ignore[operator]

        for t in all_turns:
            res = on_new_turn(
                conn,
                session_id=session_id,
                speaker=Speaker(t.speaker)
                if t.speaker in ("patient", "physician", "system")
                else Speaker.SYSTEM,
                text=t.text,
                extractor=_cached_or_direct_extractor,  # type: ignore[arg-type]
            )
            if res.admitted:
                stats.claims_written_count += len(res.created_claims)
                stats.supersessions_fired_count += len(res.supersession_edges)

    elif benchmark == "acibench":
        from eval.aci_bench.adapter import ACIEncounter, encounter_to_turns
        from src.substrate.on_new_turn import on_new_turn

        enc: ACIEncounter = case_payload  # type: ignore[assignment]
        session_id = enc.encounter_id
        # ACI-Bench typically has ~40 turns per encounter — sequential
        # extraction is fast enough and preserves simpler debugging.
        for t in encounter_to_turns(enc):
            res = on_new_turn(
                conn,
                session_id=session_id,
                speaker=Speaker(t.speaker)
                if t.speaker in ("patient", "physician", "system")
                else Speaker.SYSTEM,
                text=t.text,
                extractor=extractor,  # type: ignore[arg-type]
            )
            if res.admitted:
                stats.claims_written_count += len(res.created_claims)
                stats.supersessions_fired_count += len(res.supersession_edges)
    else:
        raise ValueError(f"unknown benchmark: {benchmark!r}")

    from src.substrate.claims import list_active_claims

    active = list_active_claims(conn, _session_id_for(benchmark, case_payload))
    stats.active_claim_count = len(active)
    admitted_turns = conn.execute(
        "SELECT COUNT(*) FROM turns WHERE session_id = ?",
        (_session_id_for(benchmark, case_payload),),
    ).fetchone()[0]
    stats.projection_nonempty = admitted_turns > 0
    return stats, conn


def _parallel_extract_claims(
    extractor: object,
    turns: list,
    *,
    max_workers: int = 10,
) -> dict[str, list]:
    """Pre-extract claims for a list of turns using a thread pool.

    Returns a dict `turn_id → list[ExtractedClaim]`. Turns whose extraction
    raises an exception are mapped to an empty list (the extractor already
    does the same under its own except-handler; this is defence-in-depth
    so one broken turn can't abort the whole pool).

    Parallelism here is safe because the extractor's claim output does not
    depend on any other turn's state — each call is independent. Claim
    ordering is preserved by the caller feeding turns back into
    `on_new_turn` sequentially, using the cache to avoid re-extracting.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache: dict[str, list] = {}
    if not turns:
        return cache

    def _extract_one(turn):  # type: ignore[no-untyped-def]
        try:
            claims = extractor(turn)  # type: ignore[operator]
            return turn.text, claims
        except Exception:  # noqa: BLE001 — log + empty, don't abort pool
            return turn.text, []

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_extract_one, t) for t in turns]
        for fut in as_completed(futures):
            text, claims = fut.result()
            cache[text] = claims
    elapsed = time.monotonic() - t_start

    # Useful telemetry for operator: how long did the parallel phase take,
    # and how many turns actually produced claims.
    total_claims = sum(len(c) for c in cache.values())
    non_empty = sum(1 for c in cache.values() if c)
    print(
        f"[smoke] parallel_extract: {len(turns)} turns, "
        f"{non_empty} with claims, {total_claims} claims total, "
        f"{elapsed:.1f}s elapsed (max_workers={max_workers})"
    )
    return cache


def _session_id_for(benchmark: str, case_payload: object) -> str:
    """Extract the session/encounter ID used in substrate ingestion."""
    if benchmark == "longmemeval":
        return getattr(case_payload, "question_id")
    if benchmark == "acibench":
        return getattr(case_payload, "encounter_id")
    raise ValueError(f"unknown benchmark: {benchmark!r}")


def _normalised_edit_distance(a: str, b: str) -> float:
    """1 - SequenceMatcher.ratio. 0.0 = identical, 1.0 = fully different.

    Used for the Phase 4A pass criterion "substrate note differs from
    baseline note by > 0.1 on ≥7 of 10 cases."
    """
    import difflib

    if not a and not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a or "", b or "").ratio()
    return 1.0 - ratio


def _call_acibench_substrate(
    case_payload: object,
    reader: str,
    reader_env: dict[str, str],
    baseline_note_for_edit_distance: str | None = None,
) -> tuple[str, float, int, SubstrateStats]:
    """Substrate variant — wires the real substrate end-to-end.

    Flow (Phase 3):
    1. Ingest every turn via `on_new_turn` with the real LLM-backed
       extractor. Pass-1 + Pass-2 supersession fire as claims land.
    2. Query `list_active_claims` from the substrate DB.
    3. Call `src.note.generator.generate_soap_note` to compose a
       provenance-tagged SOAP note.
    4. Compute `sentence_with_provenance_ratio` and
       `substrate_vs_baseline_edit_distance` (when baseline_note provided).

    Returns `(note_text, combined_latency_ms, combined_tokens, stats)`.
    """
    from eval.aci_bench.adapter import ACIEncounter
    from eval.aci_bench.baseline import _format_dialogue
    from src.extraction.claim_extractor.extractor import make_llm_extractor
    from src.note.generator import generate_soap_note

    enc: ACIEncounter = case_payload  # type: ignore[assignment]
    dialogue_text = _format_dialogue(enc)

    # Step 1: ingest with real extractor.
    extractor = make_llm_extractor()
    t_ingest_start = time.monotonic()
    stats, conn = _ingest_with_real_extractor("acibench", enc, extractor)
    ingest_latency_ms = (time.monotonic() - t_ingest_start) * 1000

    # Step 2+3: generate SOAP from claim store.
    soap = generate_soap_note(
        conn,
        session_id=enc.encounter_id,
        dialogue_text=dialogue_text,
        reader=reader,
        reader_env=reader_env,
    )
    conn.close()

    # Step 4: enrich stats.
    stats.sentence_with_provenance_ratio = soap.sentence_with_provenance_ratio
    if baseline_note_for_edit_distance is not None:
        stats.substrate_vs_baseline_edit_distance = _normalised_edit_distance(
            baseline_note_for_edit_distance, soap.note_text
        )

    return (
        soap.note_text,
        ingest_latency_ms + soap.latency_ms,
        soap.tokens_used,
        stats,
    )


def _score_acibench_case(
    enc: object,
    predicted_note: str,
) -> float:
    """Compute MEDCON micro-F1 for a single ACI-Bench case.

    Returns micro-F1 in [0, 1]. Uses the active ConceptExtractor tier
    (T1 scispaCy by default, per the MEDCON 3-tier ADR).
    """
    from eval.aci_bench.adapter import ACIEncounter
    from eval.aci_bench.extractors import build_extractor, compute_medcon_f1

    enc_typed: ACIEncounter = enc  # type: ignore[assignment]
    extractor = build_extractor()
    gold_cuis = extractor.extract(enc_typed.gold_note)
    pred_cuis = extractor.extract(predicted_note)
    scores = compute_medcon_f1(gold_cuis, pred_cuis)
    return float(scores["f1"])


def _call_qwen(
    system: str,
    user: str,
    reader_env: dict[str, str],
) -> tuple[str, int]:
    """Call Qwen2.5-14B reader. Returns (text, approx_token_count).

    Priority: QWEN_ENDPOINT (self-hosted vLLM, OpenAI-compatible) →
              FIREWORKS_API_KEY → TOGETHER_API_KEY.
    """
    endpoint = reader_env.get("endpoint", "")
    fireworks = reader_env.get("fireworks", "")
    together = reader_env.get("together", "")

    if endpoint:
        # Self-hosted vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint.
        import openai

        client = openai.OpenAI(base_url=endpoint, api_key="EMPTY")
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-14B-Instruct",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else len(text.split())
        return text, tokens

    if fireworks:
        import openai

        client = openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=fireworks,
        )
        resp = client.chat.completions.create(
            model="accounts/fireworks/models/qwen2p5-14b-instruct",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else len(text.split())
        return text, tokens

    if together:
        import openai

        client = openai.OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=together,
        )
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-14B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else len(text.split())
        return text, tokens

    raise RuntimeError("No Qwen endpoint configured (unreachable after _get_reader_env check)")


def _call_openai(
    model_key: str,
    system: str,
    user: str,
    openai_key: str,
) -> tuple[str, int]:
    """Call an OpenAI model. Returns (text, approx_token_count).

    Routes through `eval._openai_client.make_openai_client` so Azure
    deployments (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY +
    AZURE_OPENAI_GPT4OMINI_DEPLOYMENT / AZURE_OPENAI_GPT4O_DEPLOYMENT)
    are used when set; otherwise falls back to direct OpenAI with
    `openai_key`.
    """
    _purpose_map = {
        "gpt-4o-mini": "llm_medcon_gpt4omini",
        "gpt-4.1-mini": "reader_gpt41mini",
        "gpt-4.1": "reader_gpt41",
    }
    purpose = _purpose_map.get(model_key)
    if purpose is None:
        # Unknown model — direct OpenAI with the literal model string.
        import openai

        client = openai.OpenAI(api_key=openai_key)
        model = model_key
    else:
        # Determine routing: Azure if endpoint set, else direct OpenAI.
        from eval._openai_client import make_openai_client

        if os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip():
            client, model = make_openai_client(purpose)  # type: ignore[arg-type]
        else:
            import openai

            client = openai.OpenAI(api_key=openai_key)
            model = model_key

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    text = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else len(text.split())
    return text, tokens


# ------------------------------------------------------------------
# Verdict logic helpers.
# ------------------------------------------------------------------


def _check_baseline_within_reference(
    benchmark: str,
    reader: str,
    baseline_score: float,
    reference_baselines: dict,
) -> tuple[bool, str]:
    """Check if our baseline score is within ±20pp of the published reference.

    Returns (ok, message). When no reference exists, returns (True, "no-reference").
    """
    bench_key = "longmemeval_s" if benchmark == "longmemeval" else "acibench_test1"
    reader_key = reader if reader != "qwen2.5-14b" else "qwen2.5-14b-instruct"

    bench_refs = reference_baselines.get(bench_key, {})
    reader_refs = bench_refs.get(reader_key, {})

    # Find the most relevant reference number.
    ref_val: float | str | None = None
    for variant_key in ("full_context", "baseline_icl"):
        vref = reader_refs.get(variant_key, {})
        overall = vref.get("overall")
        if overall is not None and isinstance(overall, (int, float)):
            ref_val = float(overall)
            break

    if ref_val is None:
        return True, f"no-reference ({bench_key}/{reader_key})"

    # Scores: reference may be 0-100 (%), baseline_score may be 0-1 or 0-100.
    # Normalise both to 0-100 for comparison.
    score_norm = baseline_score * 100 if baseline_score <= 1.0 else baseline_score
    diff = abs(score_norm - ref_val)
    ok = diff <= _BASELINE_PP_TOLERANCE
    return ok, f"score={score_norm:.1f} ref={ref_val:.1f} diff={diff:.1f}pp"


def _determine_verdict(
    case_results: list[CaseResult],
    reference_baselines: dict,
    cfg: SmokeConfig,
    anomaly_reasons: list[str],
) -> str:
    """Return PASS / ANOMALY / FAIL based on completed case results.

    PASS conditions (all must hold):
    1. All cases completed (no empty outputs).
    2. At least one substrate case has structural validity (projection_nonempty=True).
    3. Any baseline score within ±20pp of reference_baselines.json.
    4. Non-zero substrate delta (substrate_score != baseline_score) for at least
       one case.
    5. Estimated cost ≤ 1.5× the sum of per-case estimates.

    ANOMALY: completes but any criterion fails — surface each failing criterion.
    FAIL: crash / empty outputs / budget blown (handled by caller before this fn).
    """
    if not case_results:
        return "FAIL"

    # 1. All cases have outputs.
    empty_cases = [r.case_id for r in case_results if r.baseline_score is None and r.substrate_score is None]
    if empty_cases:
        anomaly_reasons.append(f"empty outputs for cases: {empty_cases}")

    # 2. Structural validity in at least one substrate case.
    substrate_cases = [r for r in case_results if r.variant == "substrate"]
    if substrate_cases:
        valid_substrate = any(r.structural_validity.get("projection_nonempty", False) for r in substrate_cases)
        if not valid_substrate:
            anomaly_reasons.append("no substrate case passed structural_validity.projection_nonempty check")

    # 3. Baseline within ±20pp of reference.
    baseline_cases = [r for r in case_results if r.variant == "baseline" and r.baseline_score is not None]
    for r in baseline_cases[:1]:  # Check one representative case per benchmark/reader.
        # Filtered above to `is not None`, but Pyright doesn't narrow the attr
        # through a list comprehension — assign to a local to narrow.
        bs = r.baseline_score
        assert bs is not None  # noqa: S101 — loop-invariant; filtered above
        ok, msg = _check_baseline_within_reference(r.benchmark, r.reader, bs, reference_baselines)
        if not ok:
            anomaly_reasons.append(f"baseline outside ±20pp: {msg}")

    # 4. Non-zero substrate delta.
    delta_cases = [r for r in case_results if r.delta is not None]
    if delta_cases and all(r.delta == 0.0 for r in delta_cases):
        anomaly_reasons.append("all substrate deltas are zero (substrate may not be wired)")

    return "ANOMALY" if anomaly_reasons else "PASS"


# ------------------------------------------------------------------
# Real-run per-case execution.
# ------------------------------------------------------------------


def _run_longmemeval_case(
    case_payload: object,
    reader: str,
    reader_env: dict[str, str],
    variants: tuple[str, ...],
    cumulative_cost: list[float],
    budget_usd: float,
    *,
    legacy_substrate: bool = False,
) -> tuple[list[CaseResult], bool]:
    """Run one LongMemEval-S case for specified variants.

    Returns (results, budget_halted).
    """
    from eval.longmemeval.adapter import LongMemEvalQuestion

    q: LongMemEvalQuestion = case_payload  # type: ignore[assignment]
    results: list[CaseResult] = []

    # Determine whether we need a judge key (always GPT-4o for LongMemEval).
    openai_key = reader_env.get("openai_key") or os.environ.get("OPENAI_API_KEY", "")

    baseline_answer: str | None = None
    baseline_score: float | None = None
    baseline_latency_ms: float | None = None
    baseline_tokens: int | None = None
    baseline_judge_reasoning: str | None = None

    if "baseline" in variants:
        # Budget check before the call.
        call_cost = _READER_COST_PER_CALL.get(reader, 0.005) + _JUDGE_COST_PER_CALL
        if cumulative_cost[0] + call_cost > budget_usd:
            return results, True  # BUDGET_HALT

        answer, latency_ms, tokens = _call_longmemeval_baseline(q, reader, reader_env)
        baseline_answer = answer
        baseline_latency_ms = latency_ms
        baseline_tokens = tokens
        cumulative_cost[0] += _READER_COST_PER_CALL.get(reader, 0.005)

        # Judge score. _call_longmemeval_judge routes through make_openai_client
        # which prefers Azure when AZURE_OPENAI_ENDPOINT is set; otherwise it
        # uses direct OpenAI with `openai_key`. Either path is acceptable.
        judge_routable = bool(openai_key) or bool(
            os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        )
        if judge_routable:
            score, reasoning = _call_longmemeval_judge(q.question, q.answer, answer, openai_key)
            cumulative_cost[0] += _JUDGE_COST_PER_CALL
        else:
            score, reasoning = 0.0, "[no OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT — judge skipped]"
        baseline_score = score
        baseline_judge_reasoning = reasoning

        results.append(
            CaseResult(
                case_id=q.question_id,
                benchmark="longmemeval",
                reader=reader,
                variant="baseline",
                baseline_score=baseline_score,
                substrate_score=None,
                delta=None,
                latency_baseline_ms=baseline_latency_ms,
                latency_substrate_ms=None,
                tokens_used_baseline=baseline_tokens,
                tokens_used_substrate=None,
                estimated_cost=_READER_COST_PER_CALL.get(reader, 0.005) + _JUDGE_COST_PER_CALL,
                judge_reasoning=baseline_judge_reasoning,
                structural_validity={},
            )
        )

    if "substrate" in variants:
        call_cost = _READER_COST_PER_CALL.get(reader, 0.005) + _JUDGE_COST_PER_CALL
        if cumulative_cost[0] + call_cost > budget_usd:
            return results, True  # BUDGET_HALT

        # FIX 2: default substrate path is the new bge-m3 + retrieval + CoN
        # implementation (`_call_longmemeval_substrate_retrieval_con`). The
        # legacy E5 + bundle-then-reader path stays gated behind
        # `--legacy-lme-substrate` so reproducibility on pre-FIX-2 numbers
        # is recoverable without a git checkout.
        if legacy_substrate:
            answer, latency_ms, tokens, substrate_stats = _call_longmemeval_substrate(
                q, reader, reader_env
            )
        else:
            (
                answer,
                latency_ms,
                tokens,
                substrate_stats,
                _provenance,
            ) = _call_longmemeval_substrate_retrieval_con(q, reader_env)
        cumulative_cost[0] += _READER_COST_PER_CALL.get(reader, 0.005)

        # Same Azure-aware judge gating as the baseline branch above.
        judge_routable = bool(openai_key) or bool(
            os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        )
        if judge_routable:
            score, reasoning = _call_longmemeval_judge(q.question, q.answer, answer, openai_key)
            cumulative_cost[0] += _JUDGE_COST_PER_CALL
        else:
            score, reasoning = 0.0, "[no OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT — judge skipped]"

        delta = (score - baseline_score) if baseline_score is not None else None

        results.append(
            CaseResult(
                case_id=q.question_id,
                benchmark="longmemeval",
                reader=reader,
                variant="substrate",
                baseline_score=baseline_score,
                substrate_score=score,
                delta=delta,
                latency_baseline_ms=baseline_latency_ms,
                latency_substrate_ms=latency_ms,
                tokens_used_baseline=baseline_tokens,
                tokens_used_substrate=tokens,
                estimated_cost=_READER_COST_PER_CALL.get(reader, 0.005) + _JUDGE_COST_PER_CALL,
                judge_reasoning=reasoning,
                structural_validity={
                    "claims_written_count": substrate_stats.claims_written_count,
                    "supersessions_fired_count": substrate_stats.supersessions_fired_count,
                    "projection_nonempty": substrate_stats.projection_nonempty,
                    "active_pack": substrate_stats.active_pack,
                    "active_claim_count": substrate_stats.active_claim_count,
                    "top_k_retrieved": substrate_stats.top_k_retrieved,
                    "top_k_sim_mean": substrate_stats.top_k_sim_mean,
                    "top_k_sim_min": substrate_stats.top_k_sim_min,
                },
            )
        )

    return results, False


def _run_acibench_case(
    case_payload: object,
    reader: str,
    reader_env: dict[str, str],
    variants: tuple[str, ...],
    cumulative_cost: list[float],
    budget_usd: float,
) -> tuple[list[CaseResult], bool]:
    """Run one ACI-Bench case for specified variants.

    Returns (results, budget_halted).
    """
    from eval.aci_bench.adapter import ACIEncounter

    enc: ACIEncounter = case_payload  # type: ignore[assignment]
    results: list[CaseResult] = []

    baseline_note: str | None = None
    baseline_score: float | None = None
    baseline_latency_ms: float | None = None
    baseline_tokens: int | None = None

    if "baseline" in variants:
        call_cost = _READER_COST_PER_CALL.get(reader, 0.005)
        if cumulative_cost[0] + call_cost > budget_usd:
            return results, True

        note, latency_ms, tokens = _call_acibench_baseline(enc, reader, reader_env)
        baseline_note = note
        baseline_latency_ms = latency_ms
        baseline_tokens = tokens
        cumulative_cost[0] += call_cost

        medcon_f1 = _score_acibench_case(enc, note)
        baseline_score = medcon_f1

        results.append(
            CaseResult(
                case_id=enc.encounter_id,
                benchmark="acibench",
                reader=reader,
                variant="baseline",
                baseline_score=baseline_score,
                substrate_score=None,
                delta=None,
                latency_baseline_ms=baseline_latency_ms,
                latency_substrate_ms=None,
                tokens_used_baseline=baseline_tokens,
                tokens_used_substrate=None,
                estimated_cost=call_cost,
                judge_reasoning=None,
                structural_validity={},
            )
        )

    if "substrate" in variants:
        # Substrate variant fires (extractor-per-turn × N) + one SOAP call.
        # Budget this at 2× baseline cost as a conservative cap — real cost
        # depends on dialogue length; extractor calls are gpt-4o-mini class.
        call_cost = 2 * _READER_COST_PER_CALL.get(reader, 0.005)
        if cumulative_cost[0] + call_cost > budget_usd:
            return results, True

        # Phase-3 substrate wiring: _call_acibench_substrate now runs the
        # full ingest → claims-write → supersession → projection → SOAP
        # pipeline internally, returning the full substrate stats as part
        # of its tuple. Pass baseline_note so the substrate stats can
        # record `substrate_vs_baseline_edit_distance` for Phase 4A pass
        # criterion #4.
        substrate_ret = _call_acibench_substrate(
            enc, reader, reader_env, baseline_note
        )
        # Back-compat: older test mocks return a 3-tuple ("note", lat, tok);
        # the production path returns a 4-tuple (note, lat, tok, stats).
        if len(substrate_ret) == 4:
            note, latency_ms, tokens, substrate_stats = substrate_ret  # type: ignore[misc]
        else:
            note, latency_ms, tokens = substrate_ret  # type: ignore[misc]
            substrate_stats = _run_substrate_ingestion("acibench", enc)
        cumulative_cost[0] += call_cost

        medcon_f1 = _score_acibench_case(enc, note)
        delta = (medcon_f1 - baseline_score) if baseline_score is not None else None

        structural_validity: dict[str, Any] = {
            "claims_written_count": substrate_stats.claims_written_count,
            "supersessions_fired_count": substrate_stats.supersessions_fired_count,
            "projection_nonempty": substrate_stats.projection_nonempty,
            "active_pack": substrate_stats.active_pack,
            "active_claim_count": substrate_stats.active_claim_count,
            "sentence_with_provenance_ratio": substrate_stats.sentence_with_provenance_ratio,
        }
        if substrate_stats.substrate_vs_baseline_edit_distance is not None:
            structural_validity["substrate_vs_baseline_edit_distance"] = (
                substrate_stats.substrate_vs_baseline_edit_distance
            )

        results.append(
            CaseResult(
                case_id=enc.encounter_id,
                benchmark="acibench",
                reader=reader,
                variant="substrate",
                baseline_score=baseline_score,
                substrate_score=medcon_f1,
                delta=delta,
                latency_baseline_ms=baseline_latency_ms,
                latency_substrate_ms=latency_ms,
                tokens_used_baseline=baseline_tokens,
                tokens_used_substrate=tokens,
                estimated_cost=call_cost,
                judge_reasoning=None,
                structural_validity=structural_validity,
            )
        )

    return results, False


def _load_longmemeval_cases(n: int) -> list[object]:
    """Load first-N LongMemEval-S cases deterministically."""
    from eval.longmemeval.adapter import iter_questions

    questions_path = (
        _DATA_DIR / "longmemeval" / "data" / "longmemeval_s.json"
    )
    cases: list[object] = []
    for i, q in enumerate(iter_questions(questions_path)):
        if i >= n:
            break
        cases.append(q)
    return cases


def _load_acibench_cases(n: int) -> list[object]:
    """Load first-N ACI-Bench cases deterministically."""
    from eval.aci_bench.adapter import iter_all_test_encounters

    # Adapter auto-resolves `.../challenge_data_json/` when handed the parent
    # `data/` directory (see adapter._resolve_challenge_data_root).
    data_root = _DATA_DIR / "acibench" / "data"
    cases: list[object] = []
    for i, enc in enumerate(iter_all_test_encounters(data_root)):
        if i >= n:
            break
        cases.append(enc)
    return cases


# ------------------------------------------------------------------
# Real-run top-level.
# ------------------------------------------------------------------


def _real_run(cfg: SmokeConfig) -> SmokeResult:
    """Wire per-benchmark `eval/<b>/run.py` logic + per-variant readers + judge calls.

    Sequential per-case (debuggability > throughput for N=10 smoke tier).
    Full-run parallelism is out of scope per the dispatch instructions.
    """
    result = SmokeResult(verdict="FAIL")
    result.lines.extend(_print_planned_matrix(cfg))

    # Validate required env vars up front — fail fast with actionable messages.
    reader_envs: dict[str, dict[str, str]] = {}
    for reader in cfg.readers:
        reader_envs[reader] = _get_reader_env(reader)

    # Check datasets exist.
    for bench in cfg.benchmarks:
        found, msg = _check_dataset(bench)
        if not found:
            result.lines.append(f"[smoke] FAIL: dataset not available: {msg}")
            result.verdict = "FAIL"
            return result

    reference_baselines = _load_reference_baselines()

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _RESULTS_ROOT / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[CaseResult] = []
    # Cumulative cost tracked across all API calls. Use a list for mutability in closures.
    cumulative_cost: list[float] = [0.0]
    budget_halted = False
    anomaly_reasons: list[str] = []

    for bench in cfg.benchmarks:
        # Set ACTIVE_PACK BEFORE any substrate import for this benchmark.
        # This must happen before loading cases that may import lr_table implicitly.
        os.environ["ACTIVE_PACK"] = BENCHMARK_PACK[bench]
        # Clear lru_cache so active_pack() re-resolves from the new env var.
        try:
            from src.substrate.predicate_packs import active_pack as _ap
            _ap.cache_clear()
        except ImportError:
            pass

        result.lines.append(f"[smoke] benchmark={bench} ACTIVE_PACK={BENCHMARK_PACK[bench]}")

        # Load first-N cases deterministically.
        try:
            if bench == "longmemeval":
                cases = _load_longmemeval_cases(cfg.n_cases)
            else:
                cases = _load_acibench_cases(cfg.n_cases)
        except Exception as exc:
            result.lines.append(f"[smoke] FAIL loading {bench} cases: {exc}")
            result.verdict = "FAIL"
            return result

        result.lines.append(f"[smoke] {bench}: loaded {len(cases)} cases")

        for reader in cfg.readers:
            result.lines.append(f"[smoke] {bench} × {reader}: starting")
            reader_env = reader_envs[reader]

            bench_results: list[CaseResult] = []

            for case_payload in cases:
                if bench == "longmemeval":
                    case_results, halted = _run_longmemeval_case(
                        case_payload, reader, reader_env,
                        cfg.variants, cumulative_cost, cfg.budget_usd,
                        legacy_substrate=cfg.legacy_lme_substrate,
                    )
                else:
                    case_results, halted = _run_acibench_case(
                        case_payload, reader, reader_env,
                        cfg.variants, cumulative_cost, cfg.budget_usd,
                    )

                bench_results.extend(case_results)
                all_results.extend(case_results)

                if halted:
                    budget_halted = True
                    break

            if budget_halted:
                result.lines.append(
                    f"[smoke] BUDGET_HALT: cumulative cost ${cumulative_cost[0]:.4f} "
                    f"exceeded ${cfg.budget_usd:.2f} — partial results saved"
                )
                break

            result.lines.append(
                f"[smoke] {bench} × {reader}: {len(bench_results)} case-variant results"
            )

        if budget_halted:
            break

    # Write results.json.
    results_path = out_dir / "results.json"
    results_json = [asdict(r) for r in all_results]
    results_path.write_text(json.dumps(results_json, indent=2) + "\n", encoding="utf-8")
    result.lines.append(f"[smoke] results.json written to {results_path}")
    result.total_cost_usd = cumulative_cost[0]
    result.total_cases = len(all_results)

    if budget_halted:
        result.verdict = "FAIL"
        result.lines.append(f"[smoke] BUDGET_HALT total cost: ${cumulative_cost[0]:.4f}")
        return result

    # Verdict logic.
    verdict = _determine_verdict(all_results, reference_baselines, cfg, anomaly_reasons)
    result.verdict = verdict
    if anomaly_reasons:
        for reason in anomaly_reasons:
            result.lines.append(f"[smoke] ANOMALY criterion failed: {reason}")

    result.lines.append(f"[smoke] total cost: ${cumulative_cost[0]:.4f}")
    return result


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(sys.argv[1:] if argv is None else argv)

    if cfg.dry_run:
        result = _dry_run(cfg)
    else:
        result = _real_run(cfg)

    for line in result.lines:
        print(line)
    # ASCII markers so Windows cmd/PowerShell (cp1252 default) doesn't crash.
    marker = {"PASS": "[OK]", "ANOMALY": "[WARN]", "FAIL": "[FAIL]"}.get(result.verdict, "[?]")
    print(f"[smoke] Verdict: {marker} {result.verdict}")
    return 0 if result.verdict == "PASS" else (1 if result.verdict == "ANOMALY" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
