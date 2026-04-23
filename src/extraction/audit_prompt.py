"""Audit prompt + LLM call for the audit-and-revise variant (Worker 5).

This is the LEAST externally-grounded part of the proposed architecture
(see `advisory/validation/architecture_validation.md` §3 Claim E). The
closest published clinical pattern is the **npj Digital Medicine 2025
SOAP-audit framework** described in `research_report.md` §1, which audits
LLM-generated clinical notes against extracted structured facts. We adapt
that pattern here: given a draft SOAP note + the substrate's active claim
set, identify hallucinations (sentences without claim support), omissions
(claims absent from the draft), and value mismatches (sentences that
contradict a claim).

Treat this module as an empirical bet. The single-call hybrid SOAP
generator currently sits at PARITY (Δ −0.003 to −0.008 LLM-MEDCON on
n=10). We are NOT asserting this lifts that — we are providing the
building blocks so the orchestrator can run a smoke A/B between hybrid
and audit-revise. If the smoke shows the variant within ±0.020 of hybrid
PARITY, abandon it cleanly per Claim E.

Cost (rough)
------------
One gpt-4o-mini call per audit pass. Inputs scale with draft length +
active-claim count (~1.5-3K tokens), outputs ~300-800 tokens of
JSON. ~$0.0005 per audit at gpt-4o-mini pricing. For ACI-Bench smoke
(10 cases) that's ~$0.005 — negligible against the Phase 4B cap.

Routing goes through `eval/_openai_client.py::make_openai_client` with
the `claim_extractor_gpt4omini` purpose. Same client/deployment as the
claim extractor — no new model purpose required.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from src.substrate.schema import Claim

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------- prompt ------


AUDIT_SYSTEM_PROMPT = """\
You are a clinical-note auditor. You are given:

1. A draft SOAP note for a doctor-patient encounter.
2. A list of structured CLAIMS extracted from the source dialogue. Each
   claim has an id and (subject, predicate, value) fields. The claim set
   is the canonical record of what the patient and physician actually said.

Your task: produce a JSON audit report with three lists.

- "hallucinations": sentences in the draft that assert clinical content NOT
  supported by ANY claim. A sentence repeating boilerplate ("No orders
  placed.") with no clinical assertion is NOT a hallucination — only flag
  sentences that introduce facts (symptoms, findings, history,
  medications, diagnoses, plan items) absent from the claim set.

- "omissions": claims that are NOT mentioned (asserted, paraphrased, or
  logically entailed) by ANY sentence in the draft. Report the claim_id
  string only — not the full triple.

- "mismatches": draft sentences that CONTRADICT a claim (e.g., draft says
  "no chest pain" while a claim asserts severity=7/10 chest pain; or draft
  says "onset 2 days" while a claim asserts onset=5 days). Each mismatch
  carries the offending sentence text, the contradicted claim_id, and a
  short reason.

Rules
-----
- A sentence supports a claim if it asserts, paraphrases, or logically
  entails the claim's (subject, predicate, value) triple. Be generous —
  do not flag paraphrase as omission.
- Output JSON ONLY, no commentary, no markdown. Schema:
  {
    "hallucinations": ["sentence text", ...],
    "omissions": ["cl_xxx", ...],
    "mismatches": [
      {"sentence": "...", "claim_id": "cl_xxx", "reason": "..."},
      ...
    ]
  }
- If there is nothing to flag in a category, emit an empty list for that
  key. Do not omit keys.
- Use only claim_ids that appear in the input claim list. Do not invent
  ids.

This is a research prototype, not a medical device; the physician makes
every clinical decision.
"""


# --------------------------------------------------------------- dataclass ---


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Structured output of the draft-vs-claims audit.

    Attributes
    ----------
    hallucinations:
        Draft sentences asserting facts not supported by any active claim.
    omissions:
        claim_ids of active claims absent from the draft.
    mismatches:
        Draft sentences that contradict an active claim. Each entry is a
        dict with keys ``sentence``, ``claim_id``, ``reason`` (all str).
    """

    hallucinations: list[str] = field(default_factory=list)
    omissions: list[str] = field(default_factory=list)
    mismatches: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialise to a stable JSON string for logging / artifact storage."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> AuditReport:
        """Round-trip from `to_json` output. Tolerant of missing keys."""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"AuditReport.from_json expected dict, got {type(data).__name__}")
        hallucinations = _coerce_str_list(data.get("hallucinations", []))
        omissions = _coerce_str_list(data.get("omissions", []))
        mismatches_raw = data.get("mismatches", [])
        mismatches: list[dict[str, str]] = []
        if isinstance(mismatches_raw, list):
            for m in mismatches_raw:
                if isinstance(m, dict):
                    mismatches.append(
                        {
                            "sentence": str(m.get("sentence", "")),
                            "claim_id": str(m.get("claim_id", "")),
                            "reason": str(m.get("reason", "")),
                        }
                    )
        return cls(
            hallucinations=hallucinations,
            omissions=omissions,
            mismatches=mismatches,
        )


def _coerce_str_list(value: Any) -> list[str]:
    """Best-effort conversion of an LLM JSON value into a list of strings."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


# --------------------------------------------------------------- formatter ---


def format_claims_for_audit(claims: list[Claim]) -> str:
    """Render claims as a compact JSON array for the audit prompt.

    Uses one line per claim so the model can scan them quickly. Only the
    fields the auditor needs are included.
    """
    lines: list[str] = []
    for c in claims:
        lines.append(
            json.dumps(
                {
                    "claim_id": c.claim_id,
                    "subject": c.subject,
                    "predicate": c.predicate,
                    "value": c.value,
                },
                ensure_ascii=False,
            )
        )
    return "[\n  " + ",\n  ".join(lines) + "\n]" if lines else "[]"


# --------------------------------------------------------------- API ---------


def audit_draft_soap(
    *,
    draft_soap: str,
    active_claims: list[Claim],
    purpose: str = "claim_extractor_gpt4omini",
    env: Mapping[str, str] | None = None,
    temperature: float = 0.0,
    client: Any = None,
    model: str | None = None,
) -> AuditReport:
    """Compare a draft SOAP note against the substrate's active claim set.

    Returns an `AuditReport` listing potential hallucinations (draft
    sentences without claim support), omissions (claims absent from the
    draft), and value mismatches (draft sentences contradicting a claim).

    Closest external pattern: the npj Digital Medicine 2025 SOAP-audit
    framework (see `research_report.md` §1). No published clinical paper
    reports MEDCON lift from a draft-then-audit-then-revise loop on
    ACI-Bench specifically — this is an empirical bet (see
    `advisory/validation/architecture_validation.md` §3 Claim E).

    Parameters
    ----------
    draft_soap:
        The draft SOAP note text (full multi-section string).
    active_claims:
        The output of `list_active_claims(conn, session_id)`. Empty list
        is allowed; the audit will return an all-empty report (nothing
        to ground against → nothing to flag).
    purpose:
        Routing key for `eval/_openai_client.py::make_openai_client`.
        Defaults to `"claim_extractor_gpt4omini"` (Azure: gpt-4.1-mini,
        direct OpenAI: gpt-4o-mini). We do NOT introduce a new purpose
        for the audit pass — same model class as claim extraction.
    env:
        Injectable environment for tests. `None` uses `os.environ`.
    temperature:
        Chat completion temperature. Defaults to 0 for determinism — the
        audit must be reproducible across runs.
    client:
        Optional pre-built OpenAI/Azure client (bypasses make_openai_client).
        `model` must also be supplied if `client` is supplied.
    model:
        Optional model / deployment string paired with `client`.

    Raises
    ------
    RuntimeError
        Propagated from `make_openai_client` when no API credentials are
        configured. Matches the gold-leak guard pattern — no silent
        return of the unaudited draft.
    """
    if not draft_soap or not draft_soap.strip():
        return AuditReport()

    if client is None:
        # Local import keeps the substrate package importable without the
        # `openai` dependency available at module load.
        from eval._openai_client import make_openai_client

        raw_env: dict[str, str] | None
        if env is None:
            raw_env = None
        else:
            raw_env = dict(env)
        client, model = make_openai_client(purpose, raw_env)  # type: ignore[arg-type]
    elif model is None:
        raise ValueError("model must be provided when client is provided")

    claims_block = format_claims_for_audit(active_claims)
    user_content = (
        "DRAFT SOAP NOTE\n"
        "---------------\n"
        f"{draft_soap}\n\n"
        "ACTIVE CLAIMS\n"
        "-------------\n"
        f"{claims_block}\n"
    )

    try:
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            timeout=60.0,
        )
    except RuntimeError:
        # RuntimeError from make_openai_client (missing creds) bubbles up
        # unchanged — caller policy is no silent fallback to the unaudited
        # draft (matches Worker 1's gold-leak guard pattern).
        raise
    except Exception as exc:  # noqa: BLE001 — log + return empty audit
        logger.warning(
            "audit_draft.api_error",
            err=repr(exc)[:200],
            n_claims=len(active_claims),
        )
        return AuditReport()

    raw = response.choices[0].message.content or "{}"
    return _parse_audit_report(raw)


def _parse_audit_report(raw: str) -> AuditReport:
    """Parse the LLM JSON response into an AuditReport. Tolerant of shape drift.

    Accepted shapes:
      - Top-level dict with keys ``hallucinations``, ``omissions``, ``mismatches``.
      - Top-level dict wrapping an ``audit`` key whose value follows the same shape.
    Anything else returns an empty report + a WARN log.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("audit_draft.parse_failed", raw=raw[:200])
        return AuditReport()

    if isinstance(parsed, dict) and "audit" in parsed and isinstance(parsed["audit"], dict):
        parsed = parsed["audit"]

    if not isinstance(parsed, dict):
        logger.warning(
            "audit_draft.parse_unexpected_shape",
            type=type(parsed).__name__,
        )
        return AuditReport()

    return AuditReport.from_json(json.dumps(parsed))


__all__: list[str] = [
    "AUDIT_SYSTEM_PROMPT",
    "AuditReport",
    "audit_draft_soap",
    "format_claims_for_audit",
]
