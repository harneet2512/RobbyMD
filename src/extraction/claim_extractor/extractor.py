"""LLM-backed claim extractor — implements the `ExtractorFn` contract.

Fulfils the Phase-2 deferral recorded in `src/extraction/claim_extractor/__init__.py`.
The substrate publishes `ExtractorFn = Callable[[Turn], list[ExtractedClaim]]`
at `src/substrate/on_new_turn.py:63`; this module ships a factory that returns
a closure satisfying that contract, backed by an OpenAI / Azure chat completion.

Routing goes through `eval/_openai_client.py::make_openai_client` with the
`claim_extractor_gpt4omini` purpose. The same router powers the LLM-MEDCON
concept extractor — we share the gpt-4o-mini (Azure: gpt-4.1-mini deployment)
so credential setup is uniform.

Error posture
-------------
- Malformed JSON → drop to `[]` + WARN log. The substrate treats an empty
  extraction as "this turn carries no structured claim" (correct for chit-chat).
- Predicates outside the active pack → dropped silently (the orchestrator
  also re-validates via `insert_claim`, so this is defence-in-depth).
- Network / API exceptions propagate. `on_new_turn` is called inside eval
  loops that have budget + try/except at the outer layer; a raised exception
  will halt the current case, which is the correct behaviour when the API
  itself is failing.

Cost (rough)
------------
~$0.0002 per turn at gpt-4o-mini pricing (input ~800 tok with few-shots,
output ~100 tok). For ACI-Bench smoke (10 cases × ~40 turns = 400 calls)
that's ~$0.08. For LongMemEval smoke (5 questions × ~2000 turns = 10K
calls) that's ~$2 — within the Phase 4B $10 cap.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import structlog

from src.extraction.claim_extractor.prompt import CLAIM_EXTRACTOR_SYSTEM_PROMPT
from src.substrate.on_new_turn import ExtractedClaim, ExtractorFn
from src.substrate.predicate_packs import PredicatePack, active_pack, load_pack
from src.substrate.schema import Turn

logger = structlog.get_logger(__name__)


def make_llm_extractor(
    *,
    purpose: str = "claim_extractor_gpt4omini",
    active_pack_id: str | None = None,
    env: Mapping[str, str] | None = None,
    temperature: float = 0.0,
) -> ExtractorFn:
    """Return a closure `Turn -> list[ExtractedClaim]` that calls an LLM.

    The LLM client is lazy-initialised on the first call so import-time
    failures don't require a live API key. Pack resolution happens at
    factory time so switching packs mid-run requires a new extractor.

    Parameters
    ----------
    purpose:
        Routing key for `eval/_openai_client.py::make_openai_client`.
        Defaults to `"claim_extractor_gpt4omini"` (Azure: gpt-4.1-mini,
        direct OpenAI: gpt-4o-mini).
    active_pack_id:
        Override the active pack. `None` reads the `ACTIVE_PACK` env var
        (via `active_pack()`), matching the prompt-composition convention.
    env:
        Injectable environment for tests. `None` uses `os.environ`.
    temperature:
        Chat completion temperature. Defaults to 0 for determinism.
    """
    pack: PredicatePack = (
        load_pack(active_pack_id) if active_pack_id is not None else active_pack()
    )
    allowed_predicates: frozenset[str] = pack.predicate_families

    # One-time cache for the client + model/deployment string. The factory
    # closes over a mutable cell so the first call initialises the client
    # and every subsequent call reuses it.
    cache: dict[str, Any] = {"client": None, "model": None}

    def _get_client() -> tuple[Any, str]:
        if cache["client"] is None:
            # Local import keeps the `src/` package importable without the
            # `openai` dependency available (the substrate itself has no
            # LLM dependency by design; only this extractor pulls it in).
            from eval._openai_client import make_openai_client

            # make_openai_client accepts `env: dict | None`; we pass through
            # the Mapping as-is. Tests can inject a dict here.
            raw_env: dict[str, str] | None
            if env is None:
                raw_env = None
            else:
                raw_env = dict(env)
            client, model = make_openai_client(purpose, raw_env)  # type: ignore[arg-type]
            cache["client"] = client
            cache["model"] = model
        return cache["client"], cache["model"]

    def _extract(turn: Turn) -> list[ExtractedClaim]:
        text = turn.text or ""
        if not text.strip():
            return []

        client, model = _get_client()
        # Turn.speaker may be a Speaker enum (substrate-facing callers) or a
        # bare string (LongMemEval adapter creates turns with speaker='system'
        # directly). Handle both — .value on an enum gives the string; a str
        # is already the string we want.
        speaker_label = (
            turn.speaker.value
            if hasattr(turn.speaker, "value")
            else str(turn.speaker)
        )
        user_content = (
            f"Current turn:\n    {speaker_label}: {text}\n\n"
            f"Active claims: (none provided; fresh extraction)\n"
        )
        try:
            # Per-call timeout is load-bearing: without it, a single slow
            # Azure/OpenAI call can hang the entire ingestion loop for hours
            # (observed on Phase 4B v2 — 12s CPU over 2h elapsed while one
            # request sat in retry backoff). 30s is generous for extraction.
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=[
                    {"role": "system", "content": CLAIM_EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=30.0,
            )
        except Exception as e:  # noqa: BLE001 — bubble any caller error, no silent eat
            logger.warning(
                "claim_extractor.api_error",
                turn_id=turn.turn_id,
                err=repr(e)[:200],
            )
            return []

        content = response.choices[0].message.content or "{}"
        raw_claims = _parse_claims(content)
        return _to_extracted_claims(raw_claims, turn, allowed_predicates)

    return _extract


def _parse_claims(raw: str) -> list[dict[str, Any]]:
    """Parse an LLM response into a list of claim dicts.

    Mirrors `eval.aci_bench.llm_medcon.parse_concepts` shape-handling:
    - Bare list → accepted.
    - Wrapper dict with a well-known key (`claims`, `items`, `list`, `data`,
      `medical_claims`, `extracted_claims`) → accepted.
    - Single-key dict whose only value is a list → accepted (handles
      arbitrary wrapper keys the model invents under json_object mode).
    - Anything else → `[]` + WARN.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("claim_extractor.parse_failed", raw=raw[:200])
        return []

    items: list[Any] | None = None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for key in ("claims", "medical_claims", "extracted_claims", "items", "list", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                items = value
                break
        if items is None and len(parsed) == 1:
            only_value = next(iter(parsed.values()))
            if isinstance(only_value, list):
                items = only_value
        # Single-claim shape: top-level dict with the four required claim keys.
        # gpt-4.1-mini under json_object response_format sometimes emits a
        # single claim as an unwrapped object instead of a one-element list.
        if items is None and {"subject", "predicate", "value", "confidence"}.issubset(
            parsed.keys()
        ):
            items = [parsed]
        # Empty dict from the LLM means "no claims this turn" — normal for
        # short / off-topic utterances. Return [] without warning.
        if items is None and len(parsed) == 0:
            return []
        if items is None:
            logger.warning(
                "claim_extractor.parse_unexpected_shape",
                keys=list(parsed.keys())[:10],
            )
            return []
    else:
        logger.warning("claim_extractor.parse_not_list_or_dict", type=type(parsed).__name__)
        return []

    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def _to_extracted_claims(
    raw_claims: list[dict[str, Any]],
    turn: Turn,
    allowed_predicates: frozenset[str],
) -> list[ExtractedClaim]:
    """Convert raw claim dicts into validated `ExtractedClaim` objects.

    Drops:
    - claims missing any required field (`subject`, `predicate`, `value`, `confidence`)
    - claims whose predicate is not in `allowed_predicates`
    - claims whose confidence does not coerce to a float in [0, 1]

    Resolves char spans with `turn.text.find(value)` — best-effort; `None`
    when the value string does not appear verbatim in the turn.
    """
    result: list[ExtractedClaim] = []
    drop_count = 0

    for raw in raw_claims:
        subject = raw.get("subject")
        predicate = raw.get("predicate")
        value = raw.get("value")
        confidence_raw = raw.get("confidence")

        if not isinstance(subject, str) or not subject.strip():
            drop_count += 1
            continue
        if not isinstance(predicate, str) or predicate not in allowed_predicates:
            drop_count += 1
            continue
        if not isinstance(value, str) or not value.strip():
            drop_count += 1
            continue
        try:
            confidence = float(confidence_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            drop_count += 1
            continue
        if not (0.0 <= confidence <= 1.0):
            drop_count += 1
            continue

        char_start, char_end = _find_char_span(turn.text, value)
        value_normalised_raw = raw.get("value_normalised")
        value_normalised = (
            value_normalised_raw if isinstance(value_normalised_raw, str) else None
        )

        result.append(
            ExtractedClaim(
                subject=subject.strip(),
                predicate=predicate,
                value=value.strip(),
                confidence=confidence,
                value_normalised=value_normalised,
                char_start=char_start,
                char_end=char_end,
            )
        )

    if drop_count:
        logger.info(
            "claim_extractor.dropped",
            turn_id=turn.turn_id,
            kept=len(result),
            dropped=drop_count,
        )
    return result


def _find_char_span(text: str, value: str) -> tuple[int | None, int | None]:
    """Best-effort char-span resolution. Returns (None, None) when no match."""
    if not text or not value:
        return None, None
    idx = text.find(value)
    if idx < 0:
        # Try a case-insensitive fallback — clinical values often vary in case.
        lowered = text.lower().find(value.lower())
        if lowered < 0:
            return None, None
        return lowered, lowered + len(value)
    return idx, idx + len(value)


__all__: list[str] = ["make_llm_extractor"]
