"""SOAP generator — substrate-backed, provenance-tagged.

Flow
----
1. Query `list_active_claims(conn, session_id)` — the real substrate read
   API, not a free-form LLM call.
2. Group claims by SOAP section using the active pack's
   `soap_mapping.json`. Packs without a mapping file fall back to
   putting every claim in Subjective.
3. Ask the reader LLM to compose a SOAP note using ONLY the provided
   claims + dialogue. Instruct it to tag each sentence with
   `[c:<claim_id>]` markers for every claim that sentence derives from.
4. Parse the `[c:…]` tags out of the returned text, validate each
   claim_id against the set of active claims (drop unknown IDs — the
   LLM hallucinated), and emit a `SOAPResult` with the stripped note
   plus a tuple of `(sentence, claim_ids)` pairs.

The generator routes LLM calls through `eval/_openai_client.py` when
the reader is an Azure-hosted / OpenAI model; callers can also inject a
client directly for tests.

Cost
----
One chat completion per SOAP generation. Inputs scale with dialogue +
active-claim count (~2-5K tokens), outputs ~500-1000 tokens. At
gpt-4.1-mini pricing this is ~$0.005 per note — under the Phase 4A
$5 cap for 10 cases with comfortable headroom.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.substrate.claims import list_active_claims
from src.substrate.predicate_packs import active_pack
from src.substrate.schema import Claim

logger = structlog.get_logger(__name__)

# Exact regex used to strip/parse provenance tags. The shape `[c:abc123]` is
# tight enough not to collide with ordinary clinical markdown (we produce
# plain text notes, no bracketed references expected).
_CLAIM_TAG_RE = re.compile(r"\[c:([a-zA-Z0-9_\-]+)\]")
_SECTION_HEADING_RE = re.compile(
    r"^\s*(subjective|objective|assessment|plan)\s*:?\s*$",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# End-to-end SOAP drafting prompt — identical in spirit to the baseline
# variant, no claim-bundle constraint. Per P0 from architecture_recommendations.md:
# the substrate's role on summarization is audit (post-hoc provenance
# annotation), not generation. The Phase 4A smoke showed that constraining
# generation to claims cost 21pp of MEDCON recall.
DRAFT_PROMPT = """\
You are a clinical documentation assistant producing a comprehensive SOAP
note from a doctor-patient conversation.

Rules
-----
- Produce four sections in this exact order:
  SUBJECTIVE:
  OBJECTIVE:
  ASSESSMENT:
  PLAN:
- Include every clinically relevant detail present in the dialogue:
  symptoms, onset, severity, location, radiation, aggravating/alleviating
  factors, associated symptoms, duration, medical history, medications,
  allergies, family history, social history, review of systems, vitals,
  exam findings, labs, imaging, assessment, differential, plan items.
- Do not invent findings, diagnoses, orders, or test results that are
  not supported by the dialogue.
- If a section has no supported content, write one short sentence
  noting that (e.g., "No orders placed during this visit.").
- Every output sentence ends with a period. No bullet lists. No markdown.
- Do not mention this instruction set in the output.

This is a research prototype, not a medical device; the physician makes
every clinical decision.
"""


# Post-hoc provenance annotation prompt. Runs AFTER the draft. Input is
# the finished note + the list of active claims. Task: for each sentence,
# identify which claim IDs ground that sentence's content. The LLM is
# doing grounding, not generation — a much easier and more accurate task.
ANNOTATION_PROMPT = """\
You are a provenance-annotation assistant. You are given:

1. A finished clinical SOAP note (one sentence per line after normalisation).
2. A list of structured claims extracted from the source dialogue. Each
   claim has an ID and (subject, predicate, value) fields.

Your task: for EACH sentence of the note, identify which claim IDs (if
any) are supported by that sentence's content.

- A claim supports a sentence if the sentence asserts, paraphrases, or
  logically entails the claim's (subject, predicate, value) triple.
- Multiple claims can support one sentence (e.g., a sentence about chest
  pain at 7/10 radiating to the arm is supported by both the
  severity claim and the radiation claim).
- A sentence that restates information not captured by any claim → empty
  list. Do NOT invent claim IDs.

Output JSON, one entry per input sentence in order:
```
{"annotations": [
  {"sentence_index": 0, "claim_ids": ["c12", "c15"]},
  {"sentence_index": 1, "claim_ids": []},
  ...
]}
```

No commentary. No markdown outside the JSON. Only use claim IDs that
appear in the input claim list.
"""


@dataclass(frozen=True, slots=True)
class SOAPResult:
    """Output of `generate_soap_note`.

    Attributes
    ----------
    note_text:
        The clean SOAP note with `[c:…]` tags stripped. This is what the
        ACI-Bench evaluator will score against the reference note.
    sentence_provenance:
        Tuple of `(sentence_text, claim_ids)` in output order. Sentences
        without any valid claim tags appear with an empty `claim_ids`
        tuple — we keep them for honest reporting.
    active_claim_count:
        Number of active claims passed to the LLM. Zero means the
        substrate produced nothing this session (probably a wiring or
        extractor issue; the generator still returns a sensible
        all-empty result).
    tokens_used:
        Prompt + completion tokens returned by the LLM.
    latency_ms:
        Wall-clock duration of the LLM call.
    """

    note_text: str
    sentence_provenance: tuple[tuple[str, tuple[str, ...]], ...]
    active_claim_count: int
    tokens_used: int
    latency_ms: float

    @property
    def sentence_with_provenance_ratio(self) -> float:
        """Fraction of emitted sentences that carry at least one claim tag.

        Caller-facing metric — a low value signals the LLM is ignoring
        the provenance instruction.
        """
        if not self.sentence_provenance:
            return 0.0
        with_prov = sum(1 for _, ids in self.sentence_provenance if ids)
        return with_prov / len(self.sentence_provenance)


def generate_soap_note(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    dialogue_text: str,
    reader: str,
    reader_env: Mapping[str, str],
    client: Any = None,
) -> SOAPResult:
    """Generate a SOAP note end-to-end, then annotate provenance post-hoc.

    Phase 4A showed that constraining generation to the claim bundle cost
    21pp of MEDCON recall. P0 architecture: the claim graph serves as an
    audit layer applied ON TOP OF an end-to-end draft, not as a funnel
    through which the SOAP must be composed.

    Flow:
      1. `generate_soap_draft`: end-to-end SOAP from dialogue only.
         Equivalent to the baseline's generation path.
      2. `annotate_provenance`: one additional LLM call mapping each
         sentence in the draft to claim IDs that support it. If claim
         extraction produced nothing (active_claims empty), this step
         is skipped — the draft is returned untouched.

    Returns a `SOAPResult` whose `note_text` is the clean end-to-end draft
    and whose `sentence_provenance` carries the post-hoc claim groundings.
    """
    import time

    active: list[Claim] = list_active_claims(conn, session_id)
    llm_client, model = _resolve_client(reader, reader_env, client=client)

    t_start = time.monotonic()

    # Step 1: end-to-end draft.
    draft_text, draft_tokens = _generate_soap_draft(
        llm_client, model, dialogue_text, session_id
    )
    if not draft_text:
        return SOAPResult(
            note_text="",
            sentence_provenance=(),
            active_claim_count=len(active),
            tokens_used=draft_tokens,
            latency_ms=(time.monotonic() - t_start) * 1000,
        )

    # Step 2: post-hoc provenance annotation.
    sentences = _split_into_sentences(draft_text)
    if active and sentences:
        annotations, annot_tokens = _annotate_provenance(
            llm_client, model, sentences, active, session_id
        )
    else:
        annotations, annot_tokens = [()] * len(sentences), 0

    sentence_provenance = tuple(zip(sentences, annotations, strict=False))

    return SOAPResult(
        note_text=draft_text,
        sentence_provenance=sentence_provenance,
        active_claim_count=len(active),
        tokens_used=draft_tokens + annot_tokens,
        latency_ms=(time.monotonic() - t_start) * 1000,
    )


def _generate_soap_draft(
    llm_client: Any,
    model: str,
    dialogue_text: str,
    session_id: str,
) -> tuple[str, int]:
    """End-to-end SOAP generation. No claim bundle. Returns (text, tokens)."""
    try:
        response = llm_client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": DRAFT_PROMPT},
                {"role": "user", "content": dialogue_text},
            ],
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001 — callers handle; no silent success
        logger.warning(
            "soap_generator.draft_api_error",
            session_id=session_id,
            err=repr(e)[:200],
        )
        return "", 0
    raw = response.choices[0].message.content or ""
    return raw.strip(), _extract_token_count(response)


def _annotate_provenance(
    llm_client: Any,
    model: str,
    sentences: list[str],
    active_claims: list[Claim],
    session_id: str,
) -> tuple[list[tuple[str, ...]], int]:
    """Post-hoc: for each sentence, return the tuple of claim IDs grounding it.

    Returns `(list_per_sentence_of_claim_id_tuples, tokens_used)`. On error
    or malformed response, returns per-sentence empty tuples + 0 tokens so
    the caller still gets a well-formed SOAPResult.
    """
    known_ids = {c.claim_id for c in active_claims}
    claim_lines = [
        f"- {c.claim_id}: subject=`{c.subject}` predicate=`{c.predicate}` value=`{c.value}`"
        for c in active_claims
    ]
    sentence_lines = [f"{i}: {s}" for i, s in enumerate(sentences)]

    user_content = (
        "# Draft SOAP note sentences (indexed)\n\n"
        + "\n".join(sentence_lines)
        + "\n\n# Available claims\n\n"
        + "\n".join(claim_lines)
    )

    try:
        response = llm_client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": ANNOTATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "soap_generator.annotate_api_error",
            session_id=session_id,
            err=repr(e)[:200],
        )
        return [()] * len(sentences), 0

    raw = response.choices[0].message.content or "{}"
    tokens = _extract_token_count(response)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("soap_generator.annotate_parse_failed", raw=raw[:200])
        return [()] * len(sentences), tokens

    entries = parsed.get("annotations") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return [()] * len(sentences), tokens

    # Build per-index result, default empty tuple.
    per_index: dict[int, tuple[str, ...]] = {i: () for i in range(len(sentences))}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("sentence_index")
        ids = entry.get("claim_ids")
        if not isinstance(idx, int) or not isinstance(ids, list):
            continue
        if idx < 0 or idx >= len(sentences):
            continue
        valid_ids = tuple(i for i in ids if isinstance(i, str) and i in known_ids)
        per_index[idx] = valid_ids
    return [per_index[i] for i in range(len(sentences))], tokens


def _split_into_sentences(note_text: str) -> list[str]:
    """Split a SOAP note into per-sentence strings, preserving section headings.

    Section headings (e.g., "SUBJECTIVE:") are preserved as their own
    sentences so annotators see them but produce empty claim lists for
    them. The output order matches top-to-bottom reading order.
    """
    sentences: list[str] = []
    for line in note_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION_HEADING_RE.match(stripped):
            sentences.append(stripped)
            continue
        for s in _iter_sentences(stripped):
            if s:
                sentences.append(s)
    return sentences


def _group_claims_by_section(
    active: list[Claim],
) -> dict[str, list[Claim]]:
    """Bucket claims into S / O / A / P using the active pack's mapping."""
    mapping, default_section = _load_soap_mapping()
    buckets: dict[str, list[Claim]] = {"S": [], "O": [], "A": [], "P": []}
    for claim in active:
        section = mapping.get(claim.predicate, default_section)
        buckets.setdefault(section, []).append(claim)
    return buckets


def _load_soap_mapping() -> tuple[dict[str, str], str]:
    """Load `predicate_packs/<active_pack>/soap_mapping.json` if present.

    Returns `({}, "S")` for packs without a mapping file (e.g.
    `personal_assistant`) — callers fall back to putting all claims in
    Subjective.
    """
    pack = active_pack()
    mapping_path = _REPO_ROOT / "predicate_packs" / pack.pack_id / "soap_mapping.json"
    if not mapping_path.is_file():
        return {}, "S"
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping = raw.get("predicate_to_section") or {}
    if not isinstance(mapping, dict):
        return {}, "S"
    default_section = str(raw.get("default_section", "S"))
    # Cast: json may load unicode strings; normalise to str → str mapping.
    return {str(k): str(v) for k, v in mapping.items()}, default_section


def _build_prompt(
    sections: dict[str, list[Claim]],
    dialogue_text: str,
) -> str:
    """Format dialogue (primary) + claim bundle (secondary coverage hints).

    Order: dialogue first, claim bundle second — the SOAP prompt treats the
    dialogue as the source of truth and the bundle as a coverage checklist.
    Reversing this order caused the LLM to over-restrict to the bundle in
    the Phase 4A smoke (substrate F1 21pp below baseline).
    """
    lines: list[str] = ["# Dialogue transcript (primary source of truth)\n"]
    lines.append(dialogue_text)
    lines.append("\n\n# Pre-extracted claims (coverage checklist — ensure these appear)\n")
    lines.append(
        "Each claim below has an ID you can cite via `[c:<id>]` when a "
        "sentence in your SOAP note derives from it. The dialogue is richer "
        "than this bundle — write a complete note from the dialogue; these "
        "are anchors, not a boundary."
    )
    for section in ("S", "O", "A", "P"):
        claims = sections.get(section, [])
        lines.append(f"\n## Section {section} anchors")
        if not claims:
            lines.append("_(no bundle claims — rely on dialogue)_")
            continue
        for c in claims:
            lines.append(
                f"- [c:{c.claim_id}] subject=`{c.subject}` "
                f"predicate=`{c.predicate}` value=`{c.value}` "
                f"confidence={c.confidence:.2f}"
            )
    return "\n".join(lines)


def _parse_note_with_provenance(
    raw_text: str,
    active_claims: list[Claim],
) -> tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    """Strip `[c:…]` tags from the note text and extract sentence provenance.

    Returns `(clean_note_text, sentence_provenance_tuple)` where each
    element of the tuple is `(sentence_without_tags, claim_ids)`.
    """
    known_ids: set[str] = {c.claim_id for c in active_claims}
    clean_lines: list[str] = []
    sentence_prov: list[tuple[str, tuple[str, ...]]] = []

    for line_raw in raw_text.splitlines():
        line = line_raw.rstrip()
        if not line.strip():
            clean_lines.append(line)
            continue
        if _SECTION_HEADING_RE.match(line):
            clean_lines.append(line)
            continue
        # Split each non-heading line into sentences so multi-sentence
        # paragraphs still get per-sentence provenance.
        for sentence in _iter_sentences(line):
            ids_in_sentence = tuple(
                m.group(1)
                for m in _CLAIM_TAG_RE.finditer(sentence)
                if m.group(1) in known_ids
            )
            clean_sentence = _CLAIM_TAG_RE.sub("", sentence).strip()
            # Collapse double spaces that tag stripping leaves behind.
            clean_sentence = re.sub(r"\s{2,}", " ", clean_sentence)
            if clean_sentence:
                sentence_prov.append((clean_sentence, ids_in_sentence))
        stripped_line = _CLAIM_TAG_RE.sub("", line)
        stripped_line = re.sub(r"\s{2,}", " ", stripped_line).strip()
        if stripped_line:
            clean_lines.append(stripped_line)

    clean_note = "\n".join(clean_lines).strip()
    return clean_note, tuple(sentence_prov)


def _iter_sentences(line: str) -> list[str]:
    """Naive sentence splitter — period/?/! followed by space + capital.

    Good enough for SOAP notes which are short declarative sentences.
    """
    parts = _SENTENCE_SPLIT_RE.split(line)
    return [p.strip() for p in parts if p.strip()]


def _resolve_client(
    reader: str,
    reader_env: Mapping[str, str],
    *,
    client: Any = None,
) -> tuple[Any, str]:
    """Locate a chat-completions-capable client for `reader`.

    Test code can inject `client` directly and skip env resolution.
    Production uses the shared Azure/OpenAI router.
    """
    if client is not None:
        # When tests inject a client, they also pass the model/deployment
        # string in via `reader_env["model_override"]` or fall back to
        # `reader` verbatim.
        model = reader_env.get("model_override") or reader
        return client, model

    # Piggyback on the judge deployment (gpt-4.1 / gpt-4o scale model) so
    # we don't multiply Azure deployments unnecessarily. Pass `env=None` so
    # make_openai_client reads os.environ — reader_env is a harness-local
    # dict (openai_key, azure_routed flag) that does NOT include the Azure
    # endpoint / deployment vars.
    from eval._openai_client import make_openai_client

    return make_openai_client("judge_gpt4o", None)


def _extract_token_count(response: Any) -> int:
    """Best-effort token count from OpenAI/Azure chat response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return int(prompt_tokens) + int(completion_tokens)
