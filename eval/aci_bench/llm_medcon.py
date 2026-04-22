"""LLM-MEDCON — gpt-4o-mini concept extraction for ACI-Bench note-level scoring.

Implements the fourth tier of the MEDCON metric pipeline (T0 QuickUMLS,
T1 scispaCy, T2 null, **T-LLM gpt-4o-mini**). The UMLS Metathesaurus
licence did not land in the hackathon window; LLM-MEDCON is the metric
we actually score against for the ACI-Bench leaderboard row.

Contract
--------
A `LLMMedconExtractor` instance satisfies `ConceptExtractor` (see
`extractors.py`). Calling `extract(text)` returns a `set[str]` of
normalised concept strings (lowercased, whitespace-stripped). Wire via

    CONCEPT_EXTRACTOR=llm_medcon python eval/aci_bench/run.py --variant full

`compute_medcon_f1(gold_set, pred_set)` from `extractors.py` produces the
set-based precision / recall / F1 without any further changes.

Cost
----
~$0.001 per note pair (input ~500 tok + output ~100 tok at gpt-4o-mini
pricing). A full 90-encounter ACI-Bench run costs ≲ $0.20 end-to-end,
well inside the `--budget-usd 50` smoke cap.

Semantic scope
--------------
The system prompt restricts concept extraction to the same UMLS semantic
groups as the original MEDCON (Anatomy, Chemicals & Drugs, Devices,
Disorders, Genes & Molecular Sequences, Phenomena, Physiology). Results
are therefore comparable in spirit to QuickUMLS-based MEDCON; the
difference is that LLM-MEDCON handles synonyms and paraphrases that
string matching misses.

Attribution
-----------
gpt-4o-mini is a commercial OpenAI model and is permitted in eval loops
by `rules.md §2.2` (eval-time readers allow closed-weight models as long
as they are not used on the demo path). See `methodology.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

import structlog

logger = structlog.get_logger(__name__)


# Exact prompt from the approved Track B plan (vast-humming-elephant.md).
# Intentionally imperative; `json_object` response_format + the explicit
# "ONLY a JSON list" directive keeps gpt-4o-mini from wrapping the output
# in prose.
LLM_MEDCON_SYSTEM_PROMPT = (
    "Extract all medical concepts from the following clinical note. "
    "Return ONLY a JSON list of normalized concept strings. "
    "Include: diagnoses, symptoms, medications, procedures, anatomical "
    "locations, lab values, vital signs. "
    "Restrict to: Anatomy, Chemicals & Drugs, Devices, Disorders, "
    "Genes & Molecular Sequences, Phenomena and Physiology."
)


# The seven semantic groups the system prompt restricts to. Matches the
# MEDCON_SEMANTIC_GROUPS constant in extractors.py, re-stated here so the
# LLM-MEDCON module is self-documenting.
LLM_MEDCON_SEMANTIC_GROUPS: frozenset[str] = frozenset(
    {
        "Anatomy",
        "Chemicals & Drugs",
        "Devices",
        "Disorders",
        "Genes & Molecular Sequences",
        "Phenomena",
        "Physiology",
    }
)


@dataclass
class LLMMedconExtractor:
    """Tier-LLM — gpt-4o-mini concept extractor for ACI-Bench MEDCON.

    Lazy-initialises the OpenAI client so `import eval.aci_bench.llm_medcon`
    never requires a live API key. Set `OPENAI_API_KEY` before calling
    `extract()`.
    """

    name: ClassVar[str] = "llm_medcon"
    label: ClassVar[str] = "LLM-MEDCON (gpt-4o-mini concept extraction)"
    semantic_groups: ClassVar[frozenset[str]] = LLM_MEDCON_SEMANTIC_GROUPS

    # Model / deployment selection happens inside `make_openai_client`. When
    # AZURE_OPENAI_ENDPOINT is set, the call targets
    # AZURE_OPENAI_GPT4OMINI_DEPLOYMENT; otherwise it falls back to direct
    # OpenAI `gpt-4o-mini`.
    temperature: float = 0.0
    _client: Any = field(default=None, repr=False, compare=False)
    _model: str = field(default="", repr=False, compare=False)

    def _get_client(self) -> tuple[Any, str]:
        if self._client is None:
            from eval._openai_client import make_openai_client

            self._client, self._model = make_openai_client("llm_medcon_gpt4omini")
        return self._client, self._model

    def extract(self, text: str) -> set[str]:
        """Call gpt-4o-mini once; return a normalised concept-string set.

        Empty input short-circuits to `set()` without touching the API so
        pairwise runs can safely pass a missing section through.
        """
        if not text.strip():
            return set()
        client, model = self._get_client()
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": LLM_MEDCON_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return parse_concepts(content)


def parse_concepts(raw: str) -> set[str]:
    """Parse a gpt-4o-mini response into a normalised concept set.

    Handles three response shapes:
      - Bare JSON list:               `["x", "y", ...]`
      - Wrapper object with list key: `{"concepts": ["x", ...]}` (or
        "items" / "list" / "data" — gpt-4o-mini chooses one of these
        when the `response_format: json_object` coercion kicks in and
        the model needs an object).
      - Anything else → `set()` + a WARN log. We do **not** raise, so
        one malformed response can't halt a 90-encounter run.

    Each surviving string is lowercased and whitespace-stripped to make
    set-based F1 stable across minor formatting jitter.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("llm_medcon.parse_failed", raw=raw[:200])
        return set()

    items: list[Any] | None = None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        # Prefer the well-known wrapper keys first for determinism.
        for key in ("concepts", "medical_concepts", "items", "list", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                items = value
                break
        # Lenient fallback: if the dict has exactly one value and it's a list,
        # accept it. Handles arbitrary wrapper keys gpt-4o-mini invents under
        # json_object response_format (e.g. `{"results": [...]}`).
        if items is None and len(parsed) == 1:
            only_value = next(iter(parsed.values()))
            if isinstance(only_value, list):
                items = only_value
        if items is None:
            logger.warning(
                "llm_medcon.parse_unexpected_shape",
                keys=list(parsed.keys())[:10],
            )
            return set()
    else:
        return set()

    out: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        normalised = item.strip().lower()
        if normalised:
            out.add(normalised)
    return out
