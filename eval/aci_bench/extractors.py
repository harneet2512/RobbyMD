"""3-tier `ConceptExtractor` implementations for MEDCON.

Per `docs/decisions/2026-04-21_medcon-tiered-fallback.md`:

- **T0 `QuickUMLSExtractor`** — official MEDCON; directly comparable to
  WangLab 2023 MEDIQA-CHAT Task B (57.78). Requires UMLS licence +
  QuickUMLS index built by `scripts/install_umls.sh`.
- **T1 `ScispacyExtractor`** — default; `en_core_sci_lg` + bundled UMLS
  linker. No UMLS licence needed. Labelled `MEDCON-approx` on slide.
- **T2 `NullExtractor`** — hard fallback; returns empty set. Triggers the
  "MEDCON omitted" path in `run.py`; supplementary metrics kick in.

Factory: `build_extractor()` picks based on `CONCEPT_EXTRACTOR` env var.
Defaults to `scispacy`. Env variable choices: `quickumls` | `scispacy` | `null`.

All three implementations return a `set[str]` of normalised CUIs.

Attribution note: scispaCy's `en_core_sci_lg` model bundles an AllenAI-derived
UMLS knowledge-base subset. Per the main-thread clarification on attribution
scope, this KB is NOT registered in `MODEL_ATTRIBUTIONS.md` because we load it
**only from `eval/`**, never from `src/`, and the licensing test
(`tests/licensing/test_model_attributions.py`) scans `src/` only. If the
substrate (`src/`) ever needs scispaCy UMLS linking, the row must be added
before that commit merges.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

# The 7 MEDCON semantic groups per Yim et al. / Abacha et al. MEDIQA-CHAT 2023.
# Restricted set of UMLS semantic types — prevents MEDCON from being dominated
# by ontology noise (locations, organisations, temporal concepts).
MEDCON_SEMANTIC_GROUPS: frozenset[str] = frozenset(
    {
        "Anatomy",          # ANAT
        "Chemicals & Drugs",  # CHEM
        "Devices",          # DEVI
        "Disorders",        # DISO
        "Living Beings",    # LIVB  (for certain pathogen CUIs)
        "Phenomena",        # PHEN
        "Procedures",       # PROC
    }
)


@runtime_checkable
class ConceptExtractor(Protocol):
    """Protocol each tier must satisfy.

    Contract:
    - `name`: machine-readable tier id (recorded in metrics.json).
    - `label`: human-facing label (goes on slide + LIMITATIONS.md header).
    - `semantic_groups`: the 7 MEDCON groups (or empty for NullExtractor).
    - `extract(text)`: returns a set of normalised CUIs. One call per note;
      F1 computed by `run.py` over intersection / union of two extractions.

    Implementations may store the three metadata attributes as ClassVar; the
    Protocol matches both instance-attr and ClassVar-attr shapes. Pyright
    strict flags the ClassVar mismatch at the factory return site, so we
    define the attributes here as ClassVar too.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    semantic_groups: ClassVar[frozenset[str]]

    def extract(self, text: str) -> set[str]: ...


# ── T0 ──────────────────────────────────────────────────────────────────────
@dataclass
class QuickUMLSExtractor:
    """Tier 0 — QuickUMLS over UMLS 2025AB Level 0 Subset.

    Directly comparable to WangLab 2023 MEDIQA-CHAT Task B (57.78). Requires
    UMLS licence. Lazy-initialises the matcher; `QUICKUMLS_PATH` must point
    to a built index (see `scripts/install_umls.sh`).
    """

    name: ClassVar[str] = "quickumls"
    label: ClassVar[str] = "MEDCON (official, QuickUMLS)"
    semantic_groups: ClassVar[frozenset[str]] = MEDCON_SEMANTIC_GROUPS

    quickumls_path: Path
    threshold: float = 0.8          # QuickUMLS default similarity cutoff
    _matcher: object | None = None  # lazy-initialised QuickUMLS matcher

    def _get_matcher(self) -> object:
        if self._matcher is None:
            try:
                from quickumls import QuickUMLS  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "quickumls not installed; run scripts/install_umls.sh "
                    "(see docs/decisions/2026-04-21_medcon-tiered-fallback.md)"
                ) from e
            self._matcher = QuickUMLS(
                str(self.quickumls_path),
                threshold=self.threshold,
                accepted_semtypes=None,  # we filter by semantic group post-hoc
            )
        return self._matcher

    def extract(self, text: str) -> set[str]:
        matcher = self._get_matcher()
        # QuickUMLS returns list[list[dict]] — outer list is spans, inner is
        # candidate CUIs per span. We take the best match per span.
        result = matcher.match(text, best_match=True, ignore_syntax=False)  # type: ignore[attr-defined]
        cuis: set[str] = set()
        for span_candidates in result:
            if not span_candidates:
                continue
            best = span_candidates[0]
            cui = best.get("cui") if isinstance(best, dict) else None
            if cui:
                cuis.add(str(cui))
        return cuis


# ── T1 (default) ────────────────────────────────────────────────────────────
@dataclass
class ScispacyExtractor:
    """Tier 1 — scispaCy `en_core_sci_lg` + bundled UMLS linker.

    Default. No UMLS licence required. Approximates MEDCON via the
    AllenAI-distributed UMLS KB subset (shipped with the model) and
    `scispacy.linking.EntityLinker`.
    """

    name: ClassVar[str] = "scispacy"
    label: ClassVar[str] = "MEDCON-approx (scispaCy UMLS linker)"
    semantic_groups: ClassVar[frozenset[str]] = MEDCON_SEMANTIC_GROUPS

    model_name: str = "en_core_sci_lg"
    linker_threshold: float = 0.75
    _nlp: object | None = None

    def _get_nlp(self) -> object:
        if self._nlp is None:
            try:
                # `scispacy` import registers the custom components/linker.
                import scispacy  # noqa: F401  # type: ignore[import-not-found]
                import spacy
                from scispacy.linking import (  # noqa: F401  # type: ignore[import-not-found]
                    EntityLinker,
                )
            except ImportError as e:
                raise RuntimeError(
                    "scispacy not installed; run scripts/install_scispacy.sh"
                ) from e
            nlp = spacy.load(self.model_name)
            if "scispacy_linker" not in nlp.pipe_names:
                nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "threshold": self.linker_threshold,
                    },
                )
            self._nlp = nlp
        return self._nlp

    def extract(self, text: str) -> set[str]:
        nlp = self._get_nlp()
        doc = nlp(text)  # type: ignore[operator]
        cuis: set[str] = set()
        for ent in doc.ents:  # type: ignore[attr-defined]
            for cui, score in getattr(ent._, "kb_ents", []):
                if score >= self.linker_threshold:
                    cuis.add(str(cui))
        return cuis


# ── T2 (hard fallback) ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class NullExtractor:
    """Tier 2 — no concept extraction.

    Returns empty set. `run.py` detects empty extractions and omits MEDCON
    from the report; section-level ROUGE + Bias Trap Rate (MedEinst) kick
    in as supplementary clinical-rigor proxies.
    """

    name: ClassVar[str] = "null"
    label: ClassVar[str] = "MEDCON omitted (fallback; no UMLS / no scispaCy)"
    semantic_groups: ClassVar[frozenset[str]] = frozenset()

    def extract(self, text: str) -> set[str]:
        return set()


# ── Factory ─────────────────────────────────────────────────────────────────
def build_extractor(env: dict[str, str] | None = None) -> ConceptExtractor:
    """Pick an extractor based on the `CONCEPT_EXTRACTOR` env var.

    Accepts an explicit dict (for tests) or reads `os.environ` by default.
    Unknown values fall back to `scispacy` with a warning; absent key → `scispacy`.
    """
    env = env or dict(os.environ)
    choice = env.get("CONCEPT_EXTRACTOR", "scispacy").strip().lower()

    if choice == "quickumls":
        path = env.get("QUICKUMLS_PATH", "").strip()
        if not path:
            raise RuntimeError(
                "CONCEPT_EXTRACTOR=quickumls but QUICKUMLS_PATH is unset. "
                "Run scripts/install_umls.sh and export QUICKUMLS_PATH."
            )
        return QuickUMLSExtractor(quickumls_path=Path(path))

    if choice == "null":
        return NullExtractor()

    if choice != "scispacy":
        print(
            f"[extractors] WARN unknown CONCEPT_EXTRACTOR={choice!r}; "
            "falling back to scispacy (T1 default)."
        )
    return ScispacyExtractor(
        model_name=env.get("SCISPACY_MODEL", "en_core_sci_lg"),
    )


def compute_medcon_f1(gold_cuis: set[str], pred_cuis: set[str]) -> dict[str, float]:
    """Compute MEDCON precision / recall / F1 given two CUI sets.

    Matches the set-intersection definition used by the official MEDCON
    script (exact CUI match, no fuzzy mapping). Returns zeros when either
    set is empty — NullExtractor triggers this and the slide-render path
    treats it as "omit column" (see run.py).
    """
    if not gold_cuis and not pred_cuis:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_gold": 0, "n_pred": 0}
    tp = len(gold_cuis & pred_cuis)
    precision = tp / len(pred_cuis) if pred_cuis else 0.0
    recall = tp / len(gold_cuis) if gold_cuis else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gold": len(gold_cuis),
        "n_pred": len(pred_cuis),
    }
