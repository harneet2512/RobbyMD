"""Red-flag generation and semantic symptom matching.

Red flags merge two sources at encounter close:
1. Verifier's final discriminative features (dynamic, encounter-specific)
2. Predicate pack's static red-flag list (universal for complaint category)

Symptom matching uses e5-small-v2 embeddings (already in the stack via
supersession_semantic.py). No gpt-4o-mini — rules.md disallows commercial
APIs other than Opus 4.7.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COSINE_THRESHOLD = 0.65


@dataclass(frozen=True, slots=True)
class RedFlag:
    symptom: str
    source: str  # "verifier" | "pack"
    hypothesis: str | None


@dataclass(frozen=True, slots=True)
class SymptomMatch:
    patient_symptom: str
    matched_flag: str | None
    similarity: float
    is_match: bool


def load_static_red_flags(complaint: str = "chest_pain") -> list[RedFlag]:
    path = _REPO_ROOT / "predicate_packs" / "clinical_general" / "red_flags.json"
    if not path.exists():
        log.warning("red_flags.json not found", path=str(path))
        return []

    data: dict[str, Any] = json.loads(path.read_text())
    flags_list: list[str] = data.get(complaint, [])
    return [RedFlag(symptom=s, source="pack", hypothesis=None) for s in flags_list]


def generate_red_flag_list(
    verifier_missing_features: list[dict[str, str]] | None = None,
    complaint: str = "chest_pain",
) -> tuple[str, ...]:
    """Merge verifier-derived + pack-derived red flags, deduplicate."""
    static_flags = load_static_red_flags(complaint)

    dynamic_flags: list[RedFlag] = []
    if verifier_missing_features:
        for f in verifier_missing_features:
            dynamic_flags.append(
                RedFlag(
                    symptom=f.get("description", f.get("feature", "")),
                    source="verifier",
                    hypothesis=f.get("hypothesis"),
                )
            )

    all_flags = dynamic_flags + static_flags

    seen: set[str] = set()
    unique: list[str] = []
    for flag in all_flags:
        key = flag.symptom.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(flag.symptom)

    return tuple(unique)


def check_symptoms_against_flags(
    patient_symptoms: list[str],
    red_flag_list: tuple[str, ...],
    threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> list[SymptomMatch]:
    """Match patient-reported symptoms against red flags using e5-small-v2."""
    if not patient_symptoms or not red_flag_list:
        return []

    try:
        embedder = _get_embedder()
    except (ImportError, OSError):
        log.warning(
            "sentence_transformers not available, falling back to keyword matching"
        )
        return _keyword_fallback(patient_symptoms, red_flag_list)

    flag_texts = [f"query: {f}" for f in red_flag_list]
    symptom_texts = [f"query: {s}" for s in patient_symptoms]

    flag_vecs = embedder.embed(flag_texts)
    symptom_vecs = embedder.embed(symptom_texts)

    results: list[SymptomMatch] = []
    for i, symptom in enumerate(patient_symptoms):
        best_sim = 0.0
        best_flag: str | None = None
        for j, flag in enumerate(red_flag_list):
            sim = _cosine(symptom_vecs[i], flag_vecs[j])
            if sim > best_sim:
                best_sim = sim
                best_flag = flag

        results.append(
            SymptomMatch(
                patient_symptom=symptom,
                matched_flag=best_flag if best_sim >= threshold else None,
                similarity=round(best_sim, 4),
                is_match=best_sim >= threshold,
            )
        )

    return results


def _keyword_fallback(
    symptoms: list[str],
    flags: tuple[str, ...],
) -> list[SymptomMatch]:
    """Simple keyword overlap fallback when embeddings unavailable."""
    results: list[SymptomMatch] = []
    for symptom in symptoms:
        s_words = set(symptom.lower().split())
        best_overlap = 0.0
        best_flag: str | None = None
        for flag in flags:
            f_words = set(flag.lower().split())
            overlap = len(s_words & f_words) / max(len(s_words | f_words), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_flag = flag
        results.append(
            SymptomMatch(
                patient_symptom=symptom,
                matched_flag=best_flag if best_overlap >= 0.25 else None,
                similarity=round(best_overlap, 4),
                is_match=best_overlap >= 0.25,
            )
        )
    return results


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_embedder: object | None = None


def _get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        from src.substrate.supersession_semantic import E5Embedder

        _embedder = E5Embedder()
    return _embedder
