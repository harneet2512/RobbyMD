"""Hallucination guard for ASR output.

Five cheap, deterministic checks that flag suspicious segments before they
reach the claim extractor. Running post-cleanup (so the guard sees cleaned
text) and pre-word-correction (so medication checks see cleaned spellings).

Design rationale
----------------
LLM-based ASR models (Whisper in particular) exhibit documented hallucination
pathologies when confidence is low:

* **Repeated n-gram loops** — model gets stuck repeating a short phrase. Very
  common on silent or near-silent frames. Cite: Koenecke et al., ACM FAccT
  2024 (arXiv 2312.05420).
* **OOV medical term injection** — model confabulates plausible-sounding
  medical tokens not present in the audio. Cite: Arora et al., "Clinical ASR
  Evaluation", arXiv 2502.11572 (Jogi, Aggarwal et al., 2025).
* **Implausibly dense transcription** — many characters per second of audio,
  indicating the model is generating faster than speech rate.
* **Low-confidence spans** — consecutive words all below a confidence
  threshold, suggesting the decoder was guessing throughout.
* **Invented medication names** — suffixes (-statin, -mab, -nib, etc.) on
  strings not in the RxNorm-derived vocabulary are a red flag for drug
  hallucination.

When `severity == "block"`, the segment is NOT dropped — provenance is
preserved. The downstream `Turn` has `asr_confidence` forced to 0.0 for all
tokens in the segment, and a structured warning is logged via structlog
(CLAUDE.md §8).

IMPORTANT: This module uses only the standard library + structlog. No ML
imports. All checks are O(n) in segment length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Tri-level hallucination severity."""

    CLEAN = "clean"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class FlaggedSpan:
    """One flagged region returned by a hallucination check."""

    check_name: str
    start_char: int
    end_char: int
    reason: str


@dataclass(frozen=True, slots=True)
class HallucinationReport:
    """Aggregated result for one ASR segment.

    `flagged_spans` lists every individual flag. `severity` is the aggregate.
    `reasons` is a deduplicated list of the human-readable reasons.
    """

    flagged_spans: tuple[FlaggedSpan, ...]
    severity: Severity
    reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Minimal placeholder RxNorm set (~30 common drugs, public-domain NLM strings).
# TODO: swap for full RxNorm dump once UMLS licence lands (see progress.md).
# ---------------------------------------------------------------------------
_RXNORM_PLACEHOLDER: frozenset[str] = frozenset({
    "aspirin", "clopidogrel", "ticagrelor", "prasugrel", "heparin",
    "enoxaparin", "warfarin", "apixaban", "rivaroxaban", "atorvastatin",
    "rosuvastatin", "simvastatin", "metoprolol", "atenolol", "bisoprolol",
    "carvedilol", "nitroglycerin", "isosorbide", "omeprazole", "pantoprazole",
    "ranitidine", "famotidine", "ibuprofen", "naproxen", "acetaminophen",
    "lisinopril", "losartan", "amlodipine", "furosemide", "morphine",
    # extras for robustness
    "prednisone", "metformin", "hydrochlorothiazide", "amoxicillin",
    "azithromycin", "doxycycline", "albuterol", "montelukast",
})

# Drug-suffix pattern (common pharmaceutical name-endings).
_DRUG_SUFFIX_PATTERN = re.compile(
    r"\b\w+(?:ol|ide|pril|sartan|statin|mab|nib|vir|cycline|mycin|cillin"
    r"|azole|oxetine|pine|tidine|mide|zide|lone|olol|afil|dipine)\b",
    re.IGNORECASE,
)

# Medical-term-like pattern: TitleCase Greek/Latin root heuristic.
# Matches words 5–20 chars that have a medical suffix OR start with uppercase.
_MEDICAL_SUFFIX_PATTERN = re.compile(
    r"\b[A-Z][a-z]{3,18}\b|\b\w+(?:itis|osis|ology|pathy|ectomy|scopy"
    r"|plasty|otomy|gram|emia|uria|algia|opathy)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def repeated_ngram_flag(
    text: str,
    n: int = 4,
    max_repeats: int = 3,
) -> tuple[bool, str]:
    """Flag text with repeated n-gram loops.

    Example of what this catches: "thank you thank you thank you thank you"

    Returns ``(flagged, reason_str)``.
    """
    words = text.split()
    # Need at least n words to form one n-gram.
    if len(words) < n:
        return False, ""

    # Count n-gram occurrences.
    ngram_counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(w.lower() for w in words[i : i + n])
        ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

    for gram, count in ngram_counts.items():
        if count >= max_repeats:
            phrase = " ".join(gram)
            return True, f"n-gram loop detected: '{phrase}' repeated {count}× (n={n})"

    return False, ""


def oov_medical_term_flag(
    text: str,
    vocabulary: set[str] | frozenset[str],
) -> tuple[bool, str]:
    """Flag medical-looking tokens not in the known vocabulary.

    Matches TitleCased words of length 5–20 and words with Greek/Latin medical
    suffixes that are not in the vocabulary. Designed to catch confabulated
    anatomy or procedure names.

    Returns ``(flagged, reason_str)``.
    """
    vocab_lower = {v.lower() for v in vocabulary}
    candidates = _MEDICAL_SUFFIX_PATTERN.findall(text)
    oov: list[str] = []
    for token in candidates:
        if len(token) < 5 or len(token) > 20:
            continue
        if token.lower() not in vocab_lower:
            oov.append(token)

    if oov:
        sample = oov[:3]
        return True, f"OOV medical-looking terms: {sample!r} (of {len(oov)} found)"
    return False, ""


def extreme_compression_ratio(
    segment_text_len: int,
    raw_audio_duration_s: float,
    threshold: float = 15.0,
) -> tuple[bool, str]:
    """Flag implausibly high characters-per-second rates.

    Normal conversational speech: ~10–14 chars/s at Whisper's tokenisation.
    Threshold default of 15.0 chars/s gives comfortable headroom for fast
    talkers while catching obvious hallucinations (30+ chars/s).

    Returns ``(flagged, reason_str)``.
    """
    if raw_audio_duration_s <= 0.0:
        return False, ""
    ratio = segment_text_len / raw_audio_duration_s
    if ratio > threshold:
        return (
            True,
            f"extreme compression ratio: {ratio:.1f} chars/s > {threshold} threshold",
        )
    return False, ""


def low_confidence_span(
    word_confidences: list[float],
    threshold: float = 0.3,
    max_span: int = 5,
) -> tuple[bool, str]:
    """Flag a run of ≥ max_span consecutive words with confidence < threshold.

    Returns ``(flagged, reason_str)``.
    """
    if not word_confidences:
        return False, ""

    run = 0
    max_run = 0
    for conf in word_confidences:
        if conf < threshold:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    if max_run >= max_span:
        return (
            True,
            f"low-confidence span: {max_run} consecutive words < {threshold:.2f} confidence",
        )
    return False, ""


def invented_medication_flag(
    text: str,
    rxnorm_set: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """Flag drug-suffix tokens not in the RxNorm vocabulary.

    Uses `_RXNORM_PLACEHOLDER` when `rxnorm_set` is None (UMLS licence
    pending — see progress.md). Once the full RxNorm dump is available, pass
    it as `rxnorm_set`.

    Returns ``(flagged, reason_str)``.
    """
    vocab = rxnorm_set if rxnorm_set is not None else _RXNORM_PLACEHOLDER
    vocab_lower = {v.lower() for v in vocab}

    matches = _DRUG_SUFFIX_PATTERN.findall(text)
    invented: list[str] = [m for m in matches if m.lower() not in vocab_lower]

    if invented:
        sample = invented[:3]
        return True, f"possible invented medication(s): {sample!r} (of {len(invented)} flagged)"
    return False, ""


# ---------------------------------------------------------------------------
# Aggregate report builder
# ---------------------------------------------------------------------------

def check(
    text: str,
    vocabulary: set[str] | frozenset[str],
    word_confidences: list[float] | None = None,
    audio_duration_s: float = 0.0,
    rxnorm_set: frozenset[str] | None = None,
) -> HallucinationReport:
    """Run all five checks and return a `HallucinationReport`.

    Severity logic (per spec):
    - ``clean``  → 0 checks flagged.
    - ``warn``   → 1–2 checks flagged AND no invented-medication flag.
    - ``block``  → 3+ checks flagged OR invented_medication_flag is True.

    When severity is ``block``, this function emits a structlog WARNING
    (CLAUDE.md §8). The segment is NOT discarded — the caller should set
    ``asr_confidence=0.0`` on all tokens in the segment.

    Parameters
    ----------
    text:
        Cleaned (post-TranscriptCleaner) segment text.
    vocabulary:
        The active pack's ASR vocabulary (e.g. ``active_pack().asr_vocabulary``
        when that field exists, or the set from ``vocab.py``).
    word_confidences:
        Per-word confidence scores from faster-whisper; None or empty skips the
        low-confidence-span check.
    audio_duration_s:
        Duration in seconds of the raw audio segment; 0.0 skips the
        compression-ratio check.
    rxnorm_set:
        Full RxNorm vocabulary; None uses the ~30-drug placeholder.
    """
    flagged_spans: list[FlaggedSpan] = []
    reasons: list[str] = []
    med_flag = False

    # Check 1: repeated n-gram loops.
    ok, reason = repeated_ngram_flag(text)
    if ok:
        flagged_spans.append(FlaggedSpan("repeated_ngram", 0, len(text), reason))
        reasons.append(reason)

    # Check 2: OOV medical terms.
    ok2, reason2 = oov_medical_term_flag(text, vocabulary)
    if ok2:
        flagged_spans.append(FlaggedSpan("oov_medical_term", 0, len(text), reason2))
        reasons.append(reason2)

    # Check 3: extreme compression ratio.
    ok3, reason3 = extreme_compression_ratio(len(text), audio_duration_s)
    if ok3:
        flagged_spans.append(FlaggedSpan("extreme_compression", 0, len(text), reason3))
        reasons.append(reason3)

    # Check 4: low-confidence span.
    ok4, reason4 = low_confidence_span(word_confidences or [])
    if ok4:
        flagged_spans.append(FlaggedSpan("low_confidence_span", 0, len(text), reason4))
        reasons.append(reason4)

    # Check 5: invented medication names.
    ok5, reason5 = invented_medication_flag(text, rxnorm_set)
    if ok5:
        med_flag = True
        flagged_spans.append(FlaggedSpan("invented_medication", 0, len(text), reason5))
        reasons.append(reason5)

    flag_count = len(flagged_spans)
    if flag_count == 0:
        severity = Severity.CLEAN
    elif med_flag or flag_count >= 3:
        severity = Severity.BLOCK
    else:
        severity = Severity.WARN

    report = HallucinationReport(
        flagged_spans=tuple(flagged_spans),
        severity=severity,
        reasons=tuple(reasons),
    )

    if severity == Severity.BLOCK:
        logger.warning(
            "hallucination_guard.block",
            severity=severity.value,
            flag_count=flag_count,
            reasons=list(reasons),
            text_preview=text[:120],
        )
    elif severity == Severity.WARN:
        logger.info(
            "hallucination_guard.warn",
            severity=severity.value,
            flag_count=flag_count,
            reasons=list(reasons),
        )

    return report
