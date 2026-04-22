"""Admission filter — noise-regex only (no embedding novelty).

Per `docs/gt_v2_study_notes.md` §2.6 + `docs/research_brief.md` §4:

GT v2's embedding-novelty filter has a cold-start bug (empty index ⇒ 1.0
novelty ⇒ 100% admit; later ⇒ similar turns rejected) and every threshold
is unvalidated. For single-session chest-pain scope, controlled input,
embedding novelty is pure overhead and a source of flakiness. Drop it.

What remains: reject turns that are obviously non-content.

- fewer than 3 content words
- only filler tokens ("uh", "um", "mhm", "ok", "right", ...)
- silence / system markers (e.g. "[silence]", "[noise]")
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


_FILLER_WORDS: frozenset[str] = frozenset({
    "uh",
    "um",
    "er",
    "hmm",
    "mhm",
    "mm",
    "ah",
    "ok",
    "okay",
    "yeah",
    "yes",
    "no",
    "right",
    "well",
    "so",
    "like",
})

_SILENCE_MARKER = re.compile(r"^\s*\[(?:silence|noise|pause|inaudible|music)\]\s*$", re.I)
_WORD_RE = re.compile(r"[A-Za-z']+")

# Minimum number of non-filler tokens for admission. Keeps short clinically
# meaningful utterances in ("stabbing chest pain" = 3 words) while dropping
# fillers-only turns.
MIN_CONTENT_WORDS = 3


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    admitted: bool
    reason: str


def admit(text: str) -> AdmissionResult:
    """Decide whether a raw ASR turn is worth sending to extraction.

    Returns an `AdmissionResult`; the `reason` is always populated for
    telemetry regardless of outcome.
    """
    if not text or not text.strip():
        return AdmissionResult(False, "empty")
    if _SILENCE_MARKER.match(text):
        return AdmissionResult(False, "silence_marker")

    tokens = [w.lower() for w in _WORD_RE.findall(text)]
    if not tokens:
        return AdmissionResult(False, "no_word_tokens")
    content = [t for t in tokens if t not in _FILLER_WORDS]
    if len(content) < MIN_CONTENT_WORDS:
        return AdmissionResult(False, f"only_{len(content)}_content_words")
    return AdmissionResult(True, "admitted")
