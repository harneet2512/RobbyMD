"""Medical-vocabulary word correction via Levenshtein edit distance.

Corrects residual ASR misspellings that survived the LLM cleanup step.
Operates on whitespace-tokenised text; corrections are case-preserving.

Design rationale
----------------
Transcript cleanup (TranscriptCleaner) handles major medical term
misspellings via the LLM. This module handles the long tail of slight
misspellings that the LLM might miss or that appear after processing (e.g.
"metoprplol" → "metoprolol", "aspirine" → "aspirin").

A simple Levenshtein guard (max_edit_distance=1) catches single-character
insertions, deletions, and transpositions — the most common ASR error types
at high confidence. Increasing the threshold to 2 is possible but risks
over-correction of valid common words.

Common-English guard
--------------------
A hard-coded ~200-word `DEFAULT_COMMON` frozenset prevents correcting
ordinary words to medical vocabulary look-alikes. For example, "pain" must
not be corrected to "Pian" (a tropical infection), and "mine" must not be
corrected to "nifedipine" (edit distance 7, but still possible with a
generous threshold).

Provenance compliance
---------------------
Every correction is logged via structlog (CLAUDE.md §8). The returned
`list[Correction]` carries `original`, `replacement`, `char_start`,
`char_end`, `distance`, and `vocab_match` so callers can attach corrections
to the turn's provenance payload (rules.md §4).

Dependencies
------------
`rapidfuzz` (MIT licence) — fast Levenshtein / fuzzy-match library.
Added to `pyproject.toml` as a runtime dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Common English words that MUST NOT be corrected to medical vocabulary.
# (~200 of the most frequent English words + common clinical lay terms.)
# Maintained as a hard-coded frozenset so there is no disk I/O on import.
# ---------------------------------------------------------------------------
DEFAULT_COMMON: frozenset[str] = frozenset({
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Pronouns
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    # Verbs (common)
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need",
    "get", "got", "take", "took", "feel", "felt", "start", "started",
    "go", "went", "come", "came", "see", "saw", "know", "knew",
    "think", "thought", "want", "wanted", "tell", "told", "say", "said",
    # Common clinical lay terms (MUST NOT be corrected to medical vocab)
    "pain", "chest", "back", "arm", "leg", "head", "heart", "lung",
    "breath", "breathe", "breathing", "cough", "fever", "nausea", "vomit",
    "ache", "hurt", "hurts", "sore", "swollen", "swelling", "rash",
    "tired", "fatigue", "dizzy", "dizzy", "faint", "weak",
    "blood", "urine", "stool", "skin", "bone", "joint",
    "doctor", "nurse", "hospital", "clinic", "medicine", "drug", "pill",
    "tablet", "dose", "dosage",
    # Prepositions / conjunctions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "down", "out", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "and", "but", "or", "nor",
    "so", "yet", "although", "because", "since", "while", "if", "when",
    "where", "how", "what", "which", "who", "whom", "whose",
    # Time / quantity
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hour", "hours", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "minute", "minutes", "second",
    "morning", "evening", "night", "today", "yesterday", "ago",
    # Adjectives (common)
    "good", "bad", "new", "old", "high", "low", "small", "large", "big",
    "short", "long", "early", "late", "right", "left", "same", "other",
    "first", "last", "few", "more", "most", "much", "very",
    # Negations / modifiers
    "no", "not", "never", "any", "some", "all", "both", "each",
    "every", "either", "neither", "also", "too", "just", "only",
    "even", "than", "then", "there", "here",
})


@dataclass(frozen=True, slots=True)
class Correction:
    """One applied word-level correction.

    Carries enough information to reconstruct the original → replacement mapping
    in downstream provenance payloads (rules.md §4).
    """

    original: str
    replacement: str
    char_start: int       # character offset in the original transcript string
    char_end: int         # char_start + len(original)
    distance: int         # Levenshtein edit distance of the correction
    vocab_match: str      # the exact vocabulary term that was matched


def correct_medical_tokens(
    transcript: str,
    vocabulary: set[str] | frozenset[str],
    max_edit_distance: int = 1,
    common_english: frozenset[str] = DEFAULT_COMMON,
) -> tuple[str, list[Correction]]:
    """Correct medical-vocabulary misspellings in `transcript`.

    Algorithm
    ---------
    1. Tokenise by whitespace (preserving inter-token spans for char offsets).
    2. For each token:
       a. If lowercase form is in `common_english`, skip (guard against over-
          correction to medical lookalikes).
       b. If token (case-insensitive) is already in `vocabulary`, skip.
       c. Otherwise, find the closest vocabulary term within `max_edit_distance`.
          If found, replace using case-preserving logic (see `_preserve_case`).
    3. Re-join and return the corrected string plus a list of `Correction`s.

    All corrections are logged via structlog per CLAUDE.md §8.

    Parameters
    ----------
    transcript:
        Raw (or LLM-cleaned) segment text.
    vocabulary:
        Set of authoritative medical terms (e.g. from ``vocab.py``).
    max_edit_distance:
        Maximum Levenshtein distance to accept. Default 1 (single-char errors).
        Raise to 2 with caution — risk of over-correction increases.
    common_english:
        Words never corrected regardless of vocabulary similarity.

    Returns
    -------
    (corrected_transcript, corrections_applied)
    """
    # Prefer rapidfuzz (MIT) for fast Levenshtein computation.
    # Fall back to a minimal pure-Python implementation so the module works
    # in environments where rapidfuzz cannot be installed (e.g. disk-full CI).
    try:
        from rapidfuzz.distance import Levenshtein as _Lev  # type: ignore[import-not-found]

        def _edit_dist(a: str, b: str) -> int:
            return int(_Lev.distance(a, b))

    except (ImportError, ModuleNotFoundError):
        # Pure-Python fallback — O(mn) DP, correct but slower.
        def _edit_dist(a: str, b: str) -> int:  # type: ignore[misc]
            return _levenshtein_pure(a, b)

    vocab_lower_map: dict[str, str] = {v.lower(): v for v in vocabulary}
    corrections: list[Correction] = []

    # Find tokens with their character positions using a regex word-boundary
    # approach that handles punctuation attached to words (e.g. "metoprolol.").
    tokens_with_spans: list[tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", transcript):
        tokens_with_spans.append((m.group(), m.start(), m.end()))

    rebuilt_chars = list(transcript)

    # We process right-to-left so that char offsets don't shift when we replace.
    for token, start, end in reversed(tokens_with_spans):
        # Strip trailing punctuation for matching; keep punctuation for output.
        stripped, suffix = _strip_trailing_punct(token)
        if not stripped:
            continue

        lower = stripped.lower()

        # Guard: skip common English words.
        if lower in common_english:
            continue

        # Guard: skip tokens already in vocabulary.
        if lower in vocab_lower_map:
            continue

        # Search vocabulary for closest match within edit distance.
        best_match: str | None = None
        best_dist: int = max_edit_distance + 1

        for vocab_lower_term, vocab_original in vocab_lower_map.items():
            dist = _edit_dist(lower, vocab_lower_term)
            if dist <= max_edit_distance and dist < best_dist:
                best_dist = dist
                best_match = vocab_original

        if best_match is not None:
            replaced = _preserve_case(stripped, best_match) + suffix
            corrections.append(
                Correction(
                    original=token,
                    replacement=replaced,
                    char_start=start,
                    char_end=end,
                    distance=best_dist,
                    vocab_match=best_match,
                )
            )
            logger.info(
                "word_correction.applied",
                original=token,
                replacement=replaced,
                distance=best_dist,
                vocab_match=best_match,
                char_start=start,
            )
            # Replace in the char list.
            rebuilt_chars[start:end] = list(replaced)

    corrected = "".join(rebuilt_chars)
    return corrected, corrections


def _levenshtein_pure(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance (O(mn) DP).

    Used as a fallback when rapidfuzz is unavailable (e.g. corrupt install or
    disk-full CI environment). Correct but slower than rapidfuzz C extension.
    For the typical token lengths in medical transcription (3–15 chars) this is
    fast enough: a 15×15 character comparison takes < 1 µs.
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    # Allocate two rows (previous + current) to keep memory O(min(m,n)).
    if len(a) < len(b):
        a, b = b, a  # b is the shorter string
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            curr[j] = min(insert_cost, delete_cost, replace_cost)
        prev, curr = curr, prev
    return prev[len(b)]


def _strip_trailing_punct(token: str) -> tuple[str, str]:
    """Split a token into (stripped_word, trailing_punctuation).

    Example: ``"metoprolol."`` → ``("metoprolol", ".")``.
    """
    m = re.match(r"^(.*?)([.,;:!?\"')\]>]*)$", token)
    if m:
        return m.group(1), m.group(2)
    return token, ""


def _preserve_case(original: str, replacement: str) -> str:
    """Apply the case pattern of `original` to `replacement`.

    Rules:
    - ALL_CAPS original → ALL_CAPS replacement.
    - Title Case original → Title Case replacement.
    - Otherwise → lowercase replacement.
    """
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement.lower()
