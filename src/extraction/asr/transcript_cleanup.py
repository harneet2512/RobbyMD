"""Two-speaker medical transcript cleanup via cheap LLM.

Pattern inspired by:
- FreeFlow (zachlatta/freeflow, MIT): per-speaker system prompt + filler-word
  removal + custom dictionary injection. We extend their single-speaker
  dictation pattern to two-speaker medical dialogue with role-aware prompts.
  https://github.com/zachlatta/freeflow
- OpenWhispr (nicholasgcoles/openwhispr, MIT): dictionary architecture for
  medical-term biasing. We adopt the numbered-vocabulary-list injection pattern.
  https://openwhispr.com / https://github.com/nicholasgcoles/openwhispr
- Voquill (josiahsrc/voquill, AGPLv3): glossary injection pattern. We extend
  the glossary concept to two-speaker role-aware prompts. Voquill is AGPLv3;
  we do NOT copy code — we extend the concept in original code written during
  the hackathon per rules.md §1.1.
  https://github.com/josiahsrc/voquill

Key extensions beyond single-speaker dictation
----------------------------------------------
1. **Speaker-role-aware prompts**: Doctor and Patient differ in vocabulary,
   correction strategy, and output contract. An unknown speaker gets a neutral
   prompt (filler + punctuation only, no medical normalisation).
2. **Conversation context**: up to `max_context` previous cleaned turns are
   injected as numbered context so the model can resolve coreferences
   ("it" = the pain mentioned 2 turns ago).
3. **Lay-language tagging**: patient segments tag lay terms with
   ``[likely: <medical-term>]`` rather than silently replacing them —
   preserving the patient's exact vocabulary while flagging likely referents
   for the claim extractor (rules.md §4 provenance).
4. **Provenance invariant**: `original_text` is ALWAYS preserved on the
   returned `CleanedSegment`, regardless of what the LLM returns. The claim
   extractor runs on `cleaned_text`; provenance traces back to `original_text`.

NEVER use Opus 4.7 here (Eng_doc.md §3.5 — cleanup is throughput, not demo-
path). If `cleanup_model.startswith("claude-opus")`, raise ValueError at
construction (config.py enforces this invariant).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Exception for missing API key
# ---------------------------------------------------------------------------


class CleanupUnavailable(RuntimeError):
    """Raised when the cleanup LLM cannot be reached.

    Typically: OPENAI_API_KEY not set, or qwen2.5-7b-local endpoint offline.
    Callers MUST handle this by either skipping cleanup (returning the raw
    segment) or surfacing the error to the operator.
    """


# ---------------------------------------------------------------------------
# System prompts (FreeFlow-pattern, adapted for two-speaker medical use)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_DOCTOR = """\
You are a medical transcription editor processing a physician's speech.

Your task:
1. Remove filler words and sounds (um, uh, er, ah, you know, like, right, okay so, hmm).
2. Fix spelling errors and grammatical mistakes.
3. Add correct punctuation.
4. Correct medical misspellings using the vocabulary list provided.
   Example: "met oh pro lol" → "metoprolol", "aspirine" → "aspirin".
5. Resolve backtracking and self-correction.
   Example: "The pain — actually, the discomfort started" → "The discomfort started".
6. Preserve clinical intent EXACTLY. Do not rephrase, summarise, or infer.
7. Do NOT infer a diagnosis or suggest clinical actions — you are an editor, not a clinician.

Output ONLY the cleaned text, or the literal word EMPTY if the input is empty \
or contains only filler.
Do not include any preamble, explanation, or markdown formatting.
"""

SYSTEM_PROMPT_PATIENT = """\
You are a medical transcription editor processing a patient's speech.

Your task:
1. Remove filler words and sounds (um, uh, er, ah, you know, like, right, okay so, hmm).
2. Fix obvious spelling errors.
3. Add correct punctuation.
4. Resolve backtracking and self-correction.
   Example: "It hurts — well, more like a pressure — in my chest" → \
"It feels like pressure in my chest".
5. Preserve the patient's exact complaint language. Do NOT replace lay terms with \
medical terms.
   Example: "feels like someone's sitting on my chest" MUST stay verbatim.
6. When you recognise a lay phrase that likely refers to a medical concept, TAG it:
   [likely: <medical-term>]
   Example: "feels like someone's sitting on my chest [likely: pressure-type chest pain]"
   Example: "my heart was going crazy [likely: palpitations]"
   Keep the original phrase AND add the tag — do not remove the patient's words.
7. Do NOT diagnose or suggest diagnoses.

Output ONLY the cleaned text (with any [likely: ...] tags), or the literal word \
EMPTY if the input is empty or contains only filler.
Do not include any preamble, explanation, or markdown formatting.
"""

SYSTEM_PROMPT_UNKNOWN = """\
You are a medical transcription editor processing an unknown speaker's speech.

Your task:
1. Remove filler words and sounds (um, uh, er, ah, you know, like, right, hmm).
2. Add correct punctuation.
3. Do NOT apply medical normalisation — speaker role unknown.

Output ONLY the cleaned text, or the literal word EMPTY if the input is empty \
or contains only filler.
Do not include any preamble, explanation, or markdown formatting.
"""

_ROLE_TO_PROMPT: dict[str, str] = {
    "doctor": SYSTEM_PROMPT_DOCTOR,
    "physician": SYSTEM_PROMPT_DOCTOR,
    "patient": SYSTEM_PROMPT_PATIENT,
    "unknown": SYSTEM_PROMPT_UNKNOWN,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

REASON_CATEGORIES = frozenset({
    "filler_removal",
    "backtracking_resolution",
    "medical_term_correction",
    "punctuation",
    "grammar",
    "lay_language_tag",
    "other",
})


@dataclass(frozen=True, slots=True)
class DiarisedSegment:
    """One ASR segment with speaker label, before cleanup."""

    speaker_role: str   # "doctor", "patient", or "unknown"
    raw_text: str
    t_start: float = 0.0
    t_end: float = 0.0


@dataclass(frozen=True, slots=True)
class CleanupCorrection:
    """One correction applied during LLM cleanup."""

    original_span: str
    replacement: str
    reason_category: str  # must be in REASON_CATEGORIES


@dataclass(frozen=True, slots=True)
class CleanedSegment:
    """A diarised segment after LLM cleanup.

    `original_text` is ALWAYS preserved — the claim extractor runs on
    `cleaned_text`, but provenance traces back to `original_text`
    (rules.md §4).
    """

    speaker_role: str
    cleaned_text: str
    original_text: str                              # ALWAYS kept
    corrections_applied: tuple[CleanupCorrection, ...]
    confidence: float | None = None                 # None if logprobs unavailable
    t_start: float = 0.0
    t_end: float = 0.0


# ---------------------------------------------------------------------------
# TranscriptCleaner
# ---------------------------------------------------------------------------


class TranscriptCleaner:
    """Clean diarised ASR segments using a cheap LLM.

    Per Part B of the ASR hardening spec:
    - Uses speaker-role-aware system prompts.
    - Injects the medical vocabulary as a numbered list (OpenWhispr pattern).
    - Injects the last N cleaned turns as conversation context.
    - Diffs raw vs cleaned to produce per-correction attribution.

    Parameters
    ----------
    medical_vocabulary:
        Set of authoritative medical terms (from vocab.py or active pack).
    conversation_context:
        Pre-seeded list of cleaned turn strings (mutable rolling buffer).
    cleanup_model:
        Model ID to use for cleanup. Must NOT start with "claude-opus".
    max_context:
        Maximum number of previous cleaned turns to inject.
    """

    def __init__(
        self,
        medical_vocabulary: set[str],
        conversation_context: list[str] | None = None,
        cleanup_model: str = "gpt-4o-mini",
        max_context: int = 5,
    ) -> None:
        _guard_no_opus(cleanup_model)
        self._vocab = medical_vocabulary
        self._context: list[str] = list(conversation_context or [])
        self._model = cleanup_model
        self._max_context = max_context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, segment: DiarisedSegment) -> CleanedSegment:
        """Clean one diarised segment and update the conversation context.

        Provenance invariant: ``CleanedSegment.original_text`` is always set to
        ``segment.raw_text``, even if cleanup fails.
        """
        system_prompt = _ROLE_TO_PROMPT.get(
            segment.speaker_role.lower(), SYSTEM_PROMPT_UNKNOWN
        )
        user_message = self._build_user_message(segment.raw_text)

        try:
            cleaned_text = self._call_llm(system_prompt, user_message)
        except CleanupUnavailable:
            # Propagate; caller decides whether to skip cleanup or abort.
            raise
        except Exception as exc:
            logger.warning(
                "transcript_cleanup.llm_error",
                model=self._model,
                error=str(exc),
                speaker_role=segment.speaker_role,
            )
            # Fallback: return raw text with no corrections.
            cleaned_text = segment.raw_text

        if cleaned_text == "EMPTY":
            cleaned_text = ""

        corrections = _diff_corrections(segment.raw_text, cleaned_text)

        logger.info(
            "transcript_cleanup.done",
            model=self._model,
            speaker_role=segment.speaker_role,
            corrections=len(corrections),
            raw_len=len(segment.raw_text),
            cleaned_len=len(cleaned_text),
        )

        # Update rolling context buffer.
        if cleaned_text:
            self._context.append(cleaned_text)
            if len(self._context) > self._max_context:
                self._context.pop(0)

        return CleanedSegment(
            speaker_role=segment.speaker_role,
            cleaned_text=cleaned_text,
            original_text=segment.raw_text,   # ALWAYS preserved
            corrections_applied=tuple(corrections),
            confidence=None,
            t_start=segment.t_start,
            t_end=segment.t_end,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, raw_text: str) -> str:
        """Build the user turn: numbered vocabulary list + context + raw text."""
        parts: list[str] = []

        # Vocabulary injection (OpenWhispr numbered-list pattern).
        if self._vocab:
            sorted_vocab = sorted(self._vocab)
            vocab_lines = [f"{i+1}. {term}" for i, term in enumerate(sorted_vocab)]
            parts.append("Medical vocabulary (correct misspellings to match these terms):")
            parts.extend(vocab_lines)
            parts.append("")

        # Conversation context injection (last N turns).
        recent = self._context[-self._max_context:]
        if recent:
            parts.append("Recent conversation context (for coreference resolution):")
            for j, ctx_turn in enumerate(recent, start=1):
                parts.append(f"{j}. {ctx_turn}")
            parts.append("")

        parts.append("Transcription to clean:")
        parts.append(raw_text)

        return "\n".join(parts)

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call the cleanup LLM and return the cleaned text string.

        For gpt-4o-mini (and any OpenAI-compatible endpoint), uses the
        ``openai`` SDK. For qwen2.5-7b-local, expects a locally-running
        vLLM endpoint at the URL in ``QWEN_ENDPOINT`` env var (falls back to
        http://localhost:8000/v1). The openai SDK's ``base_url`` param handles
        vLLM compatibility transparently.

        Raises
        ------
        CleanupUnavailable
            If the relevant API key / endpoint is absent.
        """
        import os

        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CleanupUnavailable(
                "transcript_cleanup requires openai SDK: pip install openai"
            ) from exc

        if self._model.startswith("qwen") or self._model.endswith("-local"):
            endpoint = os.environ.get("QWEN_ENDPOINT", "http://localhost:8000/v1")
            api_key = os.environ.get("QWEN_API_KEY", "local")
            client = openai.OpenAI(base_url=endpoint, api_key=api_key)
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise CleanupUnavailable(
                    "OPENAI_API_KEY is not set. Either set the env var or "
                    "use EvalCleanupConfig with a local Qwen endpoint."
                )
            client = openai.OpenAI(api_key=api_key)

        response: Any = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return str(response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Diff-based correction attribution
# ---------------------------------------------------------------------------

_FILLER_WORDS = frozenset({
    "um", "uh", "er", "ah", "hmm", "hm", "erm", "uhh", "umm", "ahh",
    "you know", "you know what", "like", "right", "okay so", "ok so",
    "so", "well", "i mean",
})

_LAY_LANGUAGE_TAG_PATTERN = r"\[likely: [^\]]+\]"


def _diff_corrections(
    original: str,
    cleaned: str,
) -> list[CleanupCorrection]:
    """Produce a list of CleanupCorrection from an original / cleaned pair.

    Uses SequenceMatcher word-level diff to find changed spans, then
    categorises each change with a best-effort reason_category.
    """
    orig_words = original.split()
    clean_words = cleaned.split()
    if not orig_words and not clean_words:
        return []

    sm = difflib.SequenceMatcher(None, orig_words, clean_words, autojunk=False)
    corrections: list[CleanupCorrection] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        orig_span = " ".join(orig_words[i1:i2])
        new_span = " ".join(clean_words[j1:j2])

        # Determine reason category (best-effort pattern match).
        category = _categorise(orig_span, new_span)
        corrections.append(
            CleanupCorrection(
                original_span=orig_span,
                replacement=new_span,
                reason_category=category,
            )
        )

    return corrections


def _categorise(original: str, replacement: str) -> str:
    """Heuristically categorise a correction."""
    import re

    # Filler removal: original is a filler word, replacement is empty.
    if original.lower() in _FILLER_WORDS and not replacement:
        return "filler_removal"
    if not replacement and any(w.lower() in _FILLER_WORDS for w in original.split()):
        return "filler_removal"

    # Lay-language tag: replacement contains a [likely: ...] tag.
    if re.search(r"\[likely:", replacement):
        return "lay_language_tag"

    # Medical term correction: original and replacement differ only in spelling,
    # and replacement looks like a medical term (heuristic: longer words).
    if original.lower() != replacement.lower() and len(replacement) > 6:
        # Check if it looks like a drug/anatomy correction (contains common suffixes).
        import re as _re
        if _re.search(
            r"ol\b|ine\b|itis\b|osis\b|ectomy\b|pril\b|sartan\b|statin\b|mab\b|nib\b",
            replacement, _re.IGNORECASE
        ):
            return "medical_term_correction"

    # Punctuation: no word-count change, only punctuation differs.
    if original.replace(",", "").replace(".", "").replace(";", "").strip() == \
            replacement.replace(",", "").replace(".", "").replace(";", "").strip():
        return "punctuation"

    # Backtracking: original contains em-dash or " — " indicating self-correction.
    if "—" in original or " - " in original:
        return "backtracking_resolution"

    # Grammar: catch-all for word-count-preserving changes.
    if len(original.split()) == len(replacement.split()):
        return "grammar"

    return "other"
