"""Evidence verifier — classify retrieved evidence by relevance to question.

Heuristic-first (deterministic, zero LLM cost). Classification order:
1. Answer-type matching (question pattern → value signal)
2. Coverage-based DIRECT/SUPPORTING
3. Scoped conflict detection (same attribute, not just same predicate)
4. BACKGROUND / IRRELEVANT fallback
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from src.substrate.schema import Claim


class EvidenceType(StrEnum):
    DIRECT = "direct"
    SUPPORTING = "supporting"
    CONFLICT = "conflict"
    BACKGROUND = "background"
    IRRELEVANT = "irrelevant"


@dataclass(frozen=True, slots=True)
class ClassifiedEvidence:
    claim: Claim
    fused_score: float
    evidence_type: EvidenceType
    confidence: float
    conflict_with: str | None
    reason: str


_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
        "did", "do", "does", "for", "from", "get", "had", "has", "have",
        "how", "i", "if", "in", "is", "it", "just", "me", "my", "not",
        "of", "on", "or", "please", "tell", "than", "that", "the",
        "their", "there", "they", "this", "to", "us", "was", "we",
        "were", "what", "when", "where", "which", "who", "will",
        "with", "would", "you", "your",
    }
)

_DURATION_RE = re.compile(r"\d+\s*(?:minute|hour|day|week|month|year|min|hr|sec)", re.I)
_NUMBER_RE = re.compile(r"\d+")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d", re.I)


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9']+", text.lower()))
    return tokens - _STOPWORDS


def _value_tokens(claim: Claim) -> set[str]:
    return _tokenize(claim.value)


def _coverage(question_tokens: set[str], claim: Claim) -> float:
    vt = _value_tokens(claim)
    if not question_tokens or not vt:
        return 0.0
    return len(question_tokens & vt) / max(1, len(question_tokens))


def _answer_type_match(question: str, claim: Claim) -> bool:
    """Check if claim value contains the answer type the question asks for."""
    ql = question.lower()
    val = claim.value

    if any(p in ql for p in ("how long", "how many", "duration", "how much time")):
        return bool(_DURATION_RE.search(val))

    if any(p in ql for p in ("where", "what place", "which store", "what store", "which location")):
        # Proper nouns (title-cased words) or known location patterns
        return bool(re.search(r"[A-Z][a-z]{2,}", val)) or any(
            w in val.lower() for w in ("target", "walmart", "costco", "store", "city", "town")
        )

    if any(p in ql for p in ("what degree", "what did you study", "graduate", "major")):
        return any(w in val.lower() for w in (
            "degree", "bachelor", "master", "phd", "administration", "engineering",
            "science", "arts", "mba", "diploma", "certificate",
        ))

    if any(p in ql for p in ("when", "what date", "what time", "what day")):
        return bool(_TIME_RE.search(val)) or bool(re.search(r"\b(last|next|ago|yesterday|today)\b", val, re.I))

    if any(p in ql for p in ("what is the name", "what was the name", "what playlist",
                               "what play ", "name of your", "name of the")):
        if re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", val):
            return True
        if re.search(r'"[^"]+"', val):
            return True
        if re.search(r"\b(?:named|called|name is|name was)\s+[A-Z][a-z]+", val):
            return True
        return False

    return False


def _detect_scoped_conflicts(
    candidates: list[tuple[Claim, float]],
) -> dict[str, str]:
    """Detect conflicts only between claims about the SAME attribute.

    Two claims conflict when they share (subject, predicate) AND their values
    have Jaccard similarity >= 0.2 (same topic scope) AND the values are
    not identical. Claims about completely different topics (commute vs hiking)
    with the same predicate are NOT conflicts.
    """
    grouped: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for claim, _ in candidates:
        grouped[(claim.subject, claim.predicate)].append(claim)

    conflicts: dict[str, str] = {}
    for claims in grouped.values():
        if len(claims) < 2:
            continue
        for i, c1 in enumerate(claims):
            for c2 in claims[i + 1:]:
                v1 = c1.value.strip().lower()
                v2 = c2.value.strip().lower()
                if v1 == v2:
                    continue
                t1 = _tokenize(v1)
                t2 = _tokenize(v2)
                if not t1 or not t2:
                    continue
                # Short values (1-2 words) are atomic facts — same predicate
                # with different short values IS a conflict (Denver vs Boston)
                both_short = len(t1) <= 2 and len(t2) <= 2
                jaccard = len(t1 & t2) / len(t1 | t2)
                if both_short or jaccard >= 0.2:
                    conflicts[c1.claim_id] = c2.claim_id
                    conflicts[c2.claim_id] = c1.claim_id
    return conflicts


def classify_evidence(
    question: str,
    question_type: str,
    candidates: list[tuple[Claim, float]],
    *,
    supersession_pair_claim_ids: frozenset[str] | None = None,
) -> list[ClassifiedEvidence]:
    """Classify each candidate by relevance to the question.

    Order: answer-type match → coverage → scoped conflict → fallback.
    A claim that directly answers the question is DIRECT even if other
    claims share the same predicate.
    """
    if not candidates:
        return []

    question_tokens = _tokenize(question)
    conflicts = _detect_scoped_conflicts(candidates)
    pair_ids = supersession_pair_claim_ids or frozenset()

    result: list[ClassifiedEvidence] = []
    for claim, fused_score in candidates:
        cov = _coverage(question_tokens, claim)
        type_match = _answer_type_match(question, claim)

        # 1. Supersession pair members are DIRECT for knowledge_update
        if question_type == "knowledge_update" and claim.claim_id in pair_ids:
            result.append(ClassifiedEvidence(
                claim=claim, fused_score=fused_score,
                evidence_type=EvidenceType.DIRECT, confidence=0.85,
                conflict_with=None,
                reason="supersession pair for knowledge_update",
            ))
            continue

        # 2. Answer-type match → DIRECT (with coverage) or SUPPORTING (without)
        #    type_match means the claim value contains the answer TYPE the
        #    question asks for (duration for "how long", location for "where").
        #    Coverage measures question-word overlap, which is a bonus but
        #    not required when the answer type is already confirmed.
        if type_match and cov > 0.1:
            result.append(ClassifiedEvidence(
                claim=claim, fused_score=fused_score,
                evidence_type=EvidenceType.DIRECT, confidence=0.9,
                conflict_with=None,
                reason=f"answer_type_match + coverage={cov:.2f}",
            ))
            continue
        if type_match and fused_score > 0:
            result.append(ClassifiedEvidence(
                claim=claim, fused_score=fused_score,
                evidence_type=EvidenceType.SUPPORTING, confidence=0.7,
                conflict_with=None,
                reason=f"answer_type_match (no coverage), fused={fused_score:.3f}",
            ))
            continue

        # 3. High coverage alone → DIRECT
        if cov > 0.3:
            result.append(ClassifiedEvidence(
                claim=claim, fused_score=fused_score,
                evidence_type=EvidenceType.DIRECT, confidence=min(0.9, 0.5 + cov),
                conflict_with=None,
                reason=f"coverage={cov:.2f}",
            ))
            continue

        # 4. Scoped conflict (only after ruling out DIRECT)
        if claim.claim_id in conflicts:
            result.append(ClassifiedEvidence(
                claim=claim, fused_score=fused_score,
                evidence_type=EvidenceType.CONFLICT, confidence=0.8,
                conflict_with=conflicts[claim.claim_id],
                reason=f"scoped conflict with {conflicts[claim.claim_id]}",
            ))
            continue

        # 5. Moderate coverage or positive fused score → SUPPORTING
        if cov > 0.1 or (fused_score > 0 and cov > 0):
            result.append(ClassifiedEvidence(
                claim=claim, fused_score=fused_score,
                evidence_type=EvidenceType.SUPPORTING,
                confidence=0.5 + cov,
                conflict_with=None,
                reason=f"fused={fused_score:.3f} coverage={cov:.2f}",
            ))
            continue

        # 6. Positive fused score but no coverage → BACKGROUND
        if fused_score > 0:
            result.append(ClassifiedEvidence(
                claim=claim, fused_score=fused_score,
                evidence_type=EvidenceType.BACKGROUND, confidence=0.3,
                conflict_with=None,
                reason=f"fused={fused_score:.3f} no coverage",
            ))
            continue

        # 7. Nothing → IRRELEVANT
        result.append(ClassifiedEvidence(
            claim=claim, fused_score=fused_score,
            evidence_type=EvidenceType.IRRELEVANT, confidence=0.2,
            conflict_with=None,
            reason=f"fused={fused_score:.3f} coverage={cov:.2f}",
        ))

    return result


def filter_evidence(
    classified: list[ClassifiedEvidence],
    keep: tuple[EvidenceType, ...] = (
        EvidenceType.DIRECT,
        EvidenceType.SUPPORTING,
        EvidenceType.CONFLICT,
    ),
) -> list[ClassifiedEvidence]:
    return [e for e in classified if e.evidence_type in keep]


class EvidenceSufficiency:
    SUFFICIENT = "sufficient"
    MARGINAL = "marginal"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"

    @staticmethod
    def assess(classified: list[ClassifiedEvidence]) -> str:
        if not classified:
            return EvidenceSufficiency.INSUFFICIENT
        has_direct = any(e.evidence_type == EvidenceType.DIRECT for e in classified)
        has_supporting = any(e.evidence_type == EvidenceType.SUPPORTING for e in classified)
        has_conflict = any(e.evidence_type == EvidenceType.CONFLICT for e in classified)
        if has_direct:
            return EvidenceSufficiency.SUFFICIENT
        if has_conflict:
            return EvidenceSufficiency.CONFLICTED
        if has_supporting:
            return EvidenceSufficiency.MARGINAL
        return EvidenceSufficiency.INSUFFICIENT

    @staticmethod
    def should_abstain(sufficiency: str) -> bool:
        return sufficiency == EvidenceSufficiency.INSUFFICIENT


__all__ = [
    "ClassifiedEvidence",
    "EvidenceType",
    "EvidenceSufficiency",
    "classify_evidence",
    "filter_evidence",
]
