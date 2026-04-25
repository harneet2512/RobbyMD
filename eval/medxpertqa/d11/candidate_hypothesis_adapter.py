"""Adapt answer options into CandidateHypothesis objects.

Converts MedXpertQA A–J options into the generic CandidateHypothesis
dataclass used by downstream pipeline stages. Provides a product-mode
stub for future RobbyMD candidate generation.
"""
from __future__ import annotations

import structlog

from eval.medxpertqa.d11.types import CandidateHypothesis

log = structlog.get_logger(__name__)


def adapt_medxpertqa_options(
    options: dict[str, str],
) -> list[CandidateHypothesis]:
    """Convert MedXpertQA A–J options to CandidateHypothesis list.

    Each option letter becomes ``cand_{letter}`` (e.g. ``cand_A``).
    ``candidate_type`` defaults to ``"diagnosis"`` because MedXpertQA
    is overwhelmingly diagnosis-oriented; callers can override after
    construction if the case metadata indicates otherwise.

    Args:
        options: Mapping of option letters to option text,
                 e.g. ``{"A": "Acute MI", "B": "PE", ...}``.

    Returns:
        Sorted list of :class:`CandidateHypothesis` objects.
    """
    candidates: list[CandidateHypothesis] = []
    for letter in sorted(options.keys()):
        label = options[letter].strip()
        if not label:
            log.warning(
                "adapt.empty_option",
                letter=letter,
                msg="Skipping option with empty label",
            )
            continue
        candidates.append(
            CandidateHypothesis(
                candidate_id=f"cand_{letter}",
                candidate_label=label,
                candidate_type="diagnosis",
            )
        )
    log.debug("adapt.medxpertqa", n_candidates=len(candidates))
    return candidates


def adapt_product_candidates(
    clinical_context: str,
) -> list[CandidateHypothesis]:
    """Stub for future RobbyMD candidate generation.

    Returns an empty list — product mode is not yet implemented.
    This function exists to prove that downstream code works
    without MedXpertQA-specific logic.

    Args:
        clinical_context: Free-text clinical context (unused).

    Returns:
        Empty list.
    """
    log.debug(
        "adapt.product_stub",
        context_len=len(clinical_context),
        msg="Product candidate generation not yet implemented",
    )
    return []
