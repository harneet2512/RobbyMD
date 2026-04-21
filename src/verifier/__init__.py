"""Counterfactual verifier per Eng_doc.md §6.

Deterministic selection of a discriminator feature between the top-2 branches;
one Opus 4.7 call to phrase the next-best clinical question in ≤20 words.

Prior art cited in src/differential/README.md:
    - CF-Multi-Agent Dx (arXiv 2603.27820)    — CPG metric
    - MedEinst (arXiv 2601.06636)             — Bias Trap Rate

Public surface:
    verify(ranking, lr_table, active_claims, opus_client=None) -> VerifierOutput
    MockOpusClient — offline fallback used when ANTHROPIC_API_KEY is unset
"""

from __future__ import annotations

from src.verifier.verifier import (
    MockOpusClient,
    OpusClient,
    VerifierOutput,
    select_discriminator,
    verify,
)

__all__ = [
    "MockOpusClient",
    "OpusClient",
    "VerifierOutput",
    "select_discriminator",
    "verify",
]
