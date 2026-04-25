"""Adversarial tests for the D11 pairwise discriminator tournament.

Tests that the deterministic pair selection logic produces correct pairs
for supported-vs-supported, supported-vs-trap, and respects the MAX_PAIRS
ceiling.
"""
from __future__ import annotations

import pytest

from eval.medxpertqa.d11.pairwise_discriminator_tournament import select_pairs
from eval.medxpertqa.d11.types import (
    CandidateEvidence,
    Claim,
    PairwiseDiscriminator,
)


class TestPairSelection:
    def test_supported_pair_selected(self):
        """Two supported candidates -> pair selected for discrimination."""
        evidence = [
            CandidateEvidence("cand_A", "A", [Claim("supports A", True, "mechanism", "medium")], [], [], [], "supported"),
            CandidateEvidence("cand_B", "B", [Claim("supports B", True, "mechanism", "medium")], [], [], [], "supported"),
            CandidateEvidence("cand_C", "C", [], [], [], [], "insufficient"),
        ]
        pairs = select_pairs(evidence, trap_candidates=[])
        assert ("cand_A", "cand_B") in pairs or ("cand_B", "cand_A") in pairs

    def test_supported_vs_trap_selected(self):
        """Supported + trap candidate -> pair selected."""
        evidence = [
            CandidateEvidence("cand_A", "A", [Claim("supports A", True, "mechanism", "medium")], [], [], [], "supported"),
            CandidateEvidence("cand_B", "B", [Claim("looks like B", False, "mechanism", "weak")], [], [], [], "trap"),
        ]
        pairs = select_pairs(evidence, trap_candidates=["cand_B"])
        assert len(pairs) >= 1
        pair_sets = [set(p) for p in pairs]
        assert {"cand_A", "cand_B"} in pair_sets

    def test_max_5_pairs(self):
        """Never select more than 5 pairs."""
        evidence = [
            CandidateEvidence(
                f"cand_{chr(65 + i)}",
                chr(65 + i),
                [Claim(f"supports {chr(65 + i)}", True, "mechanism", "medium")],
                [],
                [],
                [],
                "supported",
            )
            for i in range(8)
        ]
        pairs = select_pairs(evidence, trap_candidates=[])
        assert len(pairs) <= 5
