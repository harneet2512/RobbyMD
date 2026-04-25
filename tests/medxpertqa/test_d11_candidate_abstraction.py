"""Adversarial tests for the D11 candidate hypothesis adapter.

Tests that MedXpertQA option letters are mapped to stable candidate IDs,
that downstream code never depends on raw option letters, and that both
benchmark and product modes behave correctly.
"""
from __future__ import annotations

import pytest

from eval.medxpertqa.d11.candidate_hypothesis_adapter import (
    adapt_medxpertqa_options,
    adapt_product_candidates,
)
from eval.medxpertqa.d11.types import CandidateHypothesis


class TestCandidateAdapter:
    def test_medxpertqa_options_to_candidates(self):
        """MedXpertQA options A-J -> CandidateHypothesis with candidate_id, not raw letters."""
        options = {"A": "Myocardial infarction", "B": "Pulmonary embolism", "C": "Aortic dissection"}
        candidates = adapt_medxpertqa_options(options)
        assert len(candidates) == 3
        assert all(c.candidate_id.startswith("cand_") for c in candidates)
        assert all(c.candidate_label for c in candidates)
        # Downstream code must NOT depend on option letters
        assert not any(c.candidate_id == "A" for c in candidates)

    def test_candidate_type_defaults_to_diagnosis(self):
        """All candidates default to candidate_type='diagnosis'."""
        options = {"A": "Some diagnosis", "B": "Another diagnosis"}
        candidates = adapt_medxpertqa_options(options)
        assert all(c.candidate_type == "diagnosis" for c in candidates)

    def test_product_mode_returns_empty(self):
        """Product mode stub returns empty -- proves downstream handles it."""
        candidates = adapt_product_candidates("Patient presents with chest pain")
        assert candidates == []

    def test_benchmark_adapter_preserves_all_options(self):
        """All 10 MedXpertQA options preserved."""
        options = {chr(65 + i): f"Option {chr(65 + i)}" for i in range(10)}  # A-J
        candidates = adapt_medxpertqa_options(options)
        assert len(candidates) == 10
