"""Pack-driven prompt-loading tests per 2026-04-21 refactor.

Addresses audit finding #2 from commit `8f0d9db`: few-shot examples and
predicate families now come from the active `PredicatePack`, not hardcoded
module constants. Switching packs (via `ACTIVE_PACK` env var) should swap
both without code changes.
"""

from __future__ import annotations

import importlib
import os

import pytest

import src.extraction.claim_extractor.prompt as prompt_module
from src.substrate.predicate_packs import active_pack


def test_predicate_families_match_active_pack() -> None:
    """Module-level `PREDICATE_FAMILIES` mirrors the active pack's closed vocabulary."""
    pack = active_pack()
    # Module exports a sorted tuple; pack exposes a frozenset.
    assert frozenset(prompt_module.PREDICATE_FAMILIES) == pack.predicate_families


def test_few_shot_examples_come_from_active_pack() -> None:
    """`FEW_SHOT_EXAMPLES` is the active pack's `few_shot_examples` verbatim."""
    pack = active_pack()
    assert prompt_module.FEW_SHOT_EXAMPLES == pack.few_shot_examples


def test_default_active_pack_is_clinical_general() -> None:
    """`ACTIVE_PACK` default (unset) resolves to clinical_general."""
    # Only applies when no env-override is active.
    if os.environ.get("ACTIVE_PACK"):
        pytest.skip("ACTIVE_PACK env override set; default-path check skipped.")
    assert active_pack().pack_id == "clinical_general"


def test_prompt_body_references_closed_set() -> None:
    """System-prompt body explicitly names each predicate family from the active pack."""
    prompt = prompt_module.CLAIM_EXTRACTOR_SYSTEM_PROMPT
    for family in prompt_module.PREDICATE_FAMILIES:
        assert family in prompt, f"family {family!r} missing from prompt body"


def test_switching_active_pack_reloads_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting `ACTIVE_PACK=personal_assistant` + reloading the module swaps in that pack's data.

    Verifies the audit-finding-#2 fix: the prompt is swappable at pack-switch,
    not just at first import. Tests reload semantics, not live mutation (the
    `lru_cache` on `active_pack` is deliberately cached; tests must clear + reload).
    """
    monkeypatch.setenv("ACTIVE_PACK", "personal_assistant")
    active_pack.cache_clear()
    importlib.reload(prompt_module)
    try:
        assert "user_fact" in prompt_module.PREDICATE_FAMILIES
        assert "onset" not in prompt_module.PREDICATE_FAMILIES  # clinical-only family
        example_names = {ex.name for ex in prompt_module.FEW_SHOT_EXAMPLES}
        assert any(
            "user" in name or "supersession" in name or "correction" in name
            for name in example_names
        )
    finally:
        # Restore default pack so subsequent tests see clinical_general.
        monkeypatch.delenv("ACTIVE_PACK", raising=False)
        active_pack.cache_clear()
        importlib.reload(prompt_module)
