"""Unit tests for the LR-table loader. Validates the canonical chest-pain table."""

from __future__ import annotations

import json

import pytest

from src.differential.lr_table import BRANCHES, PREDICATE_FAMILIES, load_lr_table
from tests.fixtures.loader import LR_TABLE_PATH


def test_load_lr_table_parses_canonical_file() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    assert len(table.rows) >= 60  # current count is 79; leave head-room for curation
    assert table.chief_complaint == "chest_pain"


def test_every_row_has_a_citation_per_rules_4_4() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    for row in table.rows:
        assert row.source.strip(), f"row {row.feature} missing source"
        assert row.source_url.strip(), f"row {row.feature} missing source_url"


def test_every_branch_is_in_closed_set() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    seen = {r.branch for r in table.rows}
    assert seen.issubset(BRANCHES)
    assert seen == BRANCHES  # all four branches populated


def test_every_predicate_family_is_in_eng_doc_closed_set() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    for row in table.rows:
        assert row.predicate in PREDICATE_FAMILIES, row.predicate_path


def test_by_predicate_path_lookup_is_o1(tmp_path) -> None:
    table = load_lr_table(LR_TABLE_PATH)
    hits = table.rows_for("aggravating_factor=inspiration")
    branches_hit = {r.branch for r in hits}
    # Pleuritic-pain predicate is shared across branches (cardiac rule-out, pulmonary, msk).
    assert {"cardiac", "pulmonary", "msk"} <= branches_hit


def test_by_branch_lookup_materialises_expected_counts() -> None:
    table = load_lr_table(LR_TABLE_PATH)
    counts = {b: len(table.rows_on_branch(b)) for b in BRANCHES}
    # Loose lower bounds matching research/clinical_chest_pain.md §1 distribution.
    assert counts["cardiac"] >= 18
    assert counts["pulmonary"] >= 15
    assert counts["msk"] >= 12
    assert counts["gi"] >= 12


def test_invented_values_are_rejected(tmp_path) -> None:
    bad = {
        "version": "0.0.1",
        "chief_complaint": "test",
        "entries": [
            {
                "branch": "cardiac",
                "feature": "x",
                "predicate_path": "onset=sudden",
                "lr_plus": 2.0,
                # missing 'source'
                "source_url": "http://example",
            }
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or empty 'source'"):
        load_lr_table(p)


def test_unknown_branch_rejected(tmp_path) -> None:
    bad = {
        "entries": [
            {
                "branch": "endocrine",
                "feature": "x",
                "predicate_path": "onset=sudden",
                "lr_plus": 2.0,
                "source": "src",
                "source_url": "url",
            }
        ]
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown branch"):
        load_lr_table(p)


def test_unknown_predicate_family_rejected(tmp_path) -> None:
    bad = {
        "entries": [
            {
                "branch": "cardiac",
                "feature": "x",
                "predicate_path": "genomics=snp_rs123",
                "lr_plus": 2.0,
                "source": "src",
                "source_url": "url",
            }
        ]
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="closed set"):
        load_lr_table(p)
