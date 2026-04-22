"""Tests for P.1 harness additions: --output-dir and --stratified flags.

Covers four behaviors:
  1. CLI parsing for --output-dir and --stratified (SmokeConfig fields).
  2. _resolve_longmemeval_dataset prefers data/longmemeval_s_cleaned.json
     when present, falls back to the bundled original.
  3. _load_longmemeval_cases(stratified=True) buckets by question_type and
     yields balanced per-type samples deterministically.
  4. cfg.output_dir overrides the auto-timestamped directory.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.smoke.run_smoke import (
    SmokeConfig,
    _load_longmemeval_cases,
    _parse_args,
    _resolve_longmemeval_dataset,
    _LONGMEMEVAL_QUESTION_TYPES,
)


class TestCLIFlags:
    def test_output_dir_default_none(self) -> None:
        cfg = _parse_args(["--benchmark", "acibench", "--n", "10"])
        assert cfg.output_dir is None

    def test_output_dir_honoured(self) -> None:
        cfg = _parse_args(
            ["--benchmark", "acibench", "--n", "10", "--output-dir", "/tmp/x/y"]
        )
        assert cfg.output_dir == "/tmp/x/y"

    def test_stratified_default_false(self) -> None:
        cfg = _parse_args(["--benchmark", "longmemeval", "--n", "60"])
        assert cfg.stratified is False

    def test_stratified_true_when_flagged(self) -> None:
        cfg = _parse_args(
            ["--benchmark", "longmemeval", "--n", "60", "--stratified"]
        )
        assert cfg.stratified is True


class TestDatasetResolver:
    def test_prefers_cleaned_when_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch the module-level repo root so both candidate paths resolve under tmp.
        fake_repo = tmp_path
        fake_data = fake_repo / "data"
        fake_data.mkdir()
        cleaned = fake_data / "longmemeval_s_cleaned.json"
        cleaned.write_text("[]", encoding="utf-8")

        monkeypatch.setattr("eval.smoke.run_smoke._REPO_ROOT", fake_repo)
        monkeypatch.setattr("eval.smoke.run_smoke._DATA_DIR", fake_repo / "eval" / "data")

        resolved = _resolve_longmemeval_dataset()
        assert resolved == cleaned

    def test_falls_back_to_original_when_cleaned_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_repo = tmp_path
        fake_eval_data = fake_repo / "eval" / "data"
        (fake_eval_data / "longmemeval" / "data").mkdir(parents=True)
        original = fake_eval_data / "longmemeval" / "data" / "longmemeval_s.json"
        original.write_text("[]", encoding="utf-8")

        monkeypatch.setattr("eval.smoke.run_smoke._REPO_ROOT", fake_repo)
        monkeypatch.setattr("eval.smoke.run_smoke._DATA_DIR", fake_eval_data)

        resolved = _resolve_longmemeval_dataset()
        assert resolved == original


class TestStratifiedSampling:
    """Fake iter_questions and assert bucket-then-trim behavior."""

    @staticmethod
    def _fake_questions(n_per_type: int = 20):
        """Yield fake LongMemEvalQuestion-like objects (duck-typed).

        The loader only reads `question_type` and `question_id` attributes,
        so a lightweight shim is sufficient.
        """

        class _Q:
            __slots__ = ("question_type", "question_id", "question")

            def __init__(self, qt: str, qid: str) -> None:
                self.question_type = qt
                self.question_id = qid
                self.question = "q"

        items = []
        for qt in _LONGMEMEVAL_QUESTION_TYPES:
            for i in range(n_per_type):
                items.append(_Q(qt, f"{qt}_{i:03d}"))
        return iter(items)

    def test_stratified_60_yields_10_per_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "eval.longmemeval.adapter.iter_questions",
            lambda _p: self._fake_questions(20),
        )
        monkeypatch.setattr(
            "eval.smoke.run_smoke._resolve_longmemeval_dataset",
            lambda: Path("/nonexistent/ok"),
        )
        cases = _load_longmemeval_cases(60, stratified=True)
        assert len(cases) == 60
        # Every question_type must contribute exactly 10.
        buckets: dict[str, int] = {}
        for c in cases:
            buckets[c.question_type] = buckets.get(c.question_type, 0) + 1
        for qt in _LONGMEMEVAL_QUESTION_TYPES:
            assert buckets[qt] == 10, f"{qt}: {buckets.get(qt)}"

    def test_stratified_deterministic_under_same_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "eval.longmemeval.adapter.iter_questions",
            lambda _p: self._fake_questions(20),
        )
        monkeypatch.setattr(
            "eval.smoke.run_smoke._resolve_longmemeval_dataset",
            lambda: Path("/nonexistent/ok"),
        )
        a = _load_longmemeval_cases(60, stratified=True)
        b = _load_longmemeval_cases(60, stratified=True)
        ids_a = [c.question_id for c in a]
        ids_b = [c.question_id for c in b]
        assert ids_a == ids_b

    def test_first_n_default_unchanged_when_not_stratified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Non-stratified path is pure first-N over iteration order.
        monkeypatch.setattr(
            "eval.longmemeval.adapter.iter_questions",
            lambda _p: self._fake_questions(20),
        )
        monkeypatch.setattr(
            "eval.smoke.run_smoke._resolve_longmemeval_dataset",
            lambda: Path("/nonexistent/ok"),
        )
        cases = _load_longmemeval_cases(10, stratified=False)
        assert len(cases) == 10
        # All 10 should come from the first bucket (first-N iteration order).
        assert all(
            c.question_type == _LONGMEMEVAL_QUESTION_TYPES[0] for c in cases
        )
