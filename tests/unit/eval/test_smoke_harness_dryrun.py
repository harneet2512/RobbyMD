"""Smoke-harness dry-run test.

`run_smoke.py --dry-run` must exit 0, print a planned-matrix summary, and make
zero API calls. This test invokes the CLI in-process (no subprocess) and
asserts the expected output shape. Dataset absence is OK — dry-run reports
"missing" rather than crashing.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from eval.smoke import run_smoke


def test_dry_run_exits_zero_with_planned_matrix() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_smoke.main(
            [
                "--dry-run",
                "--benchmark",
                "longmemeval",
                "--reader",
                "gpt-4o-mini",
                "--variant",
                "baseline",
                "--n",
                "10",
            ]
        )
    output = buf.getvalue()
    assert rc == 0
    assert "DRY RUN" in output
    assert "planned matrix" in output.lower()
    assert "Verdict: [OK] PASS" in output


def test_dry_run_on_empty_dataset_dir_is_not_a_crash() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_smoke.main(
            [
                "--dry-run",
                "--benchmark",
                "both",
                "--reader",
                "qwen2.5-14b",
                "--variant",
                "both",
                "--n",
                "10",
            ]
        )
    output = buf.getvalue()
    # Should not crash — reports missing datasets gracefully.
    assert rc == 0
    # If dataset dir doesn't exist (clean CI), we should see "missing:" markers.
    # If dataset dir has been prepared locally, we see "FOUND" — either is fine.
    assert ("missing:" in output) or ("FOUND" in output)


def test_dry_run_lists_all_benchmark_reader_variant_combinations() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_smoke.main(
            [
                "--dry-run",
                "--benchmark",
                "both",
                "--reader",
                "all",
                "--variant",
                "both",
                "--n",
                "10",
            ]
        )
    output = buf.getvalue()
    assert rc == 0
    # 2 benchmarks × 4 readers (qwen, gpt-4o-mini, gpt-4.1-mini, gpt-4.1) × 2 variants = 16.
    combination_lines = [ln for ln in output.splitlines() if ln.strip().startswith("[smoke]   -")]
    assert len(combination_lines) == 16
