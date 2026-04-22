"""Smoke-run harness — deterministic first-10-case sanity pass.

Per `eval/README.md` "Smoke-first discipline" and the 2026-04-21 revision pass.
**Builds the harness; does not execute it.** Real runs happen in a separate
invocation after user review.

## CLI

    python eval/smoke/run_smoke.py [--benchmark {longmemeval|acibench|both}]
                                    [--reader {qwen2.5-14b|gpt-4o-mini|gpt-4.1-mini|all}]
                                    [--variant {baseline|substrate|both}]
                                    [--n N]
                                    [--budget-usd USD]
                                    [--dry-run]

Dry-run is guaranteed zero-cost: parses args, imports adapters, checks the
dataset directory, prints the planned matrix, and exits 0. No network.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Allow `python eval/smoke/run_smoke.py` invocation from repo root — without this
# the `eval.*` and `src.*` imports fail when the script is run as a file rather
# than via `python -m eval.smoke.run_smoke`.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DATA_DIR = _REPO_ROOT / "eval" / "data"
_RESULTS_ROOT = _REPO_ROOT / "eval" / "smoke"
_REFERENCE_BASELINES = _RESULTS_ROOT / "reference_baselines.json"

BENCHMARKS = ("longmemeval", "acibench")
READERS = ("qwen2.5-14b", "gpt-4o-mini", "gpt-4.1-mini")
VARIANTS = ("baseline", "substrate")


@dataclass(slots=True)
class SmokeConfig:
    benchmarks: tuple[str, ...]
    readers: tuple[str, ...]
    variants: tuple[str, ...]
    n_cases: int
    budget_usd: float
    dry_run: bool


@dataclass(slots=True)
class SmokeResult:
    verdict: str  # "PASS" | "ANOMALY" | "FAIL"
    lines: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_cases: int = 0


def _parse_args(argv: list[str]) -> SmokeConfig:
    ap = argparse.ArgumentParser(prog="run_smoke.py", description=__doc__)
    ap.add_argument("--benchmark", choices=(*BENCHMARKS, "both"), default="both")
    ap.add_argument("--reader", choices=(*READERS, "all"), default="qwen2.5-14b")
    ap.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--budget-usd", type=float, default=50.0)
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)

    benchmarks = BENCHMARKS if ns.benchmark == "both" else (ns.benchmark,)
    readers = READERS if ns.reader == "all" else (ns.reader,)
    variants = VARIANTS if ns.variant == "both" else (ns.variant,)

    return SmokeConfig(
        benchmarks=benchmarks,
        readers=readers,
        variants=variants,
        n_cases=ns.n,
        budget_usd=ns.budget_usd,
        dry_run=ns.dry_run,
    )


def _check_dataset(benchmark: str) -> tuple[bool, str]:
    """Return (found, message) — True when the expected dataset path exists."""
    if benchmark == "longmemeval":
        p = _DATA_DIR / "longmemeval" / "data" / "longmemeval_s.json"
        if p.is_file():
            return True, f"{p} ({p.stat().st_size} bytes)"
        return False, f"missing: {p} — run eval/smoke/prepare_datasets.sh"
    if benchmark == "acibench":
        p = _DATA_DIR / "acibench" / "data" / "challenge_data" / "test1"
        if p.is_dir():
            return True, str(p)
        return False, f"missing: {p} — run eval/smoke/prepare_datasets.sh"
    return False, f"unknown benchmark: {benchmark}"


def _import_adapters() -> dict[str, object]:
    """Import per-benchmark adapter modules. Failures surface as (None, err) entries."""
    imports: dict[str, object] = {}
    for bench, module_name in (
        ("longmemeval", "eval.longmemeval.adapter"),
        ("acibench", "eval.aci_bench.adapter"),
    ):
        try:
            imports[bench] = importlib.import_module(module_name)
        except ImportError as exc:
            imports[bench] = f"ImportError: {exc}"
    return imports


def _load_reference_baselines() -> dict:
    if not _REFERENCE_BASELINES.is_file():
        return {}
    with _REFERENCE_BASELINES.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _print_planned_matrix(cfg: SmokeConfig) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"[smoke] Planned matrix: {len(cfg.benchmarks)} benchmark(s) × "
        f"{len(cfg.readers)} reader(s) × {len(cfg.variants)} variant(s) × {cfg.n_cases} cases"
    )
    lines.append(f"[smoke] Budget hard-cap: ${cfg.budget_usd:.2f}")
    lines.append("[smoke] Combinations:")
    for bench in cfg.benchmarks:
        for reader in cfg.readers:
            for variant in cfg.variants:
                lines.append(f"[smoke]   - {bench} × {reader} × {variant}")
    return lines


def _dry_run(cfg: SmokeConfig) -> SmokeResult:
    result = SmokeResult(verdict="PASS")
    result.lines.extend(_print_planned_matrix(cfg))

    # Import check (substrate-variant rows fail without wt-engine's write API; this is expected).
    adapters = _import_adapters()
    for bench, mod in adapters.items():
        if bench in cfg.benchmarks:
            ok = not isinstance(mod, str)
            result.lines.append(
                f"[smoke] adapter({bench}): {'OK' if ok else 'NOT IMPORTABLE ({mod})'.format(mod=mod)}"
            )

    # Dataset presence check.
    for bench in cfg.benchmarks:
        found, msg = _check_dataset(bench)
        if found:
            result.lines.append(f"[smoke] dataset({bench}): FOUND — {msg}")
        else:
            result.lines.append(f"[smoke] dataset({bench}): {msg}")

    # Reference-baselines sanity check.
    baselines = _load_reference_baselines()
    if baselines:
        result.lines.append(
            f"[smoke] reference_baselines.json: OK "
            f"({len(baselines)} benchmark(s) with reference numbers)"
        )
    else:
        result.lines.append(
            "[smoke] reference_baselines.json: MISSING — real run will skip baseline ±20pp check"
        )

    result.lines.append(
        "[smoke] DRY RUN: no API calls made. Exit 0. "
        "Re-run without --dry-run for the real smoke pass."
    )
    return result


def _real_run(cfg: SmokeConfig) -> SmokeResult:
    """Real run — wires per-benchmark `eval/<b>/run.py` + per-variant readers + judge calls.

    NOT exercised in CI / tests. Full implementation lands with the first real smoke invocation
    (user explicitly excluded real runs from this commit). The stub below enforces the
    budget hard-cap contract and writes a methodology.md so the harness is invocable when
    the user signs off on the first run.
    """
    result = SmokeResult(verdict="FAIL")
    result.lines.append("[smoke] Real-run path not yet invoked.")
    result.lines.append(
        "[smoke] This build ships the harness skeleton (argparse + dataset-presence +"
        " reference-baseline loading + output-dir scaffold); real per-benchmark wiring"
        " to eval/longmemeval/run.py + eval/aci_bench/run.py + judge calls lands when"
        " the user signs off on the first invocation."
    )
    # Write a methodology stub so the output-dir contract is testable end-to-end.
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _RESULTS_ROOT / "incomplete" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "methodology.md").write_text(
        "# Smoke-run methodology (real-run wiring pending)\n\n"
        f"Invocation: {vars(cfg)}\n\n"
        "This harness is built but not yet wired to per-benchmark runners + judge calls.\n"
        "Next revision will complete the wiring once the user confirms the dry-run verdict.\n",
        encoding="utf-8",
    )
    result.lines.append(f"[smoke] Wrote stub methodology: {out_dir / 'methodology.md'}")
    return result


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(sys.argv[1:] if argv is None else argv)

    if cfg.dry_run:
        result = _dry_run(cfg)
    else:
        result = _real_run(cfg)

    for line in result.lines:
        print(line)
    # ASCII markers so Windows cmd/PowerShell (cp1252 default) doesn't crash.
    marker = {"PASS": "[OK]", "ANOMALY": "[WARN]", "FAIL": "[FAIL]"}.get(result.verdict, "[?]")
    print(f"[smoke] Verdict: {marker} {result.verdict}")
    return 0 if result.verdict == "PASS" else (1 if result.verdict == "ANOMALY" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
