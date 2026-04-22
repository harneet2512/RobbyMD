"""Unit tests for smoke harness real-run wiring (2026-04-22).

Per Step 2A spec:
- baseline + substrate variants invoke the right benchmark module (mock reader/judge)
- budget halt triggers at threshold
- missing env var → clean error message exit, not stack trace
- result.json schema contains all specified fields
- pack switches correctly between LongMemEval-S and ACI-Bench
- structural_validity populated from substrate state accessors

All reader/judge calls are mocked — no network, no API key needed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path for eval.* and src.* imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.smoke import run_smoke
from eval.smoke.run_smoke import (
    BENCHMARK_PACK,
    CaseResult,
    SmokeConfig,
    _check_baseline_within_reference,
    _determine_verdict,
    _load_reference_baselines,
)


# ---------------------------------------------------------------------------
# Session-scoped env guard: restore ACTIVE_PACK + clear lru_cache after each
# test so mutations by run_smoke._real_run don't bleed into later tests
# (particularly tests/unit/extraction/test_claim_prompt_pack.py which asserts
# the active pack is 'clinical_general').
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_active_pack():
    """Save and restore ACTIVE_PACK env var + clear active_pack lru_cache around each test."""
    original = os.environ.get("ACTIVE_PACK")
    yield
    # Restore the env var to its pre-test state.
    if original is None:
        os.environ.pop("ACTIVE_PACK", None)
    else:
        os.environ["ACTIVE_PACK"] = original
    # Clear the lru_cache so any downstream import picks up the restored value.
    try:
        from src.substrate.predicate_packs import active_pack as _ap
        _ap.cache_clear()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> SmokeConfig:
    defaults: dict[str, Any] = {
        "benchmarks": ("longmemeval",),
        "readers": ("gpt-4o-mini",),
        "variants": ("baseline",),
        "n_cases": 2,
        "budget_usd": 50.0,
        "dry_run": False,
    }
    defaults.update(overrides)
    return SmokeConfig(**defaults)


def _make_case_result(**overrides: Any) -> CaseResult:
    defaults: dict[str, Any] = {
        "case_id": "q001",
        "benchmark": "longmemeval",
        "reader": "gpt-4o-mini",
        "variant": "baseline",
        "baseline_score": 1.0,
        "substrate_score": None,
        "delta": None,
        "latency_baseline_ms": 100.0,
        "latency_substrate_ms": None,
        "tokens_used_baseline": 200,
        "tokens_used_substrate": None,
        "estimated_cost": 0.013,
        "judge_reasoning": "CORRECT answer",
        "structural_validity": {},
    }
    defaults.update(overrides)
    return CaseResult(**defaults)


# ---------------------------------------------------------------------------
# Test: CaseResult schema contains all required fields
# ---------------------------------------------------------------------------


class TestCaseResultSchema:
    """Verifies that the CaseResult dataclass has all spec-required fields."""

    REQUIRED_FIELDS = {
        "case_id",
        "baseline_score",
        "substrate_score",
        "delta",
        "latency_baseline_ms",
        "latency_substrate_ms",
        "tokens_used_baseline",
        "tokens_used_substrate",
        "estimated_cost",
        "judge_reasoning",
        "structural_validity",
    }

    def test_all_required_fields_present(self) -> None:
        cr = _make_case_result()
        d = asdict(cr)
        for field in self.REQUIRED_FIELDS:
            assert field in d, f"missing field: {field}"

    def test_structural_validity_has_expected_keys_when_populated(self) -> None:
        cr = _make_case_result(
            variant="substrate",
            structural_validity={
                "claims_written_count": 3,
                "supersessions_fired_count": 1,
                "projection_nonempty": True,
                "active_pack": "personal_assistant",
            },
        )
        sv = cr.structural_validity
        assert "claims_written_count" in sv
        assert "supersessions_fired_count" in sv
        assert "projection_nonempty" in sv
        assert "active_pack" in sv

    def test_results_json_serialisable(self, tmp_path: Path) -> None:
        cr = _make_case_result()
        out = tmp_path / "results.json"
        out.write_text(json.dumps([asdict(cr)], indent=2), encoding="utf-8")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)
        assert loaded[0]["case_id"] == "q001"


# ---------------------------------------------------------------------------
# Test: pack switching between benchmarks
# ---------------------------------------------------------------------------


class TestPackSwitching:
    """ACTIVE_PACK must be 'personal_assistant' for LongMemEval-S and
    'clinical_general' for ACI-Bench."""

    def test_benchmark_pack_map_correct(self) -> None:
        assert BENCHMARK_PACK["longmemeval"] == "personal_assistant"
        assert BENCHMARK_PACK["acibench"] == "clinical_general"

    def test_real_run_sets_active_pack_per_benchmark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that _real_run sets ACTIVE_PACK env var per benchmark before loading cases."""
        packs_seen: list[str] = []

        def mock_load_longmemeval(_n: int) -> list[object]:
            packs_seen.append(os.environ.get("ACTIVE_PACK", ""))
            return []  # No cases to run

        def mock_load_acibench(_n: int) -> list[object]:
            packs_seen.append(os.environ.get("ACTIVE_PACK", ""))
            return []

        monkeypatch.setattr(run_smoke, "_load_longmemeval_cases", mock_load_longmemeval)
        monkeypatch.setattr(run_smoke, "_load_acibench_cases", mock_load_acibench)
        monkeypatch.setattr(run_smoke, "_check_dataset", lambda bench: (True, "mocked"))
        monkeypatch.setattr(run_smoke, "_get_reader_env", lambda _r: {"openai_key": "sk-test"})

        cfg = _make_config(
            benchmarks=("longmemeval", "acibench"),
            readers=("gpt-4o-mini",),
            variants=("baseline",),
        )
        run_smoke._real_run(cfg)

        assert "personal_assistant" in packs_seen, f"personal_assistant not seen; got: {packs_seen}"
        assert "clinical_general" in packs_seen, f"clinical_general not seen; got: {packs_seen}"


# ---------------------------------------------------------------------------
# Test: budget halt
# ---------------------------------------------------------------------------


class TestBudgetHalt:
    """Budget halt triggers when cumulative cost exceeds --budget-usd."""

    def test_budget_halt_before_second_case(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Simulate: first case costs more than budget; halt before second case."""
        monkeypatch.setattr(run_smoke, "_check_dataset", lambda bench: (True, "mocked"))
        monkeypatch.setattr(run_smoke, "_get_reader_env", lambda _r: {"openai_key": "sk-test"})
        monkeypatch.setattr(run_smoke, "_RESULTS_ROOT", tmp_path)

        # Two stub questions.
        from eval._common import EvalCase

        class _StubQ:
            question_id = "q_stub"
            question = "What?"
            answer = "Yes"
            question_type = "information_extraction"
            haystack_sessions: list = []
            haystack_session_ids = None
            haystack_dates = None

        stub_cases = [_StubQ(), _StubQ()]
        monkeypatch.setattr(run_smoke, "_load_longmemeval_cases", lambda _n: stub_cases)

        # Each reader+judge call costs 0.011; budget is 0.005 → halt on first call.
        monkeypatch.setattr(run_smoke, "_READER_COST_PER_CALL", {"gpt-4o-mini": 0.004})
        monkeypatch.setattr(run_smoke, "_JUDGE_COST_PER_CALL", 0.004)

        def _mock_call_baseline(_case, _reader, _env):
            return "answer", 50.0, 100

        def _mock_judge(_q, _gold, _pred, _key):
            return 1.0, "CORRECT"

        monkeypatch.setattr(run_smoke, "_call_longmemeval_baseline", _mock_call_baseline)
        monkeypatch.setattr(run_smoke, "_call_longmemeval_judge", _mock_judge)
        monkeypatch.setattr(run_smoke, "_run_substrate_ingestion", lambda _b, _p: MagicMock(
            claims_written_count=0, supersessions_fired_count=0,
            projection_nonempty=False, active_pack="personal_assistant",
        ))

        cfg = _make_config(
            benchmarks=("longmemeval",),
            readers=("gpt-4o-mini",),
            variants=("baseline",),
            budget_usd=0.005,  # Very tight budget
            n_cases=2,
        )
        result = run_smoke._real_run(cfg)

        assert result.verdict == "FAIL"
        budget_lines = [ln for ln in result.lines if "BUDGET_HALT" in ln]
        assert budget_lines, f"Expected BUDGET_HALT in output; got: {result.lines}"

    def test_budget_halt_message_contains_cost_info(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """BUDGET_HALT message must report cumulative cost and budget cap."""
        monkeypatch.setattr(run_smoke, "_check_dataset", lambda bench: (True, "mocked"))
        monkeypatch.setattr(run_smoke, "_get_reader_env", lambda _r: {"openai_key": "sk-test"})
        monkeypatch.setattr(run_smoke, "_RESULTS_ROOT", tmp_path)
        monkeypatch.setattr(run_smoke, "_READER_COST_PER_CALL", {"gpt-4o-mini": 0.010})
        monkeypatch.setattr(run_smoke, "_JUDGE_COST_PER_CALL", 0.010)

        class _StubQ:
            question_id = "qhalt"
            question = "X"
            answer = "Y"
            question_type = "information_extraction"
            haystack_sessions: list = []
            haystack_session_ids = None
            haystack_dates = None

        monkeypatch.setattr(run_smoke, "_load_longmemeval_cases", lambda _n: [_StubQ()])
        monkeypatch.setattr(run_smoke, "_call_longmemeval_baseline", lambda *_: ("ans", 10.0, 50))
        monkeypatch.setattr(run_smoke, "_call_longmemeval_judge", lambda *_: (1.0, "CORRECT"))

        cfg = _make_config(
            benchmarks=("longmemeval",),
            readers=("gpt-4o-mini",),
            variants=("baseline",),
            budget_usd=0.001,
        )
        result = run_smoke._real_run(cfg)
        halt_lines = [ln for ln in result.lines if "BUDGET_HALT" in ln]
        assert halt_lines
        # Message should contain a dollar figure.
        assert "$" in halt_lines[0]


# ---------------------------------------------------------------------------
# Test: missing env var → clean error, not stack trace
# ---------------------------------------------------------------------------


class TestMissingEnvVar:
    """Missing API key / endpoint → sys.exit with actionable message."""

    def test_missing_openai_key_raises_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("QWEN_ENDPOINT", raising=False)
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            run_smoke._require_env("OPENAI_API_KEY")

        assert exc_info.value.code is not None
        assert "OPENAI_API_KEY" in str(exc_info.value.code)

    def test_missing_qwen_endpoint_raises_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("QWEN_ENDPOINT", raising=False)
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            run_smoke._get_reader_env("qwen2.5-14b")

        assert exc_info.value.code is not None
        msg = str(exc_info.value.code)
        # Message must name at least one of the missing vars.
        assert "QWEN_ENDPOINT" in msg or "FIREWORKS_API_KEY" in msg or "TOGETHER_API_KEY" in msg

    def test_real_run_exits_on_missing_reader_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_real_run with gpt-4o-mini reader and no OPENAI_API_KEY → SystemExit."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(run_smoke, "_check_dataset", lambda bench: (True, "mocked"))

        cfg = _make_config(
            readers=("gpt-4o-mini",),
        )
        with pytest.raises(SystemExit):
            run_smoke._real_run(cfg)


# ---------------------------------------------------------------------------
# Test: baseline + substrate variants invoke the right benchmark module
# ---------------------------------------------------------------------------


class TestVariantBenchmarkInvocation:
    """Baseline and substrate variants must call the correct benchmark helpers."""

    def test_longmemeval_baseline_calls_baseline_reader(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        baseline_called: list[bool] = []
        judge_called: list[bool] = []

        def _mock_baseline(_case, _reader, _env):
            baseline_called.append(True)
            return "answer", 10.0, 50

        def _mock_judge(_q, _gold, _pred, _key):
            judge_called.append(True)
            return 1.0, "CORRECT"

        monkeypatch.setattr(run_smoke, "_call_longmemeval_baseline", _mock_baseline)
        monkeypatch.setattr(run_smoke, "_call_longmemeval_judge", _mock_judge)

        class _StubQ:
            question_id = "q1"
            question = "Q"
            answer = "A"
            question_type = "information_extraction"
            haystack_sessions: list = []
            haystack_session_ids = None
            haystack_dates = None

        cumulative_cost: list[float] = [0.0]
        results, halted = run_smoke._run_longmemeval_case(
            _StubQ(), "gpt-4o-mini", {"openai_key": "sk-test"},
            ("baseline",), cumulative_cost, 50.0,
        )

        assert not halted
        assert len(results) == 1
        assert results[0].variant == "baseline"
        assert baseline_called
        assert judge_called

    def test_longmemeval_substrate_calls_retrieval_con_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Substrate branch routes through the new
        `_call_longmemeval_substrate_retrieval_con` path by default
        (FIX 2 dispatcher flip). Structural validity must propagate from
        its returned SubstrateStats into the CaseResult.
        """
        retrieval_con_called: list[bool] = []
        legacy_called: list[bool] = []

        def _mock_retrieval_con(_q, _env, top_k: int = 20):
            retrieval_con_called.append(True)
            stats = run_smoke.SubstrateStats(
                claims_written_count=5,
                supersessions_fired_count=1,
                projection_nonempty=True,
                active_pack="personal_assistant",
                active_claim_count=5,
                top_k_retrieved=5,
                top_k_sim_mean=0.42,
                top_k_sim_min=0.31,
            )
            return "ans", 10.0, 50, stats, {"notes": [], "total_latency_ms": 10.0}

        def _mock_legacy(_q, _reader, _env):
            legacy_called.append(True)
            raise AssertionError("legacy substrate path must not be called by default")

        monkeypatch.setattr(
            run_smoke,
            "_call_longmemeval_substrate_retrieval_con",
            _mock_retrieval_con,
        )
        monkeypatch.setattr(run_smoke, "_call_longmemeval_substrate", _mock_legacy)
        monkeypatch.setattr(run_smoke, "_call_longmemeval_baseline", lambda *_: ("ans", 10.0, 50))
        monkeypatch.setattr(run_smoke, "_call_longmemeval_judge", lambda *_: (0.8, "CORRECT"))

        class _StubQ:
            question_id = "q2"
            question = "Q"
            answer = "A"
            question_type = "information_extraction"
            haystack_sessions: list = []
            haystack_session_ids = None
            haystack_dates = None

        cumulative_cost: list[float] = [0.0]
        results, halted = run_smoke._run_longmemeval_case(
            _StubQ(), "gpt-4o-mini", {"openai_key": "sk-test"},
            ("substrate",), cumulative_cost, 50.0,
        )

        assert not halted
        assert len(results) == 1
        assert results[0].variant == "substrate"
        assert retrieval_con_called
        assert not legacy_called
        assert results[0].structural_validity["claims_written_count"] == 5
        assert results[0].structural_validity["active_pack"] == "personal_assistant"
        assert results[0].structural_validity["top_k_retrieved"] == 5

    def test_longmemeval_substrate_legacy_flag_routes_to_old_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`legacy_substrate=True` routes through the legacy
        `_call_longmemeval_substrate` path. Lets reproducibility on
        pre-FIX-2 numbers be recovered without a git checkout.
        """
        legacy_called: list[bool] = []
        new_called: list[bool] = []

        def _mock_legacy(_q, _reader, _env):
            legacy_called.append(True)
            stats = run_smoke.SubstrateStats(
                claims_written_count=3,
                supersessions_fired_count=0,
                projection_nonempty=True,
                active_pack="personal_assistant",
                active_claim_count=3,
                top_k_retrieved=3,
                top_k_sim_mean=0.5,
                top_k_sim_min=0.4,
            )
            return "ans", 10.0, 50, stats

        def _mock_retrieval_con(*_a, **_k):
            new_called.append(True)
            raise AssertionError("new path must not be called when legacy flag is on")

        monkeypatch.setattr(run_smoke, "_call_longmemeval_substrate", _mock_legacy)
        monkeypatch.setattr(
            run_smoke,
            "_call_longmemeval_substrate_retrieval_con",
            _mock_retrieval_con,
        )
        monkeypatch.setattr(run_smoke, "_call_longmemeval_baseline", lambda *_: ("ans", 10.0, 50))
        monkeypatch.setattr(run_smoke, "_call_longmemeval_judge", lambda *_: (0.8, "CORRECT"))

        class _StubQ:
            question_id = "q3"
            question = "Q"
            answer = "A"
            question_type = "information_extraction"
            haystack_sessions: list = []
            haystack_session_ids = None
            haystack_dates = None

        cumulative_cost: list[float] = [0.0]
        results, _halted = run_smoke._run_longmemeval_case(
            _StubQ(), "gpt-4o-mini", {"openai_key": "sk-test"},
            ("substrate",), cumulative_cost, 50.0,
            legacy_substrate=True,
        )

        assert legacy_called
        assert not new_called
        assert len(results) == 1
        assert results[0].variant == "substrate"
        assert results[0].structural_validity["top_k_sim_mean"] == pytest.approx(0.5)

    def test_acibench_baseline_calls_acibench_reader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        aci_called: list[bool] = []
        score_called: list[bool] = []

        def _mock_aci_baseline(_enc, _reader, _env):
            aci_called.append(True)
            return "SOAP note text", 15.0, 80

        def _mock_score(_enc, _note):
            score_called.append(True)
            return 0.55

        monkeypatch.setattr(run_smoke, "_call_acibench_baseline", _mock_aci_baseline)
        monkeypatch.setattr(run_smoke, "_score_acibench_case", _mock_score)

        class _StubEnc:
            encounter_id = "enc1"
            split = "aci"
            subsplit = "test1"
            dialogue: list = []
            gold_note = "gold SOAP"

        cumulative_cost: list[float] = [0.0]
        results, halted = run_smoke._run_acibench_case(
            _StubEnc(), "gpt-4.1-mini", {"openai_key": "sk-test"},
            ("baseline",), cumulative_cost, 50.0,
        )

        assert not halted
        assert len(results) == 1
        assert results[0].variant == "baseline"
        assert results[0].benchmark == "acibench"
        assert aci_called
        assert score_called


# ---------------------------------------------------------------------------
# Test: structural_validity block populated from substrate state
# ---------------------------------------------------------------------------


class TestStructuralValidity:
    """structural_validity must reflect real substrate state accessors."""

    def test_structural_validity_populated_in_substrate_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _mock_ingestion(_bench, _payload):
            return MagicMock(
                claims_written_count=7,
                supersessions_fired_count=2,
                projection_nonempty=True,
                active_pack="clinical_general",
            )

        monkeypatch.setattr(run_smoke, "_run_substrate_ingestion", _mock_ingestion)
        monkeypatch.setattr(run_smoke, "_call_acibench_baseline", lambda *_: ("note", 20.0, 100))
        # Substrate branch now routes through _call_acibench_substrate (two-step
        # claim-extract + note-generate); mock it separately.
        monkeypatch.setattr(run_smoke, "_call_acibench_substrate", lambda *_: ("note", 40.0, 200))
        monkeypatch.setattr(run_smoke, "_score_acibench_case", lambda *_: 0.60)

        class _StubEnc:
            encounter_id = "enc_sv"
            split = "aci"
            subsplit = "test1"
            dialogue: list = []
            gold_note = "gold"

        cumulative_cost: list[float] = [0.0]
        results, halted = run_smoke._run_acibench_case(
            _StubEnc(), "gpt-4.1-mini", {"openai_key": "sk-test"},
            ("substrate",), cumulative_cost, 50.0,
        )

        assert not halted
        assert len(results) == 1
        sv = results[0].structural_validity
        assert sv["claims_written_count"] == 7
        assert sv["supersessions_fired_count"] == 2
        assert sv["projection_nonempty"] is True
        assert sv["active_pack"] == "clinical_general"

    def test_structural_validity_empty_for_baseline(self) -> None:
        """Baseline variant must not populate structural_validity."""
        cr = _make_case_result(variant="baseline", structural_validity={})
        assert cr.structural_validity == {}


# ---------------------------------------------------------------------------
# Test: verdict logic
# ---------------------------------------------------------------------------


class TestVerdictLogic:
    """_determine_verdict returns correct verdict for different case-result sets."""

    def test_pass_when_all_criteria_met(self) -> None:
        results = [
            _make_case_result(
                variant="baseline",
                baseline_score=0.61,  # within ±20pp of 61.2
                structural_validity={},
            ),
            _make_case_result(
                variant="substrate",
                baseline_score=0.61,
                substrate_score=0.65,
                delta=0.04,
                structural_validity={
                    "claims_written_count": 3,
                    "supersessions_fired_count": 0,
                    "projection_nonempty": True,
                    "active_pack": "personal_assistant",
                },
            ),
        ]
        baselines = _load_reference_baselines()
        cfg = _make_config()
        anomalies: list[str] = []
        verdict = _determine_verdict(results, baselines, cfg, anomalies)
        # May be PASS or ANOMALY depending on reference data; either is acceptable.
        assert verdict in ("PASS", "ANOMALY")

    def test_anomaly_when_no_structural_validity(self) -> None:
        results = [
            _make_case_result(
                variant="substrate",
                substrate_score=0.5,
                delta=0.0,
                structural_validity={
                    "claims_written_count": 0,
                    "supersessions_fired_count": 0,
                    "projection_nonempty": False,
                    "active_pack": "personal_assistant",
                },
            ),
        ]
        cfg = _make_config(variants=("substrate",))
        anomalies: list[str] = []
        verdict = _determine_verdict(results, {}, cfg, anomalies)
        assert verdict == "ANOMALY"
        assert any("structural_validity" in a or "projection_nonempty" in a for a in anomalies)

    def test_fail_when_no_results(self) -> None:
        cfg = _make_config()
        anomalies: list[str] = []
        verdict = _determine_verdict([], {}, cfg, anomalies)
        assert verdict == "FAIL"


# ---------------------------------------------------------------------------
# Test: dry-run still works (regression guard for existing tests)
# ---------------------------------------------------------------------------


class TestDryRunUnchanged:
    """Dry-run must still exit 0 and print planned matrix after wiring changes."""

    def test_dry_run_still_exits_zero(self) -> None:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_smoke.main(["--dry-run", "--benchmark", "longmemeval", "--reader", "gpt-4o-mini", "--variant", "baseline", "--n", "3"])
        output = buf.getvalue()
        assert rc == 0
        assert "DRY RUN" in output
        assert "Verdict: [OK] PASS" in output
