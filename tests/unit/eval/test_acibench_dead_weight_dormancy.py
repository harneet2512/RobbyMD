"""Dead-weight dormancy regression test for the ACI-Bench hybrid path.

Stream B deliverable (2026-04-22). Components that do NOT engage on the
ACI-Bench eval path:

- hallucination guard (`src/extraction/asr/*` — ASR cleanup 5-check)
- differential trees (`src/differential/engine.py`, `src/differential/lr_table.py`)
- verifier (`src/verifier/verifier.py`)

These modules exist for the live clinical-reasoning demo but are not in
the ACI-Bench smoke/eval code path. This test pins that invariant: a
hybrid-substrate run of one case must not import, instantiate, or invoke
into any of them. If a future refactor silently wires them in (e.g., a
"reasoning assist" flag gets flipped on), this test fails loudly and the
operator can decide whether the coupling is intentional.

Approach: monkeypatch the offending modules' public entrypoints with
spies that track invocation, run one hybrid case, and assert every spy
recorded zero calls.

No network — the reader + extractor are stubbed.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.smoke import run_smoke


@pytest.fixture(autouse=True)
def _restore_active_pack():
    original = os.environ.get("ACTIVE_PACK")
    os.environ["ACTIVE_PACK"] = "clinical_general"
    try:
        from src.substrate.predicate_packs import active_pack as _ap
        _ap.cache_clear()
    except ImportError:
        pass
    yield
    if original is None:
        os.environ.pop("ACTIVE_PACK", None)
    else:
        os.environ["ACTIVE_PACK"] = original
    try:
        from src.substrate.predicate_packs import active_pack as _ap2
        _ap2.cache_clear()
    except ImportError:
        pass


def _stub_encounter() -> Any:
    class _StubEnc:
        encounter_id = "enc_dormancy"
        split = "aci"
        subsplit = "test1"
        dialogue = [
            {"speaker": "DOCTOR", "utterance": "What brings you in?"},
            {"speaker": "PATIENT", "utterance": "Chest pain."},
        ]
        gold_note = "SOAP note."

    return _StubEnc()


class TestHybridDoesNotEngageDeadWeight:
    """Hybrid ACI-Bench run must not call into differential / verifier / hallucination-guard."""

    def test_differential_engine_not_invoked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`src.differential.engine` functions are never called during hybrid run."""
        try:
            differential_engine = importlib.import_module("src.differential.engine")
        except ImportError:
            pytest.skip("differential engine not importable in this checkout")

        call_log: list[str] = []

        # Spy on every public (non-dunder) name in the module that looks
        # callable. A single invocation of any of them fails this test.
        for name in dir(differential_engine):
            if name.startswith("_"):
                continue
            obj = getattr(differential_engine, name)
            if not callable(obj):
                continue

            def _make_spy(fname: str) -> Any:
                def _spy(*_args: Any, **_kwargs: Any) -> Any:
                    call_log.append(fname)
                    raise AssertionError(
                        f"differential.engine.{fname} invoked during ACI-Bench hybrid run"
                    )
                return _spy

            monkeypatch.setattr(differential_engine, name, _make_spy(name))

        self._run_one_hybrid_case(monkeypatch)
        assert call_log == [], f"Unexpected differential engine calls: {call_log}"

    def test_verifier_not_invoked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            verifier_mod = importlib.import_module("src.verifier.verifier")
        except ImportError:
            pytest.skip("verifier module not importable in this checkout")

        call_log: list[str] = []
        for name in dir(verifier_mod):
            if name.startswith("_"):
                continue
            obj = getattr(verifier_mod, name)
            if not callable(obj):
                continue

            def _make_spy(fname: str) -> Any:
                def _spy(*_args: Any, **_kwargs: Any) -> Any:
                    call_log.append(fname)
                    raise AssertionError(
                        f"verifier.{fname} invoked during ACI-Bench hybrid run"
                    )
                return _spy

            monkeypatch.setattr(verifier_mod, name, _make_spy(name))

        self._run_one_hybrid_case(monkeypatch)
        assert call_log == [], f"Unexpected verifier calls: {call_log}"

    def test_hallucination_guard_not_imported_lazily(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ASR hallucination guard module must not be imported during a hybrid run.

        If a future change lazy-imports `src.extraction.asr.hallucination_guard`
        on the ACI-Bench path, this test fails because the module name shows
        up in `sys.modules` post-run.

        Pre-existing imports (from other test files in the same pytest session)
        are tolerated — we only check the delta introduced by the hybrid run.
        """
        guard_module_name = "src.extraction.asr.hallucination_guard"
        was_loaded_before = guard_module_name in sys.modules

        self._run_one_hybrid_case(monkeypatch)

        is_loaded_after = guard_module_name in sys.modules

        if not was_loaded_before:
            assert not is_loaded_after, (
                f"{guard_module_name} was imported by the hybrid ACI-Bench path; "
                "dead-weight dormancy broken"
            )

    def test_lr_table_not_queried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The LR table should not be loaded during ACI-Bench hybrid runs.

        Differential branches / LR weights are clinical-reasoning demo
        machinery. The hybrid SOAP generation path has no reason to query
        them.
        """
        try:
            lr_mod = importlib.import_module("src.differential.lr_table")
        except ImportError:
            pytest.skip("lr_table module not importable in this checkout")

        call_log: list[str] = []
        for name in dir(lr_mod):
            if name.startswith("_"):
                continue
            obj = getattr(lr_mod, name)
            if not callable(obj):
                continue

            def _make_spy(fname: str) -> Any:
                def _spy(*_args: Any, **_kwargs: Any) -> Any:
                    call_log.append(fname)
                    raise AssertionError(
                        f"differential.lr_table.{fname} invoked during hybrid run"
                    )
                return _spy

            monkeypatch.setattr(lr_mod, name, _make_spy(name))

        self._run_one_hybrid_case(monkeypatch)
        assert call_log == [], f"Unexpected lr_table calls: {call_log}"

    # ------------------------------------------------------------------
    # Helper — run one hybrid substrate case with stubs in place.
    # ------------------------------------------------------------------

    def _run_one_hybrid_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run `_call_acibench_substrate` once, fully stubbed."""
        # Stub reader → deterministic note.
        monkeypatch.setattr(
            run_smoke, "_call_qwen", lambda *_, **__: ("SOAP note content.", 50)
        )
        # Stub extractor → zero claims (we don't need real extraction to
        # prove dead-weight dormancy; even the populated-scaffold path
        # doesn't call into differential/verifier).
        def _stub_extractor_factory() -> Any:
            def _noop(_turn: Any) -> list:
                return []
            return _noop

        monkeypatch.setattr(
            "src.extraction.claim_extractor.extractor.make_llm_extractor",
            _stub_extractor_factory,
        )

        run_smoke._call_acibench_substrate(
            _stub_encounter(),
            reader="qwen2.5-14b",
            reader_env={"endpoint": "http://mock"},
            baseline_note_for_edit_distance=None,
            hybrid=True,
        )
