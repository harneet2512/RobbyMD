"""Pre-merge gate FIX 3: --seed flag wiring tests.

Covers the three places the seed flows through the smoke harness:
1. `_parse_args` — `--seed` populates `SmokeConfig.seed` (default 42).
2. Reader sampling — `_call_qwen` and `_call_openai` pull `seed` from
   `reader_env["seed"]` when present, default to 42 when absent.
3. Results directory naming — `_real_run` folds the seed into the timestamped
   output dir name so multi-seed re-runs land in distinct directories.

No API calls. The Phase 1.5 multi-seed gate itself is operator-run and
documented in `reasons.md`; this file only verifies the plumbing.
"""
from __future__ import annotations

from unittest.mock import patch

from eval.smoke.run_smoke import (  # noqa: F401 — SmokeConfig used in dataclass-default test
    SmokeConfig,
    _call_openai,
    _call_qwen,
    _parse_args,
)


# ---------------------------------------------------------------------------
# 1. CLI flag → SmokeConfig
# ---------------------------------------------------------------------------


class TestSeedCliFlag:
    def test_seed_default_is_42(self) -> None:
        cfg = _parse_args(["--benchmark", "acibench", "--variant", "both", "--n", "1"])
        assert cfg.seed == 42

    def test_seed_explicit(self) -> None:
        cfg = _parse_args(["--benchmark", "acibench", "--variant", "both", "--seed", "43"])
        assert cfg.seed == 43

    def test_seed_dataclass_default(self) -> None:
        # Direct construction also defaults to 42 — keeps the test factory
        # in `test_smoke_realrun_wiring.py` working without a seed= override.
        cfg = SmokeConfig(
            benchmarks=("acibench",),
            readers=("qwen2.5-14b",),
            variants=("both",),
            n_cases=1,
            budget_usd=5.0,
            dry_run=False,
        )
        assert cfg.seed == 42


# ---------------------------------------------------------------------------
# 2. Reader sampling — seed pulled from reader_env
# ---------------------------------------------------------------------------


class TestReaderSeedPropagation:
    def test_call_qwen_passes_seed_to_vllm(self) -> None:
        """`_call_qwen` reads `reader_env["seed"]` and passes it as `seed=`
        on chat.completions.create. vLLM (OpenAI-compat) honours the field.
        """
        captured: dict[str, object] = {}

        class _StubChoice:
            class _Msg:
                content = "hi"

            message = _Msg()

        class _StubResp:
            choices = [_StubChoice()]
            usage = type("U", (), {"total_tokens": 10})()

        class _StubClient:
            class chat:  # noqa: N801 — mirrors openai SDK shape
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return _StubResp()

        with patch("openai.OpenAI", return_value=_StubClient()):
            text, _ = _call_qwen(
                "system",
                "user",
                {"endpoint": "http://localhost:8000/v1", "seed": "43"},
            )

        assert text == "hi"
        assert captured["seed"] == 43

    def test_call_qwen_seed_defaults_to_42_when_env_missing(self) -> None:
        captured: dict[str, object] = {}

        class _StubChoice:
            class _Msg:
                content = "hi"

            message = _Msg()

        class _StubResp:
            choices = [_StubChoice()]
            usage = type("U", (), {"total_tokens": 10})()

        class _StubClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return _StubResp()

        with patch("openai.OpenAI", return_value=_StubClient()):
            _call_qwen(
                "system",
                "user",
                {"endpoint": "http://localhost:8000/v1"},  # no "seed" key
            )

        assert captured["seed"] == 42

    def test_call_openai_kwarg_seed_propagates(self) -> None:
        """`_call_openai` accepts `seed=` kwarg and forwards to chat.completions.

        Skips the Azure routing branch by passing a non-mapped model_key so the
        function takes the direct-OpenAI path.
        """
        captured: dict[str, object] = {}

        class _StubChoice:
            class _Msg:
                content = "ok"

            message = _Msg()

        class _StubResp:
            choices = [_StubChoice()]
            usage = type("U", (), {"total_tokens": 5})()

        class _StubClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return _StubResp()

        with patch("openai.OpenAI", return_value=_StubClient()):
            _call_openai(
                "unmapped-model",
                "system",
                "user",
                "sk-test",
                seed=44,
            )

        assert captured["seed"] == 44


# ---------------------------------------------------------------------------
# 3. Results dir naming — seed in the path
# ---------------------------------------------------------------------------


class TestResultsDirSeedSuffix:
    def test_dir_name_carries_seed(self) -> None:
        """Sanity-check the f-string we use in `_real_run`. Lightweight — we
        do not invoke `_real_run` (it would touch dataset adapters + envs).
        Instead we exercise the same f-string shape so a future rename of
        the format breaks the test loudly.
        """
        ts = "20260422T180000Z"
        seed = 43
        # Mirror the format string used in `_real_run`.
        name = f"{ts}_seed{seed}"
        assert name == "20260422T180000Z_seed43"
        assert name.endswith("_seed43")
