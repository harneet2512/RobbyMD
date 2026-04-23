"""Tests for the LongMemEval purposes + gpt-4.1 fallback codepath.

The fallback exists so Stream A can run end-to-end before the operator
deploys `gpt-4o-2024-08-06` on a spare Azure account. When the fallback
fires, a structlog WARN must carry the methodology deviation so operator
logs show it clearly.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from eval._openai_client import (
    LME_CONCURRENT_DEFAULT,
    _DIRECT_DEFAULTS,
    make_openai_client,
)


class TestDirectOpenAIBranch:
    def test_lme_reader_direct_returns_gpt4o(self) -> None:
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch("openai.OpenAI", return_value=SimpleNamespace(marker="direct")):
            _, model = make_openai_client("longmemeval_reader", env=env)
        # Bumped 2026-04-22 from gpt-4o-2024-08-06 (retired 2026-03-31).
        assert model == "gpt-4o-2024-11-20"

    def test_lme_judge_direct_returns_gpt4o(self) -> None:
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch("openai.OpenAI", return_value=SimpleNamespace()):
            _, model = make_openai_client("longmemeval_judge", env=env)
        assert model == "gpt-4o-2024-11-20"


class TestAzureWithLmeDeployment:
    """Happy path — the operator has deployed gpt-4o-2024-08-06 on a spare Azure."""

    def test_reader_routes_to_lme_deployment(self) -> None:
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://lme.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_GPT4O_LME_DEPLOYMENT": "gpt-4o-2024-08-06",
        }
        with patch("openai.AzureOpenAI", return_value=SimpleNamespace(marker="lme")):
            _, deployment = make_openai_client("longmemeval_reader", env=env)
        assert deployment == "gpt-4o-2024-08-06"

    def test_judge_routes_to_lme_deployment(self) -> None:
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://lme.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_GPT4O_LME_DEPLOYMENT": "gpt-4o-2024-08-06",
        }
        with patch("openai.AzureOpenAI", return_value=SimpleNamespace()):
            _, deployment = make_openai_client("longmemeval_judge", env=env)
        assert deployment == "gpt-4o-2024-08-06"


class TestAzureFallbackToGpt41:
    """Fallback path — LME deployment unset, judge (gpt-4.1) deployment set."""

    def test_reader_falls_back_with_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://existing.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "az-key",
            # LME deployment INTENTIONALLY absent
            "AZURE_OPENAI_GPT4O_DEPLOYMENT": "my-gpt41-deployment",
        }
        captured: list[dict[str, Any]] = []

        def fake_warning(event: str, **kwargs: Any) -> None:
            captured.append({"event": event, **kwargs})

        with patch("openai.AzureOpenAI", return_value=SimpleNamespace(marker="fb")):
            with patch("eval._openai_client.log.warning", side_effect=fake_warning):
                _, deployment = make_openai_client("longmemeval_reader", env=env)

        # Falls back to gpt-4.1 deployment.
        assert deployment == "my-gpt41-deployment"
        # WARN emitted with the methodology note.
        assert captured, "expected a structlog WARN when falling back"
        warn = captured[0]
        assert warn["event"] == "longmemeval.gpt4o_unavailable_fallback_to_gpt41"
        assert warn["purpose"] == "longmemeval_reader"
        assert warn["expected_env"] == "AZURE_OPENAI_GPT4O_LME_DEPLOYMENT"
        assert warn["fallback_env"] == "AZURE_OPENAI_GPT4O_DEPLOYMENT"
        assert "gpt-4o-2024-08-06" in warn["methodology_note"]
        assert "fallback" in warn["methodology_note"].lower()

    def test_judge_falls_back_with_warn(self) -> None:
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://existing.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_GPT4O_DEPLOYMENT": "gpt41-dep",
        }
        captured: list[dict[str, Any]] = []

        def fake_warning(event: str, **kwargs: Any) -> None:
            captured.append({"event": event, **kwargs})

        with patch("openai.AzureOpenAI", return_value=SimpleNamespace()):
            with patch("eval._openai_client.log.warning", side_effect=fake_warning):
                _, deployment = make_openai_client("longmemeval_judge", env=env)

        assert deployment == "gpt41-dep"
        assert captured[0]["purpose"] == "longmemeval_judge"


class TestAzureFallbackAbsent:
    """Neither LME deployment nor gpt-4.1 judge deployment set → actionable error."""

    def test_missing_everything_raises(self) -> None:
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "az-key",
        }
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_GPT4O_LME_DEPLOYMENT"):
            make_openai_client("longmemeval_reader", env=env)


class TestConcurrencyConstant:
    def test_default_is_five(self) -> None:
        # Default concurrency is 5 unless LME_CONCURRENT_REQUESTS is set.
        assert LME_CONCURRENT_DEFAULT >= 1


class TestDirectDefaultsTable:
    def test_lme_purposes_are_gpt4o(self) -> None:
        # Bumped 2026-04-22 from gpt-4o-2024-08-06 (retired 2026-03-31).
        assert _DIRECT_DEFAULTS["longmemeval_reader"] == "gpt-4o-2024-11-20"
        assert _DIRECT_DEFAULTS["longmemeval_judge"] == "gpt-4o-2024-11-20"
