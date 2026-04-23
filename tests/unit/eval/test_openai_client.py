"""Unit tests for the shared Azure / OpenAI client factory.

Does not hit any network — we mock the `openai` package's `AzureOpenAI`
and `OpenAI` constructors and assert we wire the right arguments.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from eval._openai_client import make_openai_client


class TestDirectOpenAIBranch:
    def test_judge_returns_default_model(self) -> None:
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch("openai.OpenAI", return_value=SimpleNamespace(marker="direct")) as openai_cls:
            client, model = make_openai_client("judge_gpt4o", env=env)
        openai_cls.assert_called_once_with(api_key="sk-test")
        # Bumped 2026-04-22 from gpt-4o-2024-08-06 (retired 2026-03-31).
        # See _openai_client.py::_DIRECT_DEFAULTS comment.
        assert model == "gpt-4o-2024-11-20"
        assert getattr(client, "marker", None) == "direct"

    def test_llm_medcon_returns_gpt4omini(self) -> None:
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch("openai.OpenAI", return_value=SimpleNamespace()) as openai_cls:
            _, model = make_openai_client("llm_medcon_gpt4omini", env=env)
        openai_cls.assert_called_once_with(api_key="sk-test")
        assert model == "gpt-4o-mini"

    def test_missing_openai_key_raises(self) -> None:
        env: dict[str, str] = {}
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            make_openai_client("judge_gpt4o", env=env)


class TestAzureBranch:
    AZURE_BASE_ENV = {
        "AZURE_OPENAI_ENDPOINT": "https://my-azure.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "az-test-key",
    }

    def test_judge_routes_to_gpt4o_deployment(self) -> None:
        env = {
            **self.AZURE_BASE_ENV,
            "AZURE_OPENAI_GPT4O_DEPLOYMENT": "my-gpt4o-deployment",
        }
        captured: dict[str, Any] = {}

        def fake_azure(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(marker="azure")

        with patch("openai.AzureOpenAI", side_effect=fake_azure):
            client, model = make_openai_client("judge_gpt4o", env=env)
        assert model == "my-gpt4o-deployment"
        assert getattr(client, "marker", None) == "azure"
        assert captured["azure_endpoint"] == "https://my-azure.openai.azure.com"
        assert captured["api_key"] == "az-test-key"
        assert captured["api_version"] == "2024-10-21"

    def test_llm_medcon_routes_to_gpt4omini_deployment(self) -> None:
        env = {
            **self.AZURE_BASE_ENV,
            "AZURE_OPENAI_GPT4OMINI_DEPLOYMENT": "my-mini-deployment",
        }
        with patch("openai.AzureOpenAI", return_value=SimpleNamespace()):
            _, model = make_openai_client("llm_medcon_gpt4omini", env=env)
        assert model == "my-mini-deployment"

    def test_api_version_overridable(self) -> None:
        env = {
            **self.AZURE_BASE_ENV,
            "AZURE_OPENAI_GPT4O_DEPLOYMENT": "d",
            "AZURE_OPENAI_API_VERSION": "2024-07-01-preview",
        }
        captured: dict[str, Any] = {}

        def fake_azure(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace()

        with patch("openai.AzureOpenAI", side_effect=fake_azure):
            make_openai_client("judge_gpt4o", env=env)
        assert captured["api_version"] == "2024-07-01-preview"

    def test_missing_azure_key_raises(self) -> None:
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "AZURE_OPENAI_GPT4O_DEPLOYMENT": "d",
        }
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
            make_openai_client("judge_gpt4o", env=env)

    def test_missing_deployment_raises(self) -> None:
        env = dict(self.AZURE_BASE_ENV)
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_GPT4O_DEPLOYMENT"):
            make_openai_client("judge_gpt4o", env=env)

    def test_missing_mini_deployment_raises(self) -> None:
        env = dict(self.AZURE_BASE_ENV)
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_GPT4OMINI_DEPLOYMENT"):
            make_openai_client("llm_medcon_gpt4omini", env=env)


class TestPrefersAzureWhenBothAreSet:
    def test_azure_wins_over_direct(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-direct",
            "AZURE_OPENAI_ENDPOINT": "https://az.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "az-key",
            "AZURE_OPENAI_GPT4O_DEPLOYMENT": "azure-deployment",
        }
        with (
            patch("openai.AzureOpenAI", return_value=SimpleNamespace(marker="azure")) as az_cls,
            patch("openai.OpenAI", return_value=SimpleNamespace(marker="direct")) as direct_cls,
        ):
            client, model = make_openai_client("judge_gpt4o", env=env)
        az_cls.assert_called_once()
        direct_cls.assert_not_called()
        assert model == "azure-deployment"
        assert getattr(client, "marker", None) == "azure"
