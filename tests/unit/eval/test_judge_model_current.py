"""Locks the judge model to a non-deprecated snapshot.

`gpt-4o-2024-08-06` was retired 2026-03-31. Any code path still resolving to
that model would 404 against current OpenAI / Azure endpoints. Worker 1
(fix/benchmark-integrity) bumped `judge_gpt4o`, `longmemeval_reader`, and
`longmemeval_judge` to `gpt-4o-2024-11-20`. This test ensures none of those
purposes silently regress to the deprecated snapshot.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from eval._openai_client import make_openai_client


_DEPRECATED = "gpt-4o-2024-08-06"
_CURRENT = "gpt-4o-2024-11-20"


def test_judge_gpt4o_is_not_deprecated() -> None:
    env = {"OPENAI_API_KEY": "sk-test"}
    with patch("openai.OpenAI", return_value=SimpleNamespace()):
        _, model = make_openai_client("judge_gpt4o", env=env)
    assert model != _DEPRECATED, (
        f"judge_gpt4o regressed to retired model {_DEPRECATED}; "
        f"calls to OpenAI / Azure with this model id will return 404."
    )
    assert model == _CURRENT


def test_longmemeval_reader_is_not_deprecated() -> None:
    env = {"OPENAI_API_KEY": "sk-test"}
    with patch("openai.OpenAI", return_value=SimpleNamespace()):
        _, model = make_openai_client("longmemeval_reader", env=env)
    assert model != _DEPRECATED
    assert model == _CURRENT


def test_longmemeval_judge_is_not_deprecated() -> None:
    env = {"OPENAI_API_KEY": "sk-test"}
    with patch("openai.OpenAI", return_value=SimpleNamespace()):
        _, model = make_openai_client("longmemeval_judge", env=env)
    assert model != _DEPRECATED
    assert model == _CURRENT
