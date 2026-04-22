"""Shared Azure/OpenAI client factory for eval-time callers.

Every eval-time LLM call goes through this module so Azure routing is
consistent across the judge loop, the concept extractor, and any future
reader. `openai.AzureOpenAI` and `openai.OpenAI` share the same
`.chat.completions.create` shape — the only caller-visible difference
is that Azure takes the *deployment name* where direct OpenAI takes
the model name.

Routing rule
------------
If `AZURE_OPENAI_ENDPOINT` is set in the environment, route to Azure
(requires `AZURE_OPENAI_API_KEY` and the per-purpose deployment-name
env var). Otherwise fall back to direct OpenAI (`OPENAI_API_KEY`) with
a stable default model id.

Purposes and corresponding env vars
-----------------------------------
- `"judge_gpt4o"`:
    Azure deployment: `AZURE_OPENAI_GPT4O_DEPLOYMENT`
    Direct OpenAI model: `gpt-4o-2024-08-06`
    Used by the LongMemEval-S official judge loop.

- `"llm_medcon_gpt4omini"`:
    Azure deployment: `AZURE_OPENAI_GPT4OMINI_DEPLOYMENT`
    Direct OpenAI model: `gpt-4o-mini`
    Used by `eval.aci_bench.llm_medcon.LLMMedconExtractor`.

Never called from the demo path (`src/`). Demo-path LLM calls route to
Anthropic / Opus 4.7 per `Eng_doc.md §3.5`.
"""

from __future__ import annotations

import os
from typing import Any, Final, Literal

Purpose = Literal["judge_gpt4o", "llm_medcon_gpt4omini"]

# Default models for the direct-OpenAI branch. Keep in sync with
# methodology.md's model-usage policy.
_DIRECT_DEFAULTS: Final[dict[str, str]] = {
    "judge_gpt4o": "gpt-4o-2024-08-06",
    "llm_medcon_gpt4omini": "gpt-4o-mini",
}

# Per-purpose Azure deployment env-var keys.
_AZURE_DEPLOYMENT_ENVS: Final[dict[str, str]] = {
    "judge_gpt4o": "AZURE_OPENAI_GPT4O_DEPLOYMENT",
    "llm_medcon_gpt4omini": "AZURE_OPENAI_GPT4OMINI_DEPLOYMENT",
}

# Default Azure API version — picks the most recent stable that supports
# json_object response format. Overridable via AZURE_OPENAI_API_VERSION.
_DEFAULT_AZURE_API_VERSION: Final[str] = "2024-10-21"


def make_openai_client(
    purpose: Purpose,
    env: dict[str, str] | None = None,
) -> tuple[Any, str]:
    """Return (client, model_or_deployment) ready for `chat.completions.create`.

    `env` is injectable for tests; production callers pass `None` and the
    helper reads `os.environ`.

    Raises `RuntimeError` with an actionable message when required env
    vars are missing. The caller is expected to let this propagate so
    operator sees the exact missing variable.
    """
    env = dict(os.environ) if env is None else env

    azure_endpoint = env.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if azure_endpoint:
        azure_key = env.get("AZURE_OPENAI_API_KEY", "").strip()
        if not azure_key:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT is set but AZURE_OPENAI_API_KEY is missing. "
                "Set both before invoking the eval harness."
            )
        deployment_env = _AZURE_DEPLOYMENT_ENVS[purpose]
        deployment = env.get(deployment_env, "").strip()
        if not deployment:
            raise RuntimeError(
                f"{deployment_env} is not set — required for Azure routing of {purpose!r}."
            )
        api_version = env.get("AZURE_OPENAI_API_VERSION", _DEFAULT_AZURE_API_VERSION)
        try:
            from openai import AzureOpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "openai package not importable; add `openai>=1.0` to runtime deps."
            ) from e
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=api_version,
        )
        return client, deployment

    # Direct OpenAI fallback.
    direct_key = env.get("OPENAI_API_KEY", "").strip()
    if not direct_key:
        raise RuntimeError(
            f"Neither AZURE_OPENAI_ENDPOINT nor OPENAI_API_KEY is set — "
            f"at least one is required for purpose={purpose!r}."
        )
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "openai package not importable; add `openai>=1.0` to runtime deps."
        ) from e
    client = OpenAI(api_key=direct_key)
    return client, _DIRECT_DEFAULTS[purpose]
