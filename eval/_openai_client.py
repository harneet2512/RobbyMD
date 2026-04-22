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

- `"claim_extractor_gpt4omini"`:
    Azure deployment: `AZURE_OPENAI_GPT4OMINI_DEPLOYMENT`
    Direct OpenAI model: `gpt-4o-mini`
    Used by `src.extraction.claim_extractor.extractor.make_llm_extractor`.
    Shares the gpt-4o-mini deployment with llm_medcon — different prompt,
    same model class.

- `"reader_gpt41mini"`:
    Azure deployment: `AZURE_OPENAI_GPT4OMINI_DEPLOYMENT`
    Direct OpenAI model: `gpt-4.1-mini`
    Smoke-harness reader. Same deployment as llm_medcon/claim_extractor.

- `"reader_gpt41"`:
    Azure deployment: `AZURE_OPENAI_GPT4O_DEPLOYMENT`
    Direct OpenAI model: `gpt-4.1`
    Smoke-harness reader for LongMemEval (long-context variant).

- `"longmemeval_reader"`:
    Azure deployment: `AZURE_OPENAI_GPT4O_LME_DEPLOYMENT`
    Direct OpenAI model: `gpt-4o-2024-08-06`
    Paper-faithful reader for the LongMemEval substrate-variant smoke.
    ICLR 2025 Wu et al. (arXiv 2410.10813) use `gpt-4o-2024-08-06`; this
    purpose preserves apples-to-apples with the leaderboard.

- `"longmemeval_judge"`:
    Azure deployment: `AZURE_OPENAI_GPT4O_LME_DEPLOYMENT`
    Direct OpenAI model: `gpt-4o-2024-08-06`
    Official LongMemEval judge (same model as the reader per the paper).

Fallback codepath for LongMemEval purposes
------------------------------------------
If `AZURE_OPENAI_ENDPOINT` is set AND `AZURE_OPENAI_GPT4O_LME_DEPLOYMENT`
is unset AND `AZURE_OPENAI_GPT4O_DEPLOYMENT` is set, the LongMemEval
purposes fall back to the existing gpt-4.1 deployment. A structlog WARN
is emitted documenting the methodology deviation so the operator sees
it in logs and can include it in reported numbers. This exists so that
Stream A can run end-to-end *before* the operator has deployed
`gpt-4o-2024-08-06` on the chosen Azure account — reporting numbers
under the fallback is permitted only if labelled as such (see
`methodology.md`).

Note on Azure deviation: as of 2026-04-22 the Azure subscription has
deployments for `gpt-4.1` and `gpt-4.1-mini` (not `gpt-4o` / `gpt-4o-mini`).
Operators set `AZURE_OPENAI_GPT4O_DEPLOYMENT` and
`AZURE_OPENAI_GPT4OMINI_DEPLOYMENT` to the gpt-4.1 / gpt-4.1-mini deployment
names respectively. This is a methodology deviation from the paper-specified
`gpt-4o-2024-08-06` judge — flagged in `methodology.md` and
`architecture_changes.md`.

Never called from the demo path (`src/`). Demo-path LLM calls route to
Anthropic / Opus 4.7 per `Eng_doc.md §3.5`.
"""

from __future__ import annotations

import os
from typing import Any, Final, Literal

import structlog

log = structlog.get_logger(__name__)

Purpose = Literal[
    "judge_gpt4o",
    "llm_medcon_gpt4omini",
    "claim_extractor_gpt4omini",
    "reader_gpt41mini",
    "reader_gpt41",
    "longmemeval_reader",
    "longmemeval_judge",
]

# Default concurrency for the LongMemEval reader + judge loops.
# 5 matches the plan (halve to 3 if the endpoint shows backpressure). Exposed
# as a module constant so callers can read the default without re-parsing env.
LME_CONCURRENT_DEFAULT: Final[int] = int(
    os.environ.get("LME_CONCURRENT_REQUESTS", "5") or "5"
)

# Default models for the direct-OpenAI branch. Keep in sync with
# methodology.md's model-usage policy.
_DIRECT_DEFAULTS: Final[dict[str, str]] = {
    "judge_gpt4o": "gpt-4o-2024-08-06",
    "llm_medcon_gpt4omini": "gpt-4o-mini",
    "claim_extractor_gpt4omini": "gpt-4o-mini",
    "reader_gpt41mini": "gpt-4.1-mini",
    "reader_gpt41": "gpt-4.1",
    "longmemeval_reader": "gpt-4o-2024-08-06",
    "longmemeval_judge": "gpt-4o-2024-08-06",
}

# Per-purpose Azure deployment env-var keys.
_AZURE_DEPLOYMENT_ENVS: Final[dict[str, str]] = {
    "judge_gpt4o": "AZURE_OPENAI_GPT4O_DEPLOYMENT",
    "llm_medcon_gpt4omini": "AZURE_OPENAI_GPT4OMINI_DEPLOYMENT",
    "claim_extractor_gpt4omini": "AZURE_OPENAI_GPT4OMINI_DEPLOYMENT",
    "reader_gpt41mini": "AZURE_OPENAI_GPT4OMINI_DEPLOYMENT",
    "reader_gpt41": "AZURE_OPENAI_GPT4O_DEPLOYMENT",
    "longmemeval_reader": "AZURE_OPENAI_GPT4O_LME_DEPLOYMENT",
    "longmemeval_judge": "AZURE_OPENAI_GPT4O_LME_DEPLOYMENT",
}

# LongMemEval purposes fall back to the gpt-4.1 judge deployment when the
# spare gpt-4o-2024-08-06 deployment isn't configured yet. See module docstring.
_LME_FALLBACK_DEPLOYMENT_ENV: Final[str] = "AZURE_OPENAI_GPT4O_DEPLOYMENT"
_LME_PURPOSES: Final[frozenset[str]] = frozenset(
    {"longmemeval_reader", "longmemeval_judge"}
)

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
        if not deployment and purpose in _LME_PURPOSES:
            # Fallback: if the spare gpt-4o-2024-08-06 Azure deployment isn't
            # configured yet, route to the existing gpt-4.1 deployment and
            # WARN loudly so the operator captures the methodology deviation
            # in reported numbers. See module docstring for details.
            fallback_deployment = env.get(_LME_FALLBACK_DEPLOYMENT_ENV, "").strip()
            if fallback_deployment:
                log.warning(
                    "longmemeval.gpt4o_unavailable_fallback_to_gpt41",
                    purpose=purpose,
                    expected_env=deployment_env,
                    fallback_env=_LME_FALLBACK_DEPLOYMENT_ENV,
                    fallback_deployment=fallback_deployment,
                    methodology_note=(
                        "LongMemEval paper-faithful reader/judge is gpt-4o-2024-08-06 "
                        "(ICLR 2025, arXiv 2410.10813). Running under gpt-4.1 fallback; "
                        "any numbers reported must be labelled accordingly."
                    ),
                )
                deployment = fallback_deployment
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
