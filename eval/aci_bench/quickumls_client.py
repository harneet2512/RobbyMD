"""HTTP client for the T0 QuickUMLS scoring endpoint.

The local `QuickUMLSExtractor` in `extractors.py` requires a built UMLS
index on the same machine. This module is the remote variant: it talks to
a `FastAPI` server running on a GCP VM that holds the QuickUMLS index, so
operators don't need to provision a 5–6 GB index locally.

Activation: set both `CONCEPT_EXTRACTOR=quickumls` and `UMLS_T0_ENDPOINT`
in the environment. The factory in `extractors.py` then selects the
remote variant; if `UMLS_T0_ENDPOINT` is unset, the existing local-index
path applies (and `QUICKUMLS_PATH` is required as before). On any HTTP
error, `extract_cuis_t0` returns an empty set so the caller can fall
back to T1 scispaCy without raising.
"""
from __future__ import annotations

import os
from typing import Set

import httpx


def extract_cuis_t0(
    text: str,
    endpoint: str | None = None,
    timeout: float = 30.0,
) -> Set[str]:
    """Extract UMLS CUIs from text via the remote QuickUMLS T0 endpoint.

    Returns an empty set on any error. Caller decides whether to fall
    back to T1 or propagate the failure (the `RemoteQuickUMLSExtractor`
    in `extractors.py` returns the empty set verbatim, which makes
    downstream MEDCON F1 zero — `run.py` already handles that as the
    NullExtractor case).
    """
    endpoint = endpoint or os.environ.get("UMLS_T0_ENDPOINT")
    if not endpoint:
        return set()

    try:
        response = httpx.post(
            f"{endpoint.rstrip('/')}/match",
            json={"text": text},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return {m["cui"] for m in data.get("matches", []) if "cui" in m}
    except (httpx.HTTPError, KeyError, ValueError):
        return set()


def is_t0_enabled(env: dict[str, str] | None = None) -> bool:
    """True iff the environment requests T0 AND the endpoint is set."""
    env = env if env is not None else dict(os.environ)
    return (
        env.get("CONCEPT_EXTRACTOR", "").strip().lower() == "quickumls"
        and bool(env.get("UMLS_T0_ENDPOINT", "").strip())
    )
