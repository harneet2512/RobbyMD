"""Unit tests for the T0 QuickUMLS HTTP client and the factory dispatch."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.aci_bench.extractors import (
    QuickUMLSExtractor,
    RemoteQuickUMLSExtractor,
    build_extractor,
)
from eval.aci_bench.quickumls_client import extract_cuis_t0, is_t0_enabled


def test_is_t0_enabled_requires_both_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONCEPT_EXTRACTOR", raising=False)
    monkeypatch.delenv("UMLS_T0_ENDPOINT", raising=False)
    assert not is_t0_enabled()

    monkeypatch.setenv("CONCEPT_EXTRACTOR", "quickumls")
    assert not is_t0_enabled()  # endpoint still missing

    monkeypatch.setenv("UMLS_T0_ENDPOINT", "http://example.com:8000")
    assert is_t0_enabled()

    # Empty/whitespace endpoint should not count as enabled.
    monkeypatch.setenv("UMLS_T0_ENDPOINT", "   ")
    assert not is_t0_enabled()


def test_is_t0_enabled_explicit_env_dict() -> None:
    assert is_t0_enabled(
        {"CONCEPT_EXTRACTOR": "quickumls", "UMLS_T0_ENDPOINT": "http://x:8000"}
    )
    assert not is_t0_enabled({"CONCEPT_EXTRACTOR": "scispacy"})
    assert not is_t0_enabled({})


def test_extract_cuis_t0_returns_empty_on_unreachable_endpoint() -> None:
    # Bad host + short timeout — must not raise, must return empty set.
    result = extract_cuis_t0(
        "patient has chest pain",
        endpoint="http://nonexistent.invalid:8000",
        timeout=1.0,
    )
    assert result == set()


def test_extract_cuis_t0_returns_empty_when_endpoint_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UMLS_T0_ENDPOINT", raising=False)
    assert extract_cuis_t0("patient has chest pain") == set()


def test_factory_picks_remote_when_endpoint_set() -> None:
    extractor = build_extractor(
        {
            "CONCEPT_EXTRACTOR": "quickumls",
            "UMLS_T0_ENDPOINT": "http://10.0.0.1:8000",
        }
    )
    assert isinstance(extractor, RemoteQuickUMLSExtractor)
    assert extractor.endpoint == "http://10.0.0.1:8000"


def test_factory_falls_through_to_local_when_endpoint_unset(tmp_path: Path) -> None:
    # QUICKUMLS_PATH is required for the local path; the directory contents
    # are not validated at construction time (lazy QuickUMLS init).
    extractor = build_extractor(
        {
            "CONCEPT_EXTRACTOR": "quickumls",
            "QUICKUMLS_PATH": str(tmp_path),
        }
    )
    assert isinstance(extractor, QuickUMLSExtractor)


def test_factory_quickumls_without_endpoint_or_path_raises() -> None:
    with pytest.raises(RuntimeError, match="UMLS_T0_ENDPOINT.*QUICKUMLS_PATH"):
        build_extractor({"CONCEPT_EXTRACTOR": "quickumls"})
