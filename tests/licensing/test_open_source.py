"""Licensing gate per rules.md §1.2.

Every runtime + dev dependency must be OSI-approved open source.
Allowlist: MIT, Apache-2.0, BSD (2/3/4-clause), MPL-2.0, ISC, LGPL.

Claude Opus 4.7 via the Anthropic API (the `anthropic` SDK package) is the
hackathon's named sponsored tool, explicitly exempt per rules.md §1.2.

Enforcement: this test parses pyproject.toml, resolves each installed
dependency's license metadata via importlib.metadata, and fails if any
dependency declares a non-allowlisted license. Uninstalled dev-only
dependencies are reported but not fatal (they'll be checked in CI where
the full dev env is installed).
"""
from __future__ import annotations

import re
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"

# Substrings that, if found in a package's License or License classifier, indicate OSI approval.
ALLOWLIST_SUBSTRINGS = (
    "mit",
    "apache",
    "bsd",
    "mpl",
    "mozilla public license",
    "isc",
    "lgpl",
    "gnu lesser general public license",
    "python software foundation",  # PSF-2.0 is effectively permissive, used by some stdlib backports
)

# Packages that the hackathon event explicitly allows despite not-technically-OSI licensing,
# per rules.md §1.2's "sponsored-tool exception". Only the Anthropic SDK qualifies.
SPONSORED_TOOL_EXEMPT = {"anthropic"}

# Packages explicitly forbidden by rules.md §1.2 — these are the license patterns we've
# decided NOT to use (Gemma Terms, HAI-DEF, commercial ASR APIs, etc.). If any sneak in,
# fail loudly.
DENYLIST_SUBSTRINGS = (
    "gemma",
    "hai-def",
    "hai def",
    "deepgram",
    "assemblyai",
)


def _load_declared_deps() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    project = data.get("project", {})
    deps: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        deps.extend(group)
    names: list[str] = []
    for spec in deps:
        m = re.match(r"^\s*([A-Za-z0-9_\-\.]+)", spec)
        if m:
            names.append(m.group(1).lower())
    return names


def _license_signals(pkg_name: str) -> tuple[str, list[str]]:
    """Return (raw_license, classifiers_list) for an installed package, or ("", [])."""
    try:
        dist = distribution(pkg_name)
    except PackageNotFoundError:
        return "", []
    raw = dist.metadata.get("License") or dist.metadata.get("License-Expression") or ""
    classifiers = list(dist.metadata.get_all("Classifier") or [])
    return raw, classifiers


def _is_allowed(raw_license: str, classifiers: list[str]) -> bool:
    lic_haystack = (raw_license + " " + " ".join(classifiers)).lower()
    if any(d in lic_haystack for d in DENYLIST_SUBSTRINGS):
        return False
    return any(s in lic_haystack for s in ALLOWLIST_SUBSTRINGS)


def test_no_non_osi_dependencies() -> None:
    """Every installed declared dep has an OSI-allowlisted license, or is sponsored-exempt."""
    violations: list[tuple[str, str]] = []
    uninstalled: list[str] = []

    for pkg in _load_declared_deps():
        if pkg in SPONSORED_TOOL_EXEMPT:
            continue
        raw, classifiers = _license_signals(pkg)
        if not raw and not classifiers:
            uninstalled.append(pkg)
            continue
        if not _is_allowed(raw, classifiers):
            violations.append((pkg, raw or "<see classifiers>"))

    assert not violations, (
        "Non-OSI-approved dependencies detected (rules.md §1.2):\n"
        + "\n".join(f"  - {name}: {lic}" for name, lic in violations)
        + "\n\nAllowlist: MIT, Apache-2.0, BSD, MPL-2.0, ISC, LGPL."
        + "\nExempt (sponsored): " + ", ".join(sorted(SPONSORED_TOOL_EXEMPT))
    )
    # Uninstalled dev-only deps aren't fatal here — CI must run with `pip install -e .[dev]`
    # for strict enforcement. Surface the list so it's visible in local runs.
    if uninstalled:
        print(f"[licensing] uninstalled (dev env not primed): {sorted(uninstalled)}")


def test_no_denylisted_license_strings_in_pyproject() -> None:
    """Static scan of pyproject.toml for banned license indicators (cheap belt-and-braces)."""
    text = PYPROJECT.read_text().lower()
    hits = [d for d in DENYLIST_SUBSTRINGS if d in text]
    assert not hits, f"Denylisted license strings in pyproject.toml: {hits} (rules.md §1.2)"
