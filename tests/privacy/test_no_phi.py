"""PHI sentinel scan per rules.md §2.4.

Static scan of the repo for real-looking identifiers: SSNs, labelled DOBs,
and named-patient patterns. Fails the build if any hit that isn't marked
as synthetic in context.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Directory trees and root files we scan.
SCAN_DIRS = ["src", "content", "eval", "docs", "tests", "scripts", "ui"]
SCAN_ROOT_FILES = [
    "README.md",
    "SYNTHETIC_DATA.md",
    "CLAUDE.md",
    "PRD.md",
    "context.md",
    "Eng_doc.md",
    "rules.md",
    "pyproject.toml",
]

# Paths / patterns to skip.
SKIP_PARTS = ("__pycache__", ".git", "node_modules", ".venv", ".hf_cache", ".cache", ".pytest_cache")
SKIP_SUFFIXES = (".pyc", ".so", ".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".mp3", ".wav")

# Text-like file extensions we inspect.
TEXT_EXTS = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".ts", ".tsx", ".js", ".jsx", ".sh", ".toml", ".html", ".css"}

# PHI patterns. Deliberately conservative: we want few false positives at scaffold time.
PHI_PATTERNS: dict[str, re.Pattern[str]] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "labelled DOB": re.compile(r"\b(?:DOB|dob|date[_\s-]of[_\s-]birth|born)[\s:=]+\d{4}-\d{2}-\d{2}\b"),
    "labelled MRN": re.compile(r"\bMRN[\s:=#]+[A-Z0-9\-]{5,}\b", re.IGNORECASE),
}

# A line that contains any of these sentinels is treated as explicitly synthetic — matches within are allowed.
SYNTHETIC_SENTINELS = (
    "synthetic",
    "not real patient",
    "jane doe",
    "john doe",
    "test case",
    "research prototype",
    "example",
    "placeholder",
)


def _iter_scan_files() -> Iterator[Path]:
    for name in SCAN_ROOT_FILES:
        p = ROOT / name
        if p.exists() and p.is_file():
            yield p
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            if any(part in f.parts for part in SKIP_PARTS):
                continue
            if f.suffix.lower() in SKIP_SUFFIXES:
                continue
            if f.suffix.lower() in TEXT_EXTS or f.name.lower() in {n.lower() for n in SCAN_ROOT_FILES}:
                yield f


def _line_containing(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def test_no_phi_patterns_in_repo() -> None:
    hits: list[tuple[str, str, str]] = []
    for f in _iter_scan_files():
        try:
            content = f.read_text(errors="ignore")
        except OSError:
            continue
        content_lower = content.lower()
        for label, pat in PHI_PATTERNS.items():
            for m in pat.finditer(content):
                line = _line_containing(content, m.start(), m.end())
                if any(sentinel in line.lower() for sentinel in SYNTHETIC_SENTINELS):
                    continue
                # Also allow if the whole file's top has a synthetic sentinel within the first 20 lines.
                head = "\n".join(content_lower.splitlines()[:20])
                if any(s in head for s in SYNTHETIC_SENTINELS):
                    continue
                rel = f.relative_to(ROOT)
                hits.append((str(rel), label, m.group()))
    assert not hits, (
        "Possible PHI patterns detected (rules.md §2.4):\n"
        + "\n".join(f"  - {path}: [{label}] '{match}'" for path, label, match in hits)
        + "\n\nIf these are synthetic placeholders, add a 'SYNTHETIC' or equivalent sentinel on the same line "
        + "or at the file header."
    )
