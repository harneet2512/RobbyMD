"""Model-attribution gate per rules.md §1.2.

For any model weight referenced in `src/`, the identifier must appear in
`MODEL_ATTRIBUTIONS.md`. This makes CC-BY-* attribution load-bearing in
CI — no silent drift.

Enforces rules.md §1.2 model-weight + dataset clause (2026-04-21):
open-data licences (CC-BY-4.0, CC-BY-SA-4.0, CDLA-Permissive-2.0, ODbL)
permitted for model weights with attribution in MODEL_ATTRIBUTIONS.md.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
ATTRIBUTIONS = ROOT / "MODEL_ATTRIBUTIONS.md"

# Model-load call-site patterns. Capture group 1 = model identifier string.
# Conservative: require quoted string literals; skip dynamic f-strings to
# avoid false positives.
PATTERNS: tuple[re.Pattern[str], ...] = (
    # huggingface_hub.snapshot_download(repo_id="org/name")
    re.compile(r'snapshot_download\s*\(\s*(?:repo_id\s*=\s*)?["\']([^"\']+)["\']'),
    # AutoModel.from_pretrained / AutoTokenizer.from_pretrained / Pipeline.from_pretrained
    re.compile(r'\.from_pretrained\s*\(\s*["\']([^"\']+)["\']'),
    # faster-whisper: WhisperModel("large-v3") or WhisperModel("openai/whisper-large-v3")
    re.compile(r'WhisperModel\s*\(\s*["\']([^"\']+)["\']'),
    # whisperx.load_model("large-v3")
    re.compile(r'whisperx\.load_model\s*\(\s*["\']([^"\']+)["\']'),
    # sentence-transformers: SentenceTransformer("intfloat/e5-small-v2")
    re.compile(r'SentenceTransformer\s*\(\s*["\']([^"\']+)["\']'),
)

# Identifiers that look like model IDs but aren't (common false-positive strings in
# model-load signatures — e.g. file paths, cache dirs). Matched literally and skipped.
SKIP_IDENTIFIERS: frozenset[str] = frozenset({
    "path",
    "model",
    "cache_dir",
    "device",
    "torch",
    "cpu",
    "cuda",
})

# Whisper base names (without org prefix) that map to openai/whisper-*. faster-whisper
# accepts both forms. Alias so "large-v3" in code matches "openai/whisper-large-v3" in
# MODEL_ATTRIBUTIONS.md without requiring each call site to fully qualify.
WHISPER_ALIASES: dict[str, str] = {
    "large-v3": "openai/whisper-large-v3",
    "large-v2": "openai/whisper-large-v2",
    "medium": "openai/whisper-medium",
    "small": "openai/whisper-small",
    "base": "openai/whisper-base",
    "tiny": "openai/whisper-tiny",
    "distil-large-v3": "distil-whisper/distil-large-v3",
    "distil-large-v2": "distil-whisper/distil-large-v2",
}


def _discover_model_identifiers() -> set[str]:
    """Scan src/**/*.py for model identifiers referenced in model-load calls."""
    identifiers: set[str] = set()
    if not SRC.exists():
        return identifiers
    for py in SRC.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in PATTERNS:
            for m in pat.finditer(text):
                ident = m.group(1).strip()
                if not ident or ident in SKIP_IDENTIFIERS or ident.startswith(("/", ".")):
                    continue
                # Normalise Whisper aliases so "large-v3" matches the attribution row.
                identifiers.add(WHISPER_ALIASES.get(ident, ident))
    return identifiers


def _declared_identifiers() -> set[str]:
    """Parse MODEL_ATTRIBUTIONS.md for declared model identifiers.

    Accepts identifiers in `org/name` form (HuggingFace style) and plain names inside
    backticks. Both appear in the table's Model column; being liberal here avoids
    false positives in the test.
    """
    if not ATTRIBUTIONS.exists():
        return set()
    text = ATTRIBUTIONS.read_text(encoding="utf-8", errors="ignore")
    ids: set[str] = set()
    # HF-style org/name anywhere in the file (covers table cells, source URLs, prose).
    for m in re.finditer(r"([A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+)", text):
        ids.add(m.group(1))
    # Plain names in backticks (to cover models with no org prefix, e.g. `silero-vad`).
    for m in re.finditer(r"`([A-Za-z0-9_\-\./]+)`", text):
        ids.add(m.group(1))
    return ids


def test_model_attributions_file_exists_with_required_columns() -> None:
    """MODEL_ATTRIBUTIONS.md exists at repo root with the expected table header."""
    assert ATTRIBUTIONS.exists(), f"MODEL_ATTRIBUTIONS.md missing at {ATTRIBUTIONS}"
    text = ATTRIBUTIONS.read_text(encoding="utf-8", errors="ignore")
    for column in ("Model identifier", "License", "Attribution line"):
        assert column in text, (
            f"MODEL_ATTRIBUTIONS.md missing expected column: {column!r}. "
            "See rules.md §1.2 enforcement note."
        )


def test_every_model_load_is_attributed() -> None:
    """Every model identifier loaded in src/ appears in MODEL_ATTRIBUTIONS.md."""
    referenced = _discover_model_identifiers()
    declared = _declared_identifiers()
    missing = sorted(ident for ident in referenced if ident not in declared)
    assert not missing, (
        "Model identifiers referenced in src/ but NOT declared in MODEL_ATTRIBUTIONS.md:\n"
        + "\n".join(f"  - {ident}" for ident in missing)
        + "\n\n"
        + "Per rules.md §1.2, every model weight loaded from code must have an\n"
        + "entry in MODEL_ATTRIBUTIONS.md with its license and attribution line.\n"
        + "Add the row BEFORE the commit that introduces the load call."
    )
