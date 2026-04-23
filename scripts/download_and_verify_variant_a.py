"""Download and licence-verify every model Variant A uses.

Reads card_data.license from the HF model card for each repo, blocks the run
if any licence is outside the allow-list (OSI-approved for code + CC-BY-4.0 /
CC-BY-SA-4.0 for data/model-weights, per
docs/decisions/licensing_clarifications.md §Q2).

Appends an audit JSON block to progress.md.

Walks a fallback chain for BioMistral quants because the AWQ variant is not
universally mirrored.

Run from the repo root on the L4:
    HF_TOKEN=... python scripts/download_and_verify_variant_a.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ALLOWED = {
    # OSI-approved code
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "mpl-2.0",
    "lgpl-2.1",
    "lgpl-3.0",
    # Open-data (per repo ADR for model weights)
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cdla-permissive-2.0",
    "odbl",
}

# Ordered fallbacks for the cleanup LLM. Walk until one succeeds.
BIOMISTRAL_CANDIDATES = [
    "TheBloke/BioMistral-7B-DARE-AWQ",
    "LoneStriker/BioMistral-7B-DARE-AWQ",
    "LoneStriker/BioMistral-7B-DARE-GPTQ",
    "BioMistral/BioMistral-7B-DARE",
    "BioMistral/BioMistral-7B-SLERP-AWQ",
    "BioMistral/BioMistral-7B-SLERP",
]

FIXED_MODELS = [
    ("openai/whisper-large-v3-turbo", "Whisper-turbo ASR (MIT expected)"),
    (
        "pyannote/speaker-diarization-community-1",
        "pyannote diariser (MIT code + CC-BY-4.0 weights expected)",
    ),
    ("hexgrad/Kokoro-82M", "Kokoro TTS (Apache-2.0 expected)"),
]


def _licence_of(api: HfApi, repo_id: str, token: str) -> str | None:
    info = api.model_info(repo_id, token=token)
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    lic = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    return lic.lower() if isinstance(lic, str) else None


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set")
        sys.exit(2)

    api = HfApi()
    audit: list[dict] = []

    # 1. Fixed models — must each succeed.
    for repo_id, note in FIXED_MODELS:
        lic = _licence_of(api, repo_id, token)
        entry: dict[str, object] = {"repo_id": repo_id, "licence": lic, "note": note}
        if lic is None:
            entry["status"] = "no_licence_in_card — checking parents / manual review"
            print(f"WARN no licence on card: {repo_id}", flush=True)
        elif lic not in ALLOWED:
            entry["status"] = f"blocked (licence {lic!r} not in allow-list)"
            audit.append(entry)
            print(f"BLOCKED: {repo_id} licence={lic}", flush=True)
            _append_audit(audit)
            sys.exit(1)
        else:
            entry["status"] = "ok"
        audit.append(entry)

    # 2. BioMistral — walk fallbacks.
    chosen: str | None = None
    for repo_id in BIOMISTRAL_CANDIDATES:
        try:
            lic = _licence_of(api, repo_id, token)
        except Exception as e:
            audit.append(
                {"repo_id": repo_id, "status": f"fetch_failed: {e}", "licence": None}
            )
            continue
        entry = {"repo_id": repo_id, "licence": lic}
        if lic is None or lic in ALLOWED:
            entry["status"] = "candidate_ok"
            audit.append(entry)
            chosen = repo_id
            break
        entry["status"] = f"blocked (licence {lic!r} not in allow-list)"
        audit.append(entry)

    if chosen is None:
        _append_audit(audit)
        print("No BioMistral variant available — aborting", flush=True)
        sys.exit(1)

    audit.append({"biomistral_chosen": chosen})

    # 3. Actually download everything.
    for repo_id, _ in FIXED_MODELS:
        print(f"Downloading {repo_id} ...", flush=True)
        snapshot_download(repo_id, token=token)
    print(f"Downloading {chosen} ...", flush=True)
    snapshot_download(chosen, token=token)

    _append_audit(audit)
    print(json.dumps(audit, indent=2))
    print(f"\nBIOMISTRAL_REPO={chosen}", flush=True)


def _append_audit(audit: list[dict]) -> None:
    block = (
        "\n## Licence audit — Bundle 4 Variant A download\n"
        + "```json\n"
        + json.dumps(audit, indent=2)
        + "\n```\n"
    )
    Path("progress.md").open("a", encoding="utf-8").write(block)


if __name__ == "__main__":
    main()
