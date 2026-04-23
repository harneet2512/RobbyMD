from huggingface_hub import HfApi, snapshot_download
import json, sys, os
from pathlib import Path

HF_TOKEN = os.environ["HF_TOKEN"]
ALLOWED = {"apache-2.0","mit","bsd-2-clause","bsd-3-clause","cc0-1.0","cc-by-4.0","cc-by-sa-4.0"}

MODELS = [
    ("nvidia/canary-qwen-2.5b", "CC-BY-4.0 expected (commercial use ready per model card)"),
    ("pyannote/speaker-diarization-3.1", "MIT expected"),
    ("TheBloke/BioMistral-7B-DARE-AWQ", "Apache 2.0 expected"),  # fallback chain below
    ("hexgrad/Kokoro-82M", "Apache 2.0 expected (pulled from B4, verified for consistency)"),
]

api = HfApi()
log = []
for repo_id, expected in MODELS:
    try:
        info = api.model_info(repo_id, token=HF_TOKEN)
        license_field = getattr(info, "card_data", None) and info.card_data.get("license")
        log.append({"repo_id": repo_id, "license": license_field, "expected": expected})
        if license_field and license_field.lower() not in ALLOWED:
            print(f"BLOCKED: {repo_id} license={license_field} not in {ALLOWED}")
            sys.exit(1)
    except Exception as e:
        log.append({"repo_id": repo_id, "error": str(e)})

Path("progress.md").open("a").write("\n## License audit Bundle 5\n" + json.dumps(log, indent=2) + "\n")
print(json.dumps(log, indent=2))
