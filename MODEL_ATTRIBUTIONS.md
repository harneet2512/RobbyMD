# MODEL_ATTRIBUTIONS.md

Every model weight referenced by this repository is listed below with its license and attribution line. Per `rules.md §1.2`, model weights use **OSI-approved licenses** OR **open-data licenses** (CC-BY-4.0, CC-BY-SA-4.0, CDLA-Permissive-2.0, ODbL) permitting commercial use with attribution.

`tests/licensing/test_model_attributions.py` scans `src/` for model-load call sites and verifies every referenced model identifier appears in this table. **CI fails if any model is referenced from code without an entry here.**

**Adding a new model weight**: add a row below **before** you commit the code that loads it. Copy the attribution line exactly as stated on the model's canonical source (HuggingFace card, GitHub README). If the license is not on the allowed list in `rules.md §1.2`, open an ADR in `docs/decisions/` proposing the extension before use.

---

## Declared model weights

| Model identifier | Version | Author / Org | License | Source URL | Attribution line (verbatim) |
|---|---|---|---|---|---|
| `pyannote/speaker-diarization-community-1` | community-1 | Hervé Bredin, pyannote.audio | CC-BY-4.0 | https://huggingface.co/pyannote/speaker-diarization-community-1 | *"pyannote/speaker-diarization-community-1 by Hervé Bredin (pyannote.audio), licensed under CC-BY-4.0. Source: https://huggingface.co/pyannote/speaker-diarization-community-1"* |
| `openai/whisper-large-v3` | large-v3 | OpenAI (Radford et al.) | Apache-2.0 | https://huggingface.co/openai/whisper-large-v3 | *"Whisper large-v3 by OpenAI, Apache-2.0. Paper: Radford et al., arXiv:2212.04356."* (Apache-2.0 does not strictly require per-use attribution; listed for audit transparency.) |
| `distil-whisper/distil-large-v3` | v3 | Hugging Face (Gandhi et al.) | MIT | https://huggingface.co/distil-whisper/distil-large-v3 | *"Distil-Whisper large-v3 by Hugging Face, MIT License. Paper: Gandhi et al., arXiv:2311.00430."* (MIT does not strictly require per-use attribution; listed for audit transparency.) |
| `intfloat/e5-small-v2` | v2 | intfloat / Microsoft Research (Wang et al.) | MIT | https://huggingface.co/intfloat/e5-small-v2 | *"e5-small-v2 by intfloat, MIT License. Paper: Wang et al., arXiv:2212.03533."* (MIT does not strictly require per-use attribution; listed for audit transparency.) |
| `snakers4/silero-vad` | v5 | Silero Team | MIT | https://github.com/snakers4/silero-vad | *"silero-vad v5 by Silero Team, MIT License."* (MIT does not strictly require per-use attribution; listed for audit transparency.) |

---

## Attribution display in the app

CC-BY-* model attributions are load-bearing — they must be **visible** to the end-user:

- App header / "About & Credits" screen: displays every CC-BY-* row's attribution line verbatim.
- README "Acknowledgements" section: same.
- Demo video bumper: brief acknowledgement of model-weight licences (group acknowledgement acceptable; per-model line for CC-BY-* weights).
- `SYNTHETIC_DATA.md` references this file by name so the two manifests stay linked.

MIT / Apache-2.0 rows are listed here for audit transparency even though their licenses don't strictly require per-use attribution. Keeping every model weight on one page simplifies review and eliminates silent drift.

## Rule recap

- `rules.md §1.2` — licence allowlist for code (OSI) and model weights / datasets (OSI **or** open-data).
- Denylist applies here too: any model weight under CC-BY-NC, CC-BY-ND, CC-BY-NC-ND, Gemma Terms of Use, HAI-DEF, Llama 2/3 Community License, or other restrictive terms **cannot** appear in this table.
- Adding a row requires a matching attribution surface in the app UI (for CC-BY-* weights) before the code that loads the model can merge.
