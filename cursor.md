# cursor.md — Project Rules for Claude / Cursor

Rules that govern how an AI assistant works on this repo. Append-only;
do not delete entries.

---

## Rule 2 — "WisprFlow" means RobbyMD's two-speaker voice/ASR component

**Effective**: 2026-04-22

When the user (or any project doc going forward) says **WisprFlow**, that
refers to **RobbyMD's two-speaker voice / ASR / clinical-conversation
pipeline** — the in-house feature — not the external commercial product
at wisprflow.ai.

### Components this name covers
- `src/extraction/asr/` pipeline (Whisper large-v3 + WhisperX +
  pyannote diarization + transcript cleaning + hallucination guard +
  word correction)
- Live-mic demo path
- `docs/asr_engineering_spec.md` (the engineering spec for that pipeline)
- The demo-video voice-capture story

### Why
User-chosen internal naming convention.

### How to apply
- Always refer to the two-speaker voice / ASR pipeline as **WisprFlow** in
  conversation, file names I create, doc updates I write, commit messages,
  any artifact I generate.
- Note the contradiction with `docs/asr_engineering_spec.md §8.3` which
  positions external WisprFlow as the competitor. When editing that doc
  (or any existing doc that uses the old framing), surface the conflict
  to the user first — don't silently flip the meaning in committed text.
- When uncertain, default to internal-WisprFlow and ask for clarification.

---

## Rule 1 — Dual-write all generated files (repo + mirror)

**Effective**: 2026-04-22

Every file I create or modify for this project must be written to
**both** locations:

1. The intended path inside the repo working tree
2. A mirrored copy under the directory pointed to by the environment
   variable `ROBBYMD_MIRROR_DIR`

The mirror preserves the same relative folder structure
(e.g. a file at `<repo>/advisory/validation/research_report.md`
mirrors to
`${ROBBYMD_MIRROR_DIR}/advisory/validation/research_report.md`).

> Operator's local resolution (not committed to git): on this machine
> `ROBBYMD_MIRROR_DIR` resolves to a personal Downloads subdirectory.
> Set it in your shell rc (`export ROBBYMD_MIRROR_DIR=...`) before
> starting an agent session. If the variable is unset, agents should
> ask the operator before mirroring (do not guess a path).

### Why
The operator wants a local mirror of every documentation artifact so
they can inspect / share / archive without going through the repo
path.

### How to apply
- After every successful `Write` or `Edit` to any documentation file
  in this project, immediately copy the file to the corresponding
  path under `$ROBBYMD_MIRROR_DIR`.
- Use `cp` (bash) preserving the relative subdirectory:
  ```bash
  mkdir -p "$ROBBYMD_MIRROR_DIR/<relative_dir>"
  cp "<source>" "$ROBBYMD_MIRROR_DIR/<relative_dir>/"
  ```
- For folders created from scratch, copy the whole folder with
  `cp -r`.
- Skip mirroring for: code in `src/`, tests in `tests/`, generated
  eval artifacts in `eval/*/results/` (these belong only in the
  repo). Mirror **all** documentation (`.md`), plans, decision
  memos, and analysis artifacts.
- If unsure whether something should mirror, default to mirroring it.

### Scope
- Applies to **every future file** created or edited inside the repo
  working tree, not just one-off requests.
- Applies until the operator explicitly rescinds this rule.

---
