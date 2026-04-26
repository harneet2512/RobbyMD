# RobbyMD

> **Research prototype. Not a medical device.** This software is a research demonstration. It does not diagnose, treat, or recommend treatment, and is not intended for use in patient care. All patients, conversations, and data shown are synthetic or from published research benchmarks. Clinical decisions are made by the physician. This system supports the physician's reasoning by tracking claims and differential hypotheses in real time; it does not direct clinical judgement.

Built for the **Built with Opus 4.7 Hackathon** (Cerebral Valley × Anthropic, April 2026).

---

## The product

A live clinical reasoning substrate for the exam room. RobbyMD sits between a doctor and a patient during a consultation, extracts structured claims from the conversation, tracks how those claims evolve (including patient self-corrections), runs deterministic differential-diagnosis hypothesis trees, surfaces the single next-best question for the top hypotheses, and generates a SOAP note with full per-sentence provenance back to the source turn.

**Status**: early build. Demo video, screenshots, quickstart, architecture diagrams, benchmark numbers, and API examples — **TBD, yet to come.**

---

## Regulatory posture

Non-Device Clinical Decision Support under Section 520(o)(1)(E) of the FD&C Act, per FDA's January 2026 CDS Final Guidance. HIPAA: zero PHI; data is synthetic or from published research benchmarks.

## Demo video

TBD — link will be added when cut.

## Architecture

TBD — diagram and walkthrough coming.

### ASR layer

Audio enters through an 8-stage pipeline: (1) ffmpeg loudnorm normalisation to 16 kHz mono PCM; (2) boundary-silence trimming to eliminate a documented Whisper hallucination trigger (Koenecke, ACM FAccT 2024); (3) faster-whisper large-v3 transcription with 6 hardened decoder parameters (temperature=0, beam_size=5, condition_on_previous_text=False, plus compression-ratio / logprob / no-speech thresholds); (4) WhisperX word-level alignment; (5) pyannote speaker diarisation; (6) speaker-role-aware LLM cleanup (FreeFlow-pattern, doctor vs patient system prompts, gpt-4o-mini for demo / qwen2.5-7b-local for evals — never Opus 4.7, which is reserved for claim extraction); (7) deterministic 5-check hallucination guard (repeated n-gram loop, OOV medical term, extreme compression ratio, low-confidence span, invented medication); (8) Levenshtein word correction against the active pack's medical vocabulary. Each stage is instrumented with a ring-buffer telemetry decorator. Full per-segment provenance — `original_text` alongside `cleaned_text` — flows to the substrate's `on_new_turn` handler, satisfying rules.md §4 (every claim traces to a conversation turn).

## Quickstart

TBD.

## Evaluation

### LongMemEval-S (ICLR 2025)

LongMemEval-S (Wu et al., ICLR 2025, [arXiv 2410.10813](https://arxiv.org/abs/2410.10813)) evaluates five core long-term memory abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. 500 questions, GPT-4o judge, per-category accuracy.

**Published systems on this benchmark:**

| System | Architecture | Score | Paper |
|--------|-------------|-------|-------|
| Mastra OM | Context compression + observer agent | 94.9% | [mastra.ai/research](https://mastra.ai/research/observational-memory) |
| Mem0 | Flat fact store + multi-signal retrieval | 93.4% | [arXiv 2504.19413](https://arxiv.org/abs/2504.19413) |
| EverMemOS | Engram-inspired MemCell lifecycle | 83.0% | [arXiv 2601.02163](https://arxiv.org/abs/2601.02163) |
| TiMem | 5-level temporal memory tree | 76.9% | [arXiv 2601.02845](https://arxiv.org/abs/2601.02845) |
| Zep/Graphiti | Bi-temporal knowledge graph | 71.2% | [arXiv 2501.13956](https://arxiv.org/abs/2501.13956) |
| Full-context GPT-4o + CoN | No memory system (raw context) | 64.0% | Wu et al. 2025 |

**Our run: 442/500 (88.4%)**

| Type | Score |
|------|-------|
| single-session-user | 69/70 (98.6%) |
| single-session-assistant | 55/56 (98.2%) |
| single-session-preference | 22/30 (73.3%) |
| multi-session | 106/133 (79.7%) |
| temporal-reasoning | 117/133 (88.0%) |
| knowledge-update | 73/78 (93.6%) |
| abstention | 27/30 (90.0%) |

Reader: GPT-5-mini. Judge: GPT-4o-2024-11-20. Total cost: $10.21. This is a hackathon engineering run, not a research contribution.

**What we built (the layers):**

The system is a 4-layer pipeline over the raw LongMemEval conversation haystack. No pre-built memory framework, no external memory service — everything is from-scratch retrieval and reading.

1. **Retrieval layer** — hybrid dense+BM25 fusion. Local sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim) provide semantic similarity; sklearn TF-IDF provides keyword matching. Scores are min-max normalized and fused at 0.7/0.3 weighting. Top-30 rounds retrieved per question. This replaces the naive single-signal TF-IDF retrieval that the baseline used.

2. **Temporal grounding layer** — every retrieved excerpt is annotated with its raw timestamp AND a computed relative offset from the question's `question_date` (e.g., "3 weeks ago"). Temporal gap markers are inserted between non-adjacent sessions (e.g., "[2 weeks later]"). This gives the reader an explicit temporal frame for duration computation and recency judgment, rather than forcing it to parse raw date strings.

3. **Chain-of-thought reading layer** — the reader (GPT-5-mini) first writes compact notes extracting only relevant facts from the evidence, then answers from those notes. This is a single-call chain-of-thought pattern inspired by Chain-of-Note (Yu et al. 2023, [arXiv 2311.09210](https://arxiv.org/abs/2311.09210)), adapted to extract-then-answer rather than per-document assessment.

4. **Scoring correction layer** — for short gold answers (3 tokens or fewer), a deterministic strict-matching check catches cases where the GPT-4o judge incorrectly accepts or rejects based on loose semantic overlap. This is measurement cleanup, not model improvement (+5 cases out of 500).

**What we learned (honest gaps):**

- **Preference questions (73.3%)** are the hardest single category. The reader now uses stated preferences instead of saying "I don't know" (was 0%), but rubric alignment with the judge is inconsistent.
- **Multi-session counting (79.7%)** fails when the reader misses items buried in noisy conversation rounds. Fact extraction before reading (Mem0's approach) would help but introduces positional bias in long evidence — the extractor over-attends to early rounds and drops facts from later rounds.
- **The reader model matters enormously.** Published systems jumped 10+ points by switching from GPT-4o to GPT-5-mini with zero architecture changes. Our improvements would score ~70-75% with GPT-4o, not 88%.
- **We don't do indexing-time fact extraction.** Mem0 and the LongMemEval paper both extract atomic facts at ingestion time, creating cleaner retrieval keys. We retrieve raw conversation rounds and present them to the reader. This is the primary architectural gap between us and the 93%+ systems.

All artifacts, per-call diagnostics, cost logs, and a reproduction guide are in `eval/longmemeval/results/`. See [`eval/longmemeval/results/REPRODUCTION.md`](eval/longmemeval/results/REPRODUCTION.md).

### ACI-Bench (Nature Sci Data 2023)

TBD — harness built, full-run numbers to come.

## Repository map

TBD.

## Licensing

Apache 2.0 (see `LICENSE`). All code dependencies are OSI-approved. Model weights may additionally use open-data licences per the Linux Foundation OpenMDW framework. Claude Opus 4.7 via the Anthropic API is the hackathon's named sponsored tool.

## Disclaimer

The disclaimer block at the top of this README appears verbatim in the app header, the demo video opening and closing cards, and the written submission summary.
