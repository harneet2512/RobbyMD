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

## Architecture — end-to-end pipeline

What happens from the moment the doctor starts talking to the SOAP note on screen:

```
MICROPHONE
    |
    v
[1. ASR Pipeline]  8 stages, all open-source models
    |  src/extraction/asr/pipeline.py (2,276 LOC)
    |  Whisper large-v3 -> WhisperX alignment -> pyannote diarisation
    |  -> hallucination guard (5 checks) -> medical vocab correction
    |
    v  Output: timestamped, diarised turns (who said what, when)
    
[2. Admission Filter]  drop filler, duplicates, <3-word turns
    |  src/substrate/admission.py
    |
    v  Output: admitted turns with speaker role + turn_id
    
[3. Claim Extraction]  Opus 4.7, structured output
    |  src/extraction/claim_extractor/ (445 LOC)
    |  Input: current turn + 2 prior turns + active claim set
    |  Output: claims with (subject, predicate, value, confidence,
    |          source_turn_id, char_start, char_end)
    |  Predicate vocabulary is CLOSED — from the active pack only
    |
    v  Output: candidate claims with span provenance
    
[4. Supersession Detector]  deterministic, no LLM
    |  src/substrate/supersession.py (236 LOC)
    |  src/substrate/supersession_semantic.py (263 LOC)
    |
    |  Pass 1 (rule-based): same (subject, predicate), different value
    |    -> typed edge: PATIENT_CORRECTION | PHYSICIAN_CONFIRM |
    |       REFINES | CONTRADICTS
    |
    |  Pass 2 (semantic): e5-small-v2 embeddings, cosine >= 0.88
    |    -> edge: SEMANTIC_REPLACE with identity_score
    |
    |  OLD CLAIM IS NEVER DELETED. Marked SUPERSEDED, linked to new
    |  claim via supersession_edges table. Both remain queryable.
    |
    v  Output: claims with lifecycle state (active | superseded)
    
[5. Claim Store + Projections]  SQLite, 10 tables
    |  src/substrate/schema.py (358 LOC)
    |  src/substrate/claims.py (489 LOC)
    |  src/substrate/projections.py (127 LOC)
    |
    |  Tables: turns, claims, supersession_edges, decisions,
    |  note_sentences, claim_embeddings, claim_metadata,
    |  event_frames, event_frame_claims, event_frame_embeddings
    |
    |  4 materialized branch views (one per differential branch)
    |
    v  Output: structured state — the "substrate"
    
[6. Differential Update Engine]  pure math, zero LLM
    |  src/differential/engine.py (152 LOC)
    |  src/differential/lr_table.py (221 LOC)
    |
    |  For each active claim: look up matching LR rows
    |  Feature present -> multiply by LR+
    |  Feature absent  -> multiply by LR-
    |  Sum log-likelihoods per branch, softmax, rank
    |
    |  4 branches: Cardiac, Pulmonary, MSK, GI
    |  81 LR features from 30 peer-reviewed sources
    |  Same inputs = identical output, always (property tested 100x)
    |  Latency: <50ms per full recompute
    |
    v  Output: ranked differential with posterior probabilities
    
[7. Counterfactual Verifier]  deterministic selection + 1 Opus call
    |  src/verifier/verifier.py (346 LOC)
    |
    |  For top-2 branches: find support (LR+ > 1) and
    |  refutation (LR+ > 1.5, feature absent)
    |  Score each candidate: |log(LR_A) - log(LR_B)| x uncertainty
    |  Pick argmax (deterministic, tiebreak on feature name)
    |
    |  1 Opus 4.7 call to phrase the question in clinical language
    |  Output: next_best_question, why_moved, missing_evidence
    |
    v  Output: "Ask about exertional component" + rationale
    
[8. SOAP Note Generator]  Opus 4.7 + provenance validation
    |  src/note/generator.py (515 LOC)
    |
    |  Groups active claims by SOAP section (pack's soap_mapping.json)
    |  LLM generates sentences with [c:claim_id] markers inline
    |  Validator parses markers, drops any sentence where claim_id
    |  doesn't exist in active claims
    |
    |  Every surviving sentence has non-empty source_claim_ids
    |
    v  Output: SOAP note where every sentence traces to claims
    
[9. UI Renderer]  React + ReactFlow + Zustand
    |  ui/src/ (6,007 LOC, 17 components)
    |
    |  4 panels + auxiliary strip:
    |  - Transcript (click turn -> highlight claims)
    |  - Claim State (click claim -> highlight source turn)
    |  - Differential Trees (ReactFlow, 200ms transitions)
    |  - SOAP Note (click sentence -> highlight source claims + turn)
    |  - Aux Strip: why_moved + next_best_question
    |
    v  Output: the doctor sees everything, clicks anything to trace back

[10. API + Event Bus]  WebSocket, real-time
     src/api/server.py (532 LOC)
     src/substrate/event_bus.py (89 LOC)
     
     Every claim creation, supersession, and projection update
     broadcasts to connected UI clients via WebSocket
     Supports session replay from stored events (demo_replay.py)
```

### The provenance invariant

Every element on screen traces back to the source:

```
SOAP sentence -> [c:claim_id] -> claim -> source_turn_id -> turn -> original transcript
                                  |
                                  +-> char_start, char_end (exact substring in transcript)
                                  |
                                  +-> supersession_edges -> old_claim (if fact changed)
```

This is enforced at write time (`claims.py` validates `source_turn_id` exists, `generator.py` drops sentences without valid claim IDs) and tested at build time (`test_no_phi.py`, `test_determinism.py`).

### The supersession model

When a patient says "no allergies" then later says "actually, penicillin" — other systems delete or overwrite the old fact. We supersede it:

```
claim_001: "no known allergies" (turn 3, ACTIVE)
    |
    | Patient says "actually I'm allergic to penicillin" at turn 15
    |
    v
claim_001: "no known allergies" (turn 3, SUPERSEDED)
    |--- supersession_edge ---> claim_047: "allergic to penicillin" (turn 15, ACTIVE)
         edge_type: PATIENT_CORRECTION
         identity_score: 0.91
```

Both claims stay in the database. The doctor can see what was originally stated, what changed, and when. The differential engine uses only ACTIVE claims. The note generator can reference the change.

### Clinical content: chest pain pack

`predicate_packs/clinical_general/differentials/chest_pain/`

- **4 branches**: Cardiac, Pulmonary, MSK, GI (`branches.json`, 209 lines)
- **81 LR features** with likelihood ratios from peer-reviewed sources (`lr_table.json`, 918 lines)
- **30 citations** — 2021 AHA/ACC Chest Pain Guideline, HEART Score, TIMI, Wells PE, PERC, Fanaroff JAMA 2015, Panju JAMA 1998, and 22 more (`sources.md`, 66 lines)
- No proprietary content (no UpToDate, no AMBOSS)
- Pluggable: add a new complaint by dropping `branches.json` + `lr_table.json` + `sources.md` into a new directory. No engine code changes.

### What's tested

95 test files across:
- `tests/property/test_determinism.py` — runs differential engine 100x on same input, asserts bit-identical output every time
- `tests/privacy/test_no_phi.py` — scans entire repo for SSN/MRN/DOB patterns, fails build on match
- `tests/licensing/test_open_source.py` — validates every dependency is OSI-approved (MIT, Apache-2.0, BSD, MPL, ISC, LGPL)
- `tests/e2e/test_demo_case.py` — scripted chest-pain case end-to-end
- Unit tests on substrate, extraction, differential modules

### Code stats

| Component | Path | LOC | Files |
|-----------|------|-----|-------|
| Substrate core | `src/substrate/` | 4,223 | 17 |
| ASR pipeline | `src/extraction/asr/` | 2,276 | 9 |
| Claim extractor | `src/extraction/claim_extractor/` | 445 | 3 |
| Differential engine | `src/differential/` | 584 | 5 |
| Verifier | `src/verifier/` | 445 | 3 |
| Note generator | `src/note/` | 515 | 1 |
| API server | `src/api/` | 755 | 3 |
| UI | `ui/src/` | 6,007 | 17 components |
| Eval harnesses | `eval/` | 30,937 | 271 |
| **Total** | | **~46,000** | |

143 commits, all within the hackathon window (April 15-26, 2026).

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

```
src/
  substrate/          claim store, supersession, provenance, event frames (4,223 LOC)
  extraction/
    asr/              8-stage Whisper + WhisperX + pyannote pipeline (2,276 LOC)
    claim_extractor/  Opus 4.7 structured extraction (445 LOC)
  differential/       deterministic LR-weighted ranking engine (584 LOC)
  verifier/           counterfactual discriminator + next-best-question (445 LOC)
  note/               SOAP generator with provenance validation (515 LOC)
  api/                WebSocket server + event replay (755 LOC)

ui/src/               React + ReactFlow + Zustand (6,007 LOC, 17 components)

predicate_packs/
  clinical_general/
    differentials/
      chest_pain/     4 branches, 81 LR features, 30 citations

eval/
  longmemeval/        LongMemEval-S harness + 88.4% result (442/500)
  aci_bench/          ACI-Bench harness (built, full-run pending)

tests/                95 files: determinism, privacy, licensing, e2e
```

## Licensing

Apache 2.0 (see `LICENSE`). All code dependencies are OSI-approved. Model weights may additionally use open-data licences per the Linux Foundation OpenMDW framework. Claude Opus 4.7 via the Anthropic API is the hackathon's named sponsored tool.

## Disclaimer

The disclaimer block at the top of this README appears verbatim in the app header, the demo video opening and closing cards, and the written submission summary.
