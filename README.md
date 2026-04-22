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

## Quickstart

TBD.

## Evaluation

TBD — LongMemEval-S (ICLR 2025) and ACI-Bench (Nature Sci Data 2023) harnesses are built; full-run numbers to come.

## Repository map

TBD.

## Licensing

Apache 2.0 (see `LICENSE`). All code dependencies are OSI-approved. Model weights may additionally use open-data licences per the Linux Foundation OpenMDW framework. Claude Opus 4.7 via the Anthropic API is the hackathon's named sponsored tool.

## Disclaimer

The disclaimer block at the top of this README appears verbatim in the app header, the demo video opening and closing cards, and the written submission summary.
