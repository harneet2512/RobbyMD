# eval/medqa — stub

**Status**: stub. Scaffolding planned for next iteration.

Per `context.md §6` and `Eng_doc.md §10`, MedQA is one of the four benchmark picks. This directory will hold the adapter + baseline + full + run scaffolding parallel to `eval/{ddxplus, longmemeval, aci_bench}/` when the MedQA harness ships.

## Planned configuration

| Item | Value |
|---|---|
| **Dataset** | [Jin et al. 2021 MedQA (USMLE)](https://github.com/jind11/MedQA). Multi-choice licensing-exam questions. |
| **Split** | 1,273-question USMLE test split (canonical; per published benchmark convention). |
| **License** | CC-BY-4.0 (data; per `rules.md §1.2` data-license allowance). Redistribution permitted with attribution. |
| **Pack loaded** | `clinical_general` in MCQ mode (question stem → claims → answer selection). |
| **Reader** (baseline + substrate variant) | `gpt-4.1-mini` per `eval/README.md` model-usage policy. |
| **LLM judge** | None (multi-choice accuracy is deterministic). |
| **Metric** | Accuracy on the test split; comparator numbers include GPT-4 (86.7%), Med-PaLM-2 (86.5%), modern frontier models. |

## Why not seeded this build

Scope discipline. Three full harnesses (DDXPlus, LongMemEval-S, ACI-Bench) are enough to defend the substrate thesis and produce the demo-video eval slide. MedQA joins next iteration; this stub documents the intent so the architecture doesn't need to be retro-fitted when the harness ships.
