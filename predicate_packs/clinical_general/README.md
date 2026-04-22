# predicate_packs/clinical_general

The first (and, this build, only) shipped `PredicatePack`. Covers **general clinical vocabulary** — any chief complaint, not just chest pain.

Per `Eng_doc.md §4.2`, predicate families are declared in pluggable domain packs. The engine has no clinical medicine hardcoded — predicate families are data, not code. This pack's closed vocabulary is shared across every chief complaint the substrate runs on.

## Predicate families (closed vocabulary)

```
onset                  character                severity
location               radiation                aggravating_factor
alleviating_factor     associated_symptom       duration
medical_history        medication               allergy
family_history         social_history           risk_factor
vital_sign             lab_value                imaging_finding
physical_exam_finding  review_of_systems
```

## Structured sub-slots

| Predicate | Sub-slots |
|---|---|
| `medication` | `name, dose, route, frequency, indication, start_date` |
| `vital_sign` | `kind ∈ {BP, HR, RR, SpO2, Temp}, value, unit, measured_at` |
| `lab_value` | `name, value, unit, reference_range, specimen_type, collected_at` |
| `imaging_finding` | `modality ∈ {X-ray, CT, MRI, US}, body_part, finding_text, reported_at` |
| `physical_exam_finding` | `body_part, finding, elicitation_method` |
| `allergy` | `agent, reaction, severity, verified_by` |

## Differential branch groups

Under `differentials/`:

| Directory | Status | Notes |
|---|---|---|
| `chest_pain/` | **Seeded + rehearsed**. | 79-row LR table (`lr_table.json`), 4 branches (Cardiac/Pulmonary/MSK/GI), 22 citations. This is the demo case. |
| `abdominal_pain/` | Stub | Schema-ready; seed next iteration. |
| `dyspnoea/` | Stub | Schema-ready; seed next iteration. |
| `headache/` | Stub | Schema-ready; seed next iteration. |

## Adding a new branch group (within clinical_general)

1. Create `predicate_packs/clinical_general/differentials/<complaint>/`.
2. Populate `branches.json` (tree structure), `lr_table.json` (LR schema per `Eng_doc.md §4.4`), `sources.md` (citation list).
3. Every LR row cites a peer-reviewed or guideline source (`rules.md §4.4`).
4. No engine code changes required. The loader picks up the new complaint as soon as the LR table's `chief_complaint` field is set and the directory exists.

## Adding a non-clinical pack (e.g., personal_assistant)

See `Eng_doc.md §4.2`. Register via `src/substrate/predicate_packs.py::register_pack(pack)`. Each pack declares its own `predicate_families`, `sub_slots`, and optional `lr_table_path`. The engine treats all registered packs identically.
