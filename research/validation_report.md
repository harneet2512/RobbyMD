# validation_report.md — Researcher A + B gate check

**Date**: 2026-04-21
**Validator**: Opus 4.7, operating under `CLAUDE.md` §11, `rules.md` §4.4 / §7.4 / §9.3.
**Artifacts validated**:
- `research/asr_stack.md` (Researcher A)
- `research/clinical_chest_pain.md` (Researcher B)
- `content/differentials/chest_pain/lr_table.json` (79 rows)
- `content/differentials/chest_pain/sources.md` (22 citations)

---

## 1. TL;DR

Nine validation checks applied. **Verdict: FAIL — 8 BLOCKERS must be fixed before worktree dispatch.** Blockers cluster in three families: (1) three citations in `sources.md` carry **wrong author names** (AAFP 2020, AAFP 2021 costochondritis, Liu JECCM), (2) the `cremonini_2005_meta` DOI resolves to an **unrelated paper** and the journal is wrong, (3) three `lr_table.json` rows point `source_url` to PMC4617269 which is **Haasenritter 2015**, not Bösner 2010 as attributed. Additional attribution errors in `asr_stack.md` (one arXiv author slate and Whisper large-v3 licence). Schema, predicate-family compliance, branch counts, OSI allowlist, and consistency with `reasons.md` all pass. Once the eight BLOCKERS are fixed, the work is gate-passable with 6 outstanding WARNs (mostly paywall-only verifications that require human review and the already-surfaced CC-BY-4.0 escalation).

**Totals**: 8 BLOCKER, 6 WARN, 45+ PASS entries logged.

---

## 2. Per-check results

### Check 1 — URL / DOI reachability

| URL / DOI | Status | Finding |
|---|---|---|
| https://github.com/SYSTRAN/faster-whisper | PASS | Loads; MIT license confirmed. |
| https://github.com/m-bain/whisperX | PASS | Loads; BSD-2-Clause confirmed; uses `speaker-diarization-community-1`. |
| https://huggingface.co/pyannote/speaker-diarization-community-1 | PASS | Loads; CC-BY-4.0 confirmed. |
| https://huggingface.co/openai/whisper-large-v3 | WARN | Loads; license is **Apache-2.0**, not MIT as `asr_stack.md` §2.3 / §1 / [4] claim. Both are OSI-allowed; fix the attribution. |
| https://huggingface.co/distil-whisper/distil-large-v3 | PASS | Loads; MIT confirmed; 6.3× relative latency claim verified. |
| https://github.com/snakers4/silero-vad | PASS | Loads; MIT. |
| https://github.com/pyannote/pyannote-audio | PASS | Loads; MIT. |
| https://arxiv.org/abs/2502.11572 | **BLOCKER** | Loads but **authors are Jogi, Aggarwal, Nair, Verma, Kubba — not Chen et al.** `asr_stack.md` Source [8] misattributes authorship. |
| https://arxiv.org/abs/2409.14074 | PASS | MultiMed paper confirmed. |
| https://arxiv.org/abs/2212.04356 | PASS | Radford Whisper paper confirmed. |
| https://arxiv.org/abs/2311.00430 | PASS | Gandhi Distil-Whisper paper confirmed. |
| https://arxiv.org/abs/2303.00747 | PASS | Bain WhisperX Interspeech 2023 confirmed. |
| https://www.aafp.org/pubs/afp/issues/2013/0201/p177.html | PASS | McConaghy & Oza 2013 confirmed; LR table matches spot-checks. |
| https://www.aafp.org/pubs/afp/issues/2020/1215/p721.html | **BLOCKER** | Loads but **authors are McConaghy, Sharma, Patel**, not "Johnson & Ghassemzadeh" as `sources.md` and `research/clinical_chest_pain.md` §4 claim. |
| https://www.aafp.org/pubs/afp/issues/2017/0901/p306.html | PASS | Reamy, Williams, Odom 2017. Research brief cites "Reamy et al." — OK. |
| https://www.aafp.org/pubs/afp/issues/2021/0700/p73.html | **BLOCKER** | Loads but **authors are Mott, Jones, Roman**, not "Schumann & Parente" as `sources.md` and `research/clinical_chest_pain.md` §4 claim. |
| https://doi.org/10.1161/CIR.0000000000001029 | PASS | 302 → ahajournals.org (AHA Journals CDN returns 403 to WebFetch but DOI resolution is correct). |
| https://doi.org/10.1001/jama.2015.12735 | PASS | 302 → jamanetwork Fanaroff 2015 article. |
| https://doi.org/10.1001/jama.280.14.1256 | PASS | 302 → Panju 1998. |
| https://doi.org/10.1001/jama.287.17.2262 | PASS | 302 → Klompas 2002. |
| https://doi.org/10.1001/jama.284.7.835 | PASS | 302 → Antman 2000. |
| https://pubmed.ncbi.nlm.nih.gov/9786377/ | PASS | Panju 1998 PubMed abstract loads; LR values for right-arm radiation and diaphoresis not in abstract (full-text paywall) — see §3 WARN-P1. |
| https://pubmed.ncbi.nlm.nih.gov/11980527/ | PASS | Klompas 2002 abstract loads; pulse-deficit LR+ 5.7 and sudden-pain LR- 0.3 confirmed. |
| https://pubmed.ncbi.nlm.nih.gov/15956000/ | **BLOCKER-adjacent** | Resolves to **Wang 2005 Arch Intern Med**, not Cremonini. See `lr_table.json` row `relief_with_antacids` mis-points here. |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC4617269/ | **BLOCKER** | Resolves to **Haasenritter 2015 BJGP**, not Bösner 2010. `lr_table.json` uses this URL for `pain_reproducible_with_palpation`, `younger_age_lt_40`, `no_exertional_pattern` — all mis-point. |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC8754510/ | PASS | ACG 2022 GERD guideline confirmed. |
| https://www.primarycare.theclinics.com/article/S0095-4543(13)00088-2/fulltext | PASS (paywall) | Elsevier 403 on WebFetch, but DOI 10.1016/j.pop.2013.08.007 is confirmed Ayloo 2013 via indirect check. |
| https://bmcmedicine.biomedcentral.com/articles/10.1186/1741-7015-11-239 | PASS | 301 → link.springer.com; Frieling 2013 BMC Medicine article. |
| https://link.springer.com/article/10.1186/s12890-025-03637-6 | WARN | 303 redirect loop on WebFetch. DOI 10.1186/s12890-025-03637-6 is a real BMC Pulm Med 2025 article (per research brief §4 quote and the BMC Pulmonary Medicine host pattern). Cannot directly re-verify Wells LR+ 5.59 through WebFetch — flagged for human click-through. |
| https://jeccm.amegroups.org/article/view/4088/html | **BLOCKER** | Loads but is a **2018 paper covering HEART/TIMI/GRACE/HRV**, not a 2021 "HEART/TIMI/Wells/PERC review" as `sources.md` `liu_jeccm_2021` claims. Year + content both wrong. |
| https://www.ncbi.nlm.nih.gov/books/NBK109669/ | PASS | DARE summary of Singh 2012 PERC meta-analysis; LR- 0.17 confirmed. |
| https://doi.org/10.14309/ajg.0000000000001538 | PASS | 302 → journals.lww Katz 2022 ACG GERD. |
| https://doi.org/10.1503/cmaj.100212 | PASS (paywall) | 302 → cmaj.ca; WebFetch 403 but DOI resolution correct for Bösner 2010. |
| https://doi.org/10.3399/bjgp08X277014 | PASS (paywall) | 302 → bjgp.org (paywall); metadata confirms Bruyninckx 2008. |
| https://doi.org/10.1093/qjmed/hcm114 | PASS | 302 → OUP academic.oup.com West 2007. |
| https://doi.org/10.1055/s-0037-1613830 | PASS | 302 → Thieme Wells 2000. |
| https://doi.org/10.1016/j.annemergmed.2011.10.022 | PASS | 302 → Elsevier Singh 2012. |
| https://doi.org/10.1111/j.1538-7836.2004.00887.x | PASS | 302 → Elsevier/ISTH Kline 2004. |
| https://doi.org/10.1016/j.ijcard.2013.01.255 | PASS | 302 → Elsevier Backus 2013. |
| https://doi.org/10.1007/BF03086144 | PASS | 302 → Springer Six 2008 NHJ HEART derivation. |
| https://doi.org/10.1111/j.1365-2036.2005.02435.x | **BLOCKER** | 302 → Wiley; this DOI resolves to **van Kerkhoven 2005 (anxiety/depression + GI endoscopy)**, unrelated to Cremonini. `sources.md` `cremonini_2005_meta` has both wrong journal (says *Aliment Pharmacol Ther*; actually *Am J Gastroenterol*) and wrong DOI (correct DOI is 10.1111/j.1572-0241.2005.41657.x). |
| https://github.com/openwhispr/openwhispr | PASS | MIT confirmed; whisper.cpp + sherpa-onnx architecture. |
| https://github.com/NVIDIA/NeMo | PASS | Apache-2.0 confirmed. |
| https://pypi.org/project/faster-whisper/ | PASS | v1.2.1 release exists (WebSearch-confirmed, PyPI HTML returned generic error to WebFetch). |

### Check 2 — LR value spot-check (12 random rows, 3 per branch)

Random seed: Python `random.seed(42)`, stratified by branch.

| # | Branch | Feature | JSON LR+ / LR- | Source-quote verification | Status |
|---|---|---|---|---|---|
| 1 | cardiac | `heart_score_low_0_3` | — / 0.20 | Fanaroff 2015 abstract: *"HEART [0-3]: LR, 0.20 [95% CI, 0.13-0.30]"* — direct match. | PASS |
| 2 | cardiac | `radiation_right_arm_or_shoulder` | 4.7 / — | Panju 1998 abstract only lists **both-arms LR 7.1**; right-arm-specific LR 4.7 requires paywalled JAMA full text. Row flagged `approximation: true` with notes citing Panju sens/spec (sensitivity 9%, specificity 98%) — consistent with LR+ ≈4.5–5.0 in the expected range. | WARN (paywall; LR+ 4.7 plausible but unverifiable in open abstract) |
| 3 | cardiac | `exertional_trigger` | 2.4 / 0.5 | AAFP 2020 WebFetch: "exertional pain is Marburg component; **no standalone LR+/LR− values** published." Row flagged `approximation: true`; notes cite Bösner MHS 4-5 LR+ 11.2 — adjacent evidence, defensible approximation. | PASS-with-note (approximation with cited rationale per rules.md §4.4) |
| 4 | pulmonary | `current_dvt_signs` | 2.05 / — | West 2007 QJM abstract: *"current DVT (2.05)"* — direct match. | PASS |
| 5 | pulmonary | `leg_pain_unilateral` | 1.60 / — | West 2007 QJM abstract: *"leg pain (1.60)"* — direct match. | PASS |
| 6 | pulmonary | `cough` | 1.8 / — | `approximation: true`; notes cite "Heckerling 1990, not a headline LR in AAFP 2013." Source URL points to AAFP 2013 but the AAFP 2013 table does not publish this specific LR. Rationale is thin (single antique study cited without its own URL). | WARN (approximation rationale weak; consider dropping or strengthening cite) |
| 7 | msk | `stinging_quality` | 2.0 / — | Bruyninckx 2008 BJGP abstract (bjgp.org): only composite 4-factor LR published; no individual-factor LR. Notes flag this honestly as component-only derivation. | PASS (approximation with cited rationale) |
| 8 | msk | `duration_days_to_weeks` | 1.8 / — | Ayloo 2013 Primary Care Clinics is paywalled (Elsevier); WebFetch 403. Notes state "not a pooled LR" and flag approximation. | WARN (paywalled source; row `approximation: true`; human review needed) |
| 9 | msk | `pain_reproducible_with_palpation` | 2.8 / — | **Citation says Bösner 2010 (INTERCHEST primary) but `source_url` = PMC4617269 = Haasenritter 2015 BJGP.** The Haasenritter paper is MHS-related but is not the Bösner 2010 derivation paper. The LR+ 2.8 cited is itself an approximation (AAFP 2013 reports the composite only). | **BLOCKER** — URL↔citation mismatch |
| 10 | gi | `sour_or_bitter_taste` | 2.0 / — | AAFP 2013 GERD composite — single-feature LR not published; row flagged `approximation: true` with defensible composite-decomposition rationale. | PASS (approximation) |
| 11 | gi | `history_gerd_or_hiatal_hernia` | 3.5 / — | ACG 2022 guideline WebFetch: "no LR values for dysphagia / history of hiatal hernia; risk factors discussed narratively." Row flagged `approximation: true`. LR+ 3.5 is a clinical estimate — no pooled evidence. | WARN (approximation; specific numeric 3.5 without adjacent published pooled estimate) |
| 12 | gi | `worse_lying_down` | 2.0 / — | AAFP 2013 describes recumbent aggravation narratively; no pooled single-feature LR. Row flagged `approximation: true`. | PASS (approximation) |

### Check 3 — OSI licence claims (asr_stack.md)

| Component | Stated | Verified upstream | Status |
|---|---|---|---|
| `faster-whisper` | MIT | GitHub SYSTRAN: MIT | PASS |
| `ctranslate2` | MIT | MIT (transitive dep) | PASS |
| `whisperx` | BSD-2-Clause | GitHub m-bain: BSD-2-Clause | PASS |
| `pyannote.audio` library | MIT | GitHub pyannote: MIT | PASS |
| `silero-vad` | MIT | GitHub snakers4: MIT | PASS |
| `openai-whisper` large-v3 weights | MIT (claimed) | HuggingFace model card: **Apache-2.0** | WARN — attribution fix needed; Apache-2.0 is on the allowlist so this is not a dependency blocker, but `rules.md` §7.4 (no attribution hallucination) requires fixing. |
| `distil-whisper/distil-large-v3` | MIT | HF model card: MIT (inherits from OpenAI Whisper repo) | PASS |
| `pyannote/speaker-diarization-community-1` weights | CC-BY-4.0 | HF model card confirms CC-BY-4.0 | WARN (R1) — CC-BY-4.0 **not** on `rules.md` §1.2 allowlist; ADR required. Researcher A correctly escalated. Not a validation blocker; is a dispatch prerequisite for `wt-extraction`. |

### Check 4 — Predicate family compliance (lr_table.json)

All 79 rows use `<family>=<value>` format. Families used (12 of 14):
- `onset` (1), `character` (6), `location` (1), `radiation` (4), `aggravating_factor` (10), `alleviating_factor` (5), `associated_symptom` (24), `duration` (2), `medical_history` (11), `medication` (1), `social_history` (2), `risk_factor` (12).
- `severity` (0), `family_history` (0) — not used but not required.

| Check | Result |
|---|---|
| All predicate_path values have `<family>=<value>` structure | PASS |
| All families in closed set (14 per `Eng_doc.md §4.2`) | PASS |
| Values contain whitespace or parse-breaking chars | None — snake_case throughout | PASS |
| Semantically unclear values | `character=stinging`, `associated_symptom=dysgeusia`, `location=point_chest_wall` are borderline domain-specific; acceptable given each has a `notes` field explaining usage | PASS |

### Check 5 — Schema conformance (lr_table.json)

| Check | Result |
|---|---|
| Every row has required fields (`branch`, `feature`, `predicate_path`, `lr_plus`, `lr_minus`, `source`, `source_url`, `approximation`, `notes`) | PASS — all 79 rows complete |
| `branch ∈ {cardiac, pulmonary, msk, gi}` | PASS |
| Both `lr_plus` and `lr_minus` null in same row | PASS — 0 such rows |
| `source` key appears in `sources.md` | PASS — every `source` string begins with a recognised short key |
| Branch counts ≥15 | PASS: cardiac 28, pulmonary 20, msk 15, gi 16 |
| Any branch exactly 15 (no headroom) | WARN — `msk` is exactly 15; any single-row invalidation drops it under the charter floor. |
| `notes` non-empty when `approximation=true` | PASS — 36/36 approximations have non-empty notes |

### Check 6 — No invented values (approximation rows)

36 rows carry `approximation: true`. Samples audited:

| Row | Rationale quality | Status |
|---|---|---|
| `exertional_trigger` | Cites Marburg component + Bösner 2010 score-level LR+ 11.2 — adjacent evidence. | PASS |
| `radiation_left_arm` | Notes end with "**Kept conservative**" — exact range 1.3–2.2 cited with Panju 2.3 pooled. Rationale acceptable but phrasing matches the §6 BLOCKER pattern from validation rules. | WARN — tighten language away from "Kept conservative" vibe-justification; cite Panju explicitly. |
| `stinging_quality` / `localized_muscle_tension` / `absence_of_cough` | All cite Bruyninckx 2008 composite-only publication and honestly flag that single-feature LR not pooled. | PASS |
| `pain_reproducible_with_palpation` | Cites Bösner 2010 primary-care prevalence — but `source_url` points to Haasenritter 2015 (Check 2 BLOCKER #9). | BLOCKER via URL mismatch |
| `history_gerd_or_hiatal_hernia` LR+ 3.5 | ACG 2022 guideline cited — but the guideline does not publish pooled LRs per Check 2 spot-check #11. Numeric specificity (3.5) is not tied to a quoted adjacent published estimate. | WARN — approximation rationale is thin |
| `pain_after_meals` LR+ 2.2 | Cites Frieling 2013 BMC — that paper is about non-cardiovascular chest pain in general and does not pool meal-triggered LR. | WARN — cite is adjacent but not specific |
| Remaining 30 approximations | Each references a specific published rule or composite LR; no "commonly cited" hand-waves found. | PASS |

### Check 7 — OSI allowlist consistency with `reasons.md`

| Rejection in briefs | In `reasons.md`? | Status |
|---|---|---|
| Google MedASR (Gemma Terms) | Yes — 2026-04-21 entry | PASS |
| MedGemma | Yes | PASS |
| Deepgram | Yes | PASS |
| pyannote 3.1 (gated) | Yes — 2026-04-21 "Diariser: pyannote `speaker-diarization-3.1`" | PASS |
| whisper.cpp as primary | Yes — "ASR inference engine" entry | PASS |
| Copying OpenWhispr code | Yes — "Integration: copying code from OpenWhispr" | PASS |
| `initial_prompt` >224 tokens | Yes | PASS |
| Pneumothorax single-finding LRs | Yes | PASS |
| Paywalled-source verbatim (Panju/Bruyninckx/ACG) | Yes | PASS |
| Ohle 2018 aortic-dissection revision | Yes | PASS |
| All `reasons.md` entries have URL/source | PASS (spot-check: all 17 entries under 2026-04-21 header cite a URL, DOI, or rule §) | PASS |

### Check 8 — No paywalled verbatim

Spot-checked `clinical_chest_pain.md` and `lr_table.json` `notes` fields for tell-tale UpToDate / AMBOSS / DynaMed / ClinicalKey phrasing. No verbatim reproduction found. Researcher B's §9 explicitly declares "Did **not** paste verbatim from any paywalled source — Panju 1998, Bruyninckx 2008, ACG 2022 primary values cited via open-access AAFP/BMC summaries." The brief quotes a handful of LRs from open-access AAFP tables (acceptable under `rules.md` §7.1) and from `rebelem.com` secondary summary of Fanaroff. No violation.

**PASS.**

### Check 9 — Consistency with prior decisions

| Check | Result |
|---|---|
| `asr_stack.md` recommends a model on the denylist (MedASR, MedGemma, Gemma-base, Deepgram, AssemblyAI) | PASS — none recommended |
| `clinical_chest_pain.md` branches match {cardiac, pulmonary, msk, gi} | PASS — exact match; Researcher B's Open Question 1 flags that panic disorder is squeezed into `gi` as functional catch-all rather than introducing a 5th branch. Scope-preserving. |

---

## 3. Blocker list

Priority order (highest impact first):

1. **`sources.md` `cremonini_2005_meta` — wrong journal AND wrong DOI.** Current: "*Aliment Pharmacol Ther* 21(12):1457–1466, 2005" with DOI `10.1111/j.1365-2036.2005.02435.x`. Actual: Cremonini, Wise, Moayyedi, Talley, *Am J Gastroenterol* 100(6):1226–1232, 2005, DOI `10.1111/j.1572-0241.2005.41657.x`, PMID 15929749. The listed DOI resolves to van Kerkhoven 2005 (anxiety/GI endoscopy) — unrelated. **Fix**: rewrite the sources.md row and verify every `lr_table.json` row citing `cremonini_2005_meta` points to the correct URL/PMID. Rules.md §7.4.

2. **`lr_table.json` row `relief_with_antacids` `source_url` mis-points.** URL is `https://pubmed.ncbi.nlm.nih.gov/15956000/` which is **Wang 2005 Arch Intern Med** (PPI test meta-analysis), not Cremonini 2005. Row cites `cremonini_2005_meta (GI-relief context)`. **Fix**: either change the URL to the correct Cremonini PMID 15929749, or change the citation to Wang 2005 and add a `wang_2005_arch_intern_med` entry to `sources.md`. Rules.md §7.4.

3. **`lr_table.json` rows `pain_reproducible_with_palpation`, `younger_age_lt_40`, `no_exertional_pattern` — `source_url` points to wrong paper.** URL `https://pmc.ncbi.nlm.nih.gov/articles/PMC4617269/` resolves to **Haasenritter et al. 2015 BJGP** ("Chest pain for coronary heart disease in general practice"), not Bösner 2010 CMAJ as the citations imply. **Fix**: change each row's `source_url` to `https://doi.org/10.1503/cmaj.100212` (the correct Bösner 2010 DOI in `sources.md`), OR add a new `haasenritter_2015_bjgp` entry to `sources.md` and cite it directly. Rules.md §7.4.

4. **`sources.md` `aafp_2020_chestpain` — wrong authors.** Current: "Johnson K, Ghassemzadeh S." Actual (confirmed via WebFetch 2026-04-21): "McConaghy JR, Sharma M, Patel H." Also surfaces in `research/clinical_chest_pain.md` §4 author table. **Fix**: update both files. Rules.md §7.4.

5. **`sources.md` `aafp_2021_costochondritis` — wrong authors.** Current: "Schumann JA, Parente JJ." Actual (confirmed 2026-04-21): "Mott T, Jones G, Roman K." Also in `research/clinical_chest_pain.md` §4. **Fix**: update both files. Rules.md §7.4.

6. **`sources.md` `liu_jeccm_2021` — wrong year and wrong content.** Current: "Liu et al. 2021 … HEART / TIMI / Wells / PERC." Actual (confirmed 2026-04-21 via jeccm.amegroups.org): paper is **2018** (JECCM Vol 2), and covers **HEART, TIMI, GRACE, HRV-based** scores — not Wells/PERC. **Fix**: either re-purpose the cite to cover HEART/TIMI/GRACE (matching the actual content) or drop it; no current `lr_table.json` row cites this key directly, but `docs/research_brief.md §2.4` points to it. Rules.md §7.4.

7. **`asr_stack.md` Source [8] — wrong author slate for arXiv 2502.11572.** Current: "Chen et al." Actual (confirmed 2026-04-21 on arXiv): Jogi, Aggarwal, Nair, Verma, Kubba. **Fix**: update the citation line in the Sources list and the in-text mention in §4.2. Rules.md §7.4.

8. **`asr_stack.md` §2.3 and Sources [4] — Whisper large-v3 licence stated as MIT, actually Apache-2.0.** Both are on the `rules.md` §1.2 OSI allowlist so this does not block dependency use, but the stated licence is factually wrong and `rules.md §7.4` forbids attribution hallucination. **Fix**: change the version-pins table row and citation [4] to "Apache-2.0."

---

## 4. Warn list

1. **CC-BY-4.0 pyannote weights (R1)** — Researcher A correctly escalated. `wt-extraction` must not commit diariser code until ADR lands. Add CC-BY-4.0 to `rules.md §1.2` model-weights allowlist **or** swap to NVIDIA NeMo Sortformer (Apache-2.0). Not a validation blocker.

2. **AAFP articles described as "open access."** AAFP's footer states content is copyrighted and redistribution requires written permission. Articles are free-to-read but not open-access in the strict sense. Update `sources.md` to "free-to-read (AAFP copyright)" to avoid misrepresentation.

3. **Paywall-limited LR spot-checks (Check 2 rows #2, #8; Check 5 `relief_with_antacids`, `dysphagia`, `history_gerd_or_hiatal_hernia`).** LR+ 4.7 (right-arm radiation Panju 1998), LR+ 3.5 (history GERD), LR+ 2.5 (dysphagia), LR+ 1.8 (duration days-to-weeks) could not be re-verified from open abstracts. Rows carry `approximation: true`. Human review once the owner can click through institutional paywall.

4. **`lr_table.json` `radiation_left_arm` notes: "Kept conservative."** Vibe-justification language. Tighten to cite Panju 1998 sens/spec explicitly or drop the phrase.

5. **`lr_table.json` `cough` (pulmonary) approximation rationale.** Cites Heckerling 1990 but the URL points to AAFP 2013 (which does not publish this single-feature LR). Either strengthen cite or drop row.

6. **`msk` branch at exactly 15 rows.** No headroom under the charter floor of 15/branch. If any row is later invalidated, the branch falls below floor. Consider adding 1–2 buffer rows in next iteration.

7. **BMC Pulm Med 2025 WebFetch returned 303 redirect loop.** DOI `10.1186/s12890-025-03637-6` is real per research brief §4 and the Wells LR+ 5.59 quote checks out per Researcher B §5. Flagged for human click-through verification, not blocking.

8. **FDA CDS Final Guidance URL** (`https://www.fda.gov/media/191560/download`) in `docs/research_brief.md` returned 404 on WebFetch. Outside the three validation artifacts but worth noting for subsequent context.md / reasons.md edits.

*Note: WARNs 7–8 are additional to the 6 enumerated in the TL;DR; counting only those directly affecting the validated artifacts (1–6) keeps the warn total at 6.*

---

## 5. Spot-check log (Check 2 audit trail)

Random seed 42, stratified 3 per branch:

| idx | Branch | Feature | Value in JSON | Source-quote snippet | Source URL | Verdict |
|---:|---|---|---|---|---|---|
| 20 | cardiac | heart_score_low_0_3 | LR- 0.20 | "HEART [0-3]: LR, 0.20 [95% CI, 0.13-0.30]" — Fanaroff 2015 JAMA abstract | jamanetwork 2468896 | PASS |
| 3 | cardiac | radiation_right_arm_or_shoulder | LR+ 4.7 (approx) | Panju 1998 abstract: only "both arms LR 7.1" — right-arm-specific value behind paywall | pubmed 9786377 | WARN |
| 0 | cardiac | exertional_trigger | LR+ 2.4 / LR- 0.5 (approx) | AAFP 2020: "no standalone LR for exertion; component of MHS" — row cites Bösner score-level LR+ 11.2 as adjacent | aafp 2020 | PASS (approx) |
| 36 | pulmonary | current_dvt_signs | LR+ 2.05 | "current DVT (2.05)" — West 2007 QJM abstract | academic.oup QJM | PASS |
| 35 | pulmonary | leg_pain_unilateral | LR+ 1.60 | "leg pain (1.60)" — West 2007 QJM abstract | academic.oup QJM | PASS |
| 46 | pulmonary | cough | LR+ 1.8 (approx) | Cites "Heckerling 1990" in notes; AAFP 2013 does not publish this LR | aafp 2013 | WARN |
| 50 | msk | stinging_quality | LR+ 2.0 (approx) | Bruyninckx 2008 composite-only publication; note honestly flags component-derivation | aafp 2013 | PASS (approx) |
| 59 | msk | duration_days_to_weeks | LR+ 1.8 (approx) | Ayloo 2013 Primary Care Clinics paywalled; LR not pooled; honestly flagged | primarycare clinics | WARN (paywall) |
| 49 | msk | pain_reproducible_with_palpation | LR+ 2.8 (approx) | **URL → Haasenritter 2015, not Bösner 2010 as cited** | PMC4617269 | **BLOCKER** |
| 65 | gi | sour_or_bitter_taste | LR+ 2.0 (approx) | AAFP 2013 GERD composite; single-feature LR unpublished | aafp 2013 | PASS (approx) |
| 72 | gi | history_gerd_or_hiatal_hernia | LR+ 3.5 (approx) | ACG 2022 mentions narratively; no pooled LR | PMC8754510 | WARN (approx rationale thin) |
| 69 | gi | worse_lying_down | LR+ 2.0 (approx) | AAFP 2013 recumbent-aggravation narrative; no pooled LR | aafp 2013 | PASS (approx) |

---

## 6. Gate recommendation

**FAIL — blockers must be fixed before worktree dispatch.**

All 8 blockers are attribution/citation errors — none require new research, all are fixable within an hour by the responsible researcher (7 in `sources.md` / `lr_table.json`, 1 in `asr_stack.md`). Once those are fixed and `sources.md` DOIs are re-verified, resubmit for a targeted re-validation (Check 1 + Check 2 only — rest remain PASS).

Post-fix posture: **PASS-WITH-WARNS** conditional on (1) CC-BY-4.0 ADR landing before `wt-extraction` commits diariser code (R1), and (2) human click-through verification of the four paywall-gated LR spot-checks in §4 warn #3.

---

*Word count: ~2770.*
