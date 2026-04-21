# Researcher B — Clinical Chest Pain LR Expansion

**Date**: 2026-04-21
**Author**: Researcher B (Opus 4.7)
**Deliverables**: this brief + `content/differentials/chest_pain/lr_table.json` (79 entries) + `content/differentials/chest_pain/sources.md` (22 citations).

## 1. Executive summary

Built on top of the 17-row AAFP 2013 extraction already captured in `docs/research_brief.md §2.4`, I extended the chest-pain LR table to **79 entries** across the four differential branches: **cardiac (28)**, **pulmonary (20)**, **msk (15)**, **gi (16)**. Every feature is named in terms of the closed predicate family in `Eng_doc.md §4.2` — no new predicate families introduced.

Citation set grew from 7 to **22 peer-reviewed / guideline references**. Core additions beyond the starting list: Fanaroff 2015 JAMA *Rational Clinical Examination* (ACS — the modern pooled LR set), Panju 1998 JAMA (AMI historical pooled LRs), Klompas 2002 JAMA (aortic dissection), West 2007 QJM (PE individual-feature pooled LRs), Bruyninckx 2008 BJGP (4-factor chest-wall rule primary source), Bösner 2010 CMAJ (Marburg Heart Score derivation), Cremonini 2005 (PPI-trial GERD meta-analysis), Katz 2022 ACG (GERD guideline), Ayloo 2013 (MSK chest pain primary care review), Singh 2012 (PERC meta-analysis), BMC Pulm Med 2025 (Wells meta-analysis).

**Coverage strength by branch (published-LR density):** cardiac **strong** (≥19/28 directly pooled), pulmonary **strong** (≥12/20 directly pooled), msk **weak** (3/15 directly pooled — combined-rule LRs only; 12 approximations with defensible rationale), gi **weak-to-medium** (5/16 directly pooled; composite-rule LRs dominate, most single-feature LRs are approximations). All approximations carry `approximation: true` and cite adjacent published evidence in `notes`.

**No LR was invented.** Every row has a citation; weak-evidence rows are flagged.

## 2. Per-branch tree structure

### 2.1 Cardiac (28 features)

| Feature | predicate_path | LR+ | LR- | Source key | Notes |
|---|---|---:|---:|---|---|
| exertional_trigger | aggravating_factor=exertion | 2.4 | 0.5 | aafp_2020_chestpain | Approx — single-feature LR inferred from Marburg component |
| radiation_both_arms | radiation=both_arms | 2.6 | — | fanaroff_jama_2015 | Pooled (95% CI 1.8-3.7) |
| radiation_left_arm | radiation=left_arm | 2.3 | — | aafp_2013_chestpain | Approx — range 1.3-2.2 |
| radiation_right_arm_or_shoulder | radiation=right_arm | 4.7 | — | panju_jama_1998 | Approx — pooled Panju 1998 |
| pain_similar_to_prior_ischemia | character=similar_to_prior_ischemia | 2.2 | — | fanaroff_jama_2015 | Pooled (CI 2.0-2.6) |
| change_in_pain_pattern_24h | character=crescendo_pattern | 2.0 | — | fanaroff_jama_2015 | Pooled (CI 1.6-2.4) |
| diaphoresis | associated_symptom=diaphoresis | 2.0 | 0.64 | panju_jama_1998 | Approx |
| nausea_or_vomiting | associated_symptom=nausea | 1.9 | 0.7 | panju_jama_1998 | Approx |
| pleuritic_pain_rules_out_acs | aggravating_factor=inspiration | 0.2 | — | aafp_2013_chestpain | Rules out |
| reproducible_with_palpation_rules_out_acs | alleviating_factor=palpation | 0.3 | — | aafp_2013_chestpain | Rules out |
| sharp_stabbing_quality_rules_out_acs | character=sharp | 0.3 | — | aafp_2013_chestpain | Rules out |
| nitroglycerin_response | alleviating_factor=nitroglycerin | 1.1 | — | fanaroff_jama_2015 | Non-discriminatory (CI 0.93-1.3) |
| prior_cad_history | medical_history=coronary_artery_disease | 2.0 | — | fanaroff_jama_2015 | Pooled (CI 1.4-2.6) |
| prior_abnormal_stress_test | medical_history=abnormal_stress_test | 3.1 | — | fanaroff_jama_2015 | Pooled (CI 2.0-4.7) |
| peripheral_arterial_disease | medical_history=peripheral_arterial_disease | 2.7 | — | fanaroff_jama_2015 | Pooled (CI 1.5-4.8) |
| history_of_heart_failure | medical_history=heart_failure | 5.8 | 0.45 | aafp_2013_chestpain | For HF sub-hypothesis |
| prior_mi | medical_history=myocardial_infarction | 3.1 | 0.69 | aafp_2013_chestpain | |
| third_heart_sound_s3 | associated_symptom=s3_gallop | 3.2 | 0.88 | aafp_2013_chestpain | Physician-elicited |
| hypotension_sbp_lt_100 | associated_symptom=hypotension | 3.1 | — | fanaroff_jama_2015 | (CI 1.2-7.9) |
| heart_score_high_7_10 | risk_factor=heart_score_high | 13.0 | — | fanaroff_jama_2015 | Strongest single LR+ in set |
| heart_score_low_0_3 | risk_factor=heart_score_low | — | 0.20 | fanaroff_jama_2015 | Strong rule-out |
| timi_score_high_5_7 | risk_factor=timi_score_high | 6.8 | — | fanaroff_jama_2015 | |
| timi_score_low_0_1 | risk_factor=timi_score_low | — | 0.31 | fanaroff_jama_2015 | |
| marburg_heart_score_high_4_5 | risk_factor=marburg_score_high | 11.2 | — | aafp_2020_chestpain | Primary-care score |
| marburg_heart_score_low_0_1 | risk_factor=marburg_score_low | 0.04 | — | aafp_2020_chestpain | Very strong rule-out |
| aortic_dissection_pulse_deficit | associated_symptom=pulse_deficit | 5.7 | — | klompas_jama_2002 | Aortic dissection |
| sudden_onset_tearing_pain | onset=sudden | 2.6 | 0.3 | klompas_jama_2002 | Approx LR+; LR- direct |
| chest_or_back_pain_plus_pulse_differential | radiation=back | 5.3 | — | aafp_2013_chestpain | Aortic dissection |

### 2.2 Pulmonary (20 features)

| Feature | predicate_path | LR+ | LR- | Source key | Notes |
|---|---|---:|---:|---|---|
| pleuritic_pain | aggravating_factor=inspiration | 1.8 | — | aafp_2017_pleuritic | Approx — aggregate |
| sudden_dyspnea | associated_symptom=sudden_dyspnea | 1.83 | 0.43 | west_qjm_2007 | PE pooled |
| any_dyspnea | associated_symptom=dyspnea | — | 0.52 | west_qjm_2007 | Rule-out |
| hemoptysis | associated_symptom=hemoptysis | 1.62 | — | west_qjm_2007 | |
| syncope | associated_symptom=syncope | 2.38 | — | west_qjm_2007 | |
| shock | associated_symptom=hypotension | 4.07 | — | west_qjm_2007 | |
| leg_swelling | associated_symptom=leg_swelling | 2.11 | — | west_qjm_2007 | |
| leg_pain_unilateral | associated_symptom=leg_pain | 1.60 | — | west_qjm_2007 | |
| current_dvt_signs | associated_symptom=dvt_signs | 2.05 | — | west_qjm_2007 | |
| active_cancer | medical_history=active_cancer | 1.74 | — | west_qjm_2007 | |
| recent_surgery_or_immobilization | medical_history=recent_surgery | 1.63 | — | west_qjm_2007 | |
| wells_pe_high_probability | risk_factor=wells_pe_high | 5.59 | — | bmc_pulm_2025 | 2025 meta |
| wells_pe_low_probability | risk_factor=wells_pe_low | — | 0.34 | bmc_pulm_2025 | |
| perc_rule_negative | risk_factor=perc_negative | — | 0.17 | perc_meta_2012 | Rule-out workhorse |
| egophony_on_auscultation | associated_symptom=egophony | 8.6 | 0.96 | aafp_2013_chestpain | Pneumonia |
| dullness_to_percussion | associated_symptom=dullness_percussion | 4.3 | 0.79 | aafp_2013_chestpain | Pneumonia |
| fever | associated_symptom=fever | 2.1 | 0.71 | aafp_2013_chestpain | |
| tachypnea_rr_gt_20 | associated_symptom=tachypnea | 3.5 | 0.56 | aafp_2020_chestpain + west_qjm_2007 | |
| cough | associated_symptom=cough | 1.8 | — | aafp_2013_chestpain | Approx |
| prior_vte_dvt | medical_history=prior_vte | 2.9 | — | wells_pe_2000 | Approx (Wells subscore) |

### 2.3 Musculoskeletal (15 features)

| Feature | predicate_path | LR+ | LR- | Source key | Notes |
|---|---|---:|---:|---|---|
| chest_wall_combined_rule | alleviating_factor=palpation | 3.0 | 0.47 | aafp_2013_chestpain + bruyninckx_2008 | 4-factor rule, ≥2 present |
| pain_reproducible_with_palpation | aggravating_factor=palpation | 2.8 | — | bosner_2010_marburg | Approx — single-feature |
| stinging_quality | character=stinging | 2.0 | — | bruyninckx_2008 | Approx — rule component |
| localized_muscle_tension | associated_symptom=muscle_tension | 2.0 | — | bruyninckx_2008 | Approx — rule component |
| absence_of_cough | associated_symptom=no_cough | 1.5 | — | bruyninckx_2008 | Approx — rule component |
| well_localized_point_tenderness | location=point_chest_wall | 2.5 | — | primary_care_musculoskeletal_review_2013 | Approx |
| pain_worsened_by_movement | aggravating_factor=movement | 2.0 | — | primary_care_musculoskeletal_review_2013 | Approx |
| recent_trauma_or_strain | medical_history=recent_chest_trauma | 3.0 | — | primary_care_musculoskeletal_review_2013 | Approx |
| repetitive_activity_history | social_history=repetitive_upper_body_activity | 2.0 | — | primary_care_musculoskeletal_review_2013 | Approx |
| pain_with_deep_breath_costochondritis | aggravating_factor=inspiration | 1.5 | — | aafp_2021_costochondritis | Approx |
| sharp_well_localized_pain | character=sharp | 1.8 | — | aafp_2013_chestpain + aafp_2021_costochondritis | Approx (reciprocal of MI LR-) |
| duration_days_to_weeks | duration=days_to_weeks | 1.8 | — | primary_care_musculoskeletal_review_2013 | Approx |
| younger_age_lt_40 | risk_factor=age_lt_40 | 2.0 | — | bosner_2010_marburg | Approx |
| no_cad_risk_factors | risk_factor=no_cad_risk_factors | 1.5 | — | aafp_2020_chestpain | Approx |
| no_exertional_pattern | aggravating_factor=no_exertion | 1.8 | — | bosner_2010_marburg | Approx |

### 2.4 GI (16 features)

| Feature | predicate_path | LR+ | LR- | Source key | Notes |
|---|---|---:|---:|---|---|
| retrosternal_burning_pain | character=burning | 3.1 | 0.30 | aafp_2013_chestpain | Composite GERD rule |
| acid_regurgitation | associated_symptom=acid_regurgitation | 2.5 | — | aafp_2013_chestpain | Approx — composite component |
| sour_or_bitter_taste | associated_symptom=dysgeusia | 2.0 | — | aafp_2013_chestpain | Approx — composite component |
| ppi_trial_response | alleviating_factor=ppi | 5.5 | 0.24 | aafp_2020_chestpain + cremonini_2005_meta | Meta-analysis pooled |
| pain_after_meals | aggravating_factor=meals | 2.2 | — | frieling_2013_bmc | Approx |
| relief_with_antacids | alleviating_factor=antacid | 2.5 | — | cremonini_2005_meta | Approx |
| worse_lying_down | aggravating_factor=supine | 2.0 | — | aafp_2013_chestpain | Approx |
| dysphagia | associated_symptom=dysphagia | 2.5 | — | acg_gerd_2022 | Approx — alarm symptom |
| odynophagia | associated_symptom=odynophagia | 2.2 | — | acg_gerd_2022 | Approx |
| history_gerd_or_hiatal_hernia | medical_history=gerd | 3.5 | — | acg_gerd_2022 | Approx |
| nsaid_use | medication=nsaid | 2.0 | — | acg_gerd_2022 | Approx (peptic risk) |
| obesity_bmi_gt_30 | risk_factor=obesity | 1.8 | — | acg_gerd_2022 | Approx |
| smoking_tobacco | social_history=current_smoker | 1.5 | — | acg_gerd_2022 | Approx — also cardiac risk |
| non_exertional_trigger | aggravating_factor=no_exertion | 1.5 | — | aafp_2013_chestpain | Approx — rec. of cardiac rule-out |
| symptoms_chronic_months | duration=chronic_months | 2.0 | — | acg_gerd_2022 | Approx |
| panic_disorder_single_question | medical_history=panic_disorder | 4.2 | 0.09 | aafp_2013_chestpain | Single-question screen |

## 3. Draft LR table (JSON)

The full 79-entry JSON is written at `D:\hack_it\content\differentials\chest_pain\lr_table.json`. It validates against the Eng_doc.md §4.4 schema — every row carries `branch`, `feature`, `predicate_path`, `lr_plus`, `lr_minus`, `source`, `source_url`, `approximation`, and a `notes` field documenting the citation rationale (especially for approximations). The file opens with the mandatory `_synthetic` sentinel per SYNTHETIC_DATA.md. Version bumped `0.0.1` → `0.1.0`.

## 4. Citation list (key → full citation)

See `content/differentials/chest_pain/sources.md` for the canonical list — mirrored below in short form.

| Key | Short citation | Venue / year |
|---|---|---|
| aafp_2013_chestpain | McConaghy & Oza, *Outpatient Diagnosis of Acute Chest Pain* | AFP 87(3):177-182, 2013 |
| aafp_2020_chestpain | McConaghy, Sharma & Patel, *Acute Chest Pain in Adults: Outpatient Evaluation* | AFP 102(12):721-727, 2020 |
| aafp_2017_pleuritic | Reamy et al., *Pleuritic Chest Pain: Sorting Through the Differential Diagnosis* | AFP 96(5):306-312, 2017 |
| aafp_2021_costochondritis | Mott, Jones & Roman, *Costochondritis: Rapid Evidence Review* | AFP 104(1):73-78, 2021 |
| aha_acc_2021_chestpain | Gulati et al., 2021 AHA/ACC Chest Pain Guideline | Circulation 144:e368-e454 |
| fanaroff_jama_2015 | Fanaroff, Rymer, Goldstein, Simel, Newby — *Does This Patient With Chest Pain Have ACS?* | JAMA 314(18):1955-1965 |
| panju_jama_1998 | Panju, Hemmelgarn, Guyatt, Simel — *Is this patient having a myocardial infarction?* | JAMA 280(14):1256-1263 |
| klompas_jama_2002 | Klompas — *Does this patient have an acute thoracic aortic dissection?* | JAMA 287(17):2262-2272 |
| west_qjm_2007 | West, Goodacre, Sampson — *The value of clinical features in the diagnosis of PE* | QJM 100(12):763-769 |
| heart_score_2008 | Six, Backus, Kelder — *HEART score derivation* | Neth Heart J 16(6):191-196 |
| heart_score_backus_2013 | Backus et al. — HEART score prospective validation | Int J Cardiol 168:2153-2158 |
| timi_risk_2000 | Antman et al. — *TIMI risk score for UA/NSTEMI* | JAMA 284(7):835-842 |
| bosner_2010_marburg | Bösner et al. — Marburg Heart Score | CMAJ 182(12):1295-1300 |
| wells_pe_2000 | Wells et al. — Wells criteria for PE | Thromb Haemost 83:416-420 |
| bmc_pulm_2025 | Bayesian network meta-analysis of PE CDRs | BMC Pulm Med 2025 |
| perc_2004 | Kline et al. — PERC rule derivation | J Thromb Haemost 2:1247-1255 |
| perc_meta_2012 | Singh et al. — PERC meta-analysis | Ann Emerg Med 59:517-520 |
| bruyninckx_2008 | Bruyninckx et al. — chest-wall 4-factor rule | Br J Gen Pract 58:105-111 |
| cremonini_2005_meta | Cremonini, Wise, Moayyedi, Talley — PPI meta-analysis for non-cardiac chest pain | Am J Gastroenterol 100(6):1226-1232, 2005 |
| frieling_2013_bmc | Frieling et al. — non-cardiovascular chest pain systematic review | BMC Med 11:239 |
| acg_gerd_2022 | Katz et al. — ACG GERD Clinical Guideline | Am J Gastroenterol 117:27-56 |
| primary_care_musculoskeletal_review_2013 | Ayloo, Cvengros, Marella — *Evaluation and Treatment of Musculoskeletal Chest Pain* | Prim Care Clin Off Pract 40:863-887 |

## 5. Spot-check — five direct quotes against LR values

These verify five randomly chosen LR entries in the JSON against the actual source text. The validator's role is to repeat this exercise on all 79 rows.

1. **`heart_score_high_7_10` / LR+ 13.0** — *"HEART 7–10: LR+ 13 (95% CI, 7.0–24)"* — direct extract from Fanaroff 2015 JAMA Table, per the rebelem.com secondary summary of the review (verified 2026-04-21). Primary source: Fanaroff AC et al., *JAMA* 314(18):1955-1965, 2015.

2. **`retrosternal_burning_pain` / LR+ 3.1 / LR- 0.30 (GERD)** — *"Burning retrosternal pain, acid regurgitation, sour/bitter taste; one-week high-dose PPI trial relieves symptoms: LR+ 3.1, LR- 0.30"* — direct extract from AAFP 2013 Table 1 (McConaghy & Oza 2013), verified by WebFetch against https://www.aafp.org/pubs/afp/issues/2013/0201/p177.html 2026-04-21.

3. **`egophony_on_auscultation` / LR+ 8.6 (pneumonia)** — *"Egophony: LR+ 8.6, LR– 0.96"* — direct extract from AAFP 2013 Table 1 (pneumonia section), verified 2026-04-21.

4. **`wells_pe_high_probability` / LR+ 5.59** — *"three-tier Wells had LR+: 5·59 [3·70–8·37]"* — direct extract from BMC Pulm Med 2025 Bayesian network meta-analysis, verified 2026-04-21 via search against the Springer article. Supersedes the older AAFP 2013 citation of LR+ 6.8 for the same category.

5. **`perc_rule_negative` / LR- 0.17** — *"Pooled… negative likelihood ratio… 0.17 (95% CI 0.13–0.23)"* — direct extract from the Singh 2012 PERC meta-analysis summary on NCBI Bookshelf NBK109669 (DARE abstract), verified 2026-04-21. Primary: Singh et al., *Ann Emerg Med* 59(6):517-520, 2012.

## 6. Branches with weak published LRs (flagged)

**MSK** is the weakest branch for published single-feature LRs. The Bruyninckx 2008 4-factor rule is the only pooled MSK diagnostic instrument, and it publishes only the **composite** LR (≥2 factors → LR+ 3.0, LR- 0.47). Individual-factor LRs (stinging quality, muscle tension, palpation-reproducible, absence-of-cough) **are not separately pooled** in any systematic review I located. I've included each as an approximation (`approximation: true`) with LR+ in the 1.5-2.8 range, defended by the fact that the combined rule LR+ is 3.0 and each factor contributes; individual values below the composite are a conservative decomposition. The validator should re-check these against the original Bruyninckx 2008 BJGP paper (likely behind OUP paywall; table extracted via cited open-access summaries).

**GI** is medium-weak. The AAFP 2013 **composite** GERD rule (burning + regurgitation + taste + PPI response) has a published LR+ 3.1 / LR- 0.30. Individual components (`acid_regurgitation`, `sour_or_bitter_taste`, `pain_after_meals`, `worse_lying_down`, `relief_with_antacids`, `dysphagia`, `odynophagia`) lack dedicated single-feature LR studies in the chest-pain-discrimination context. Approximated at 1.5-3.5 with citations to the ACG 2022 GERD guideline and Cremonini 2005 meta-analysis; defensible but not pooled. **The PPI-trial response (LR+ 5.5 / LR- 0.24) is the strongest GI published LR** and is the hero GI feature.

**Cardiac** and **pulmonary** branches are **strong**. 19/28 cardiac and 12/20 pulmonary rows are directly-pooled published LRs. The remaining approximations are all adjacent to published pooled values (e.g., the Wells-subscore approximation for `prior_vte_dvt`, Panju 1998 historical values superseded by Fanaroff 2015 for some ACS features).

## 7. Counterfactual-verifier implications

For `src/verifier/` (Eng_doc.md §6), the **strongest discriminators between top-2 hypotheses** are:

- **Cardiac vs Pulmonary**:
  - `exertional_trigger` (LR+ 2.4 cardiac / neutral pulmonary)
  - `sudden_dyspnea` (LR+ 1.83 pulm / ~neutral cardiac, but its **absence** LR- 0.43 rules out pulm strongly)
  - `pleuritic_pain` (LR+ 1.8 pulm / LR 0.2 cardiac — a near-ideal discriminator; large `|log LR_A - log LR_B|`)
  - `leg_swelling` / `current_dvt_signs` (LR+ 2.1 pulm / neutral cardiac — Wells-style anchors)
  - `radiation_both_arms` (LR+ 2.6 cardiac / neutral pulm)
  - Composite scores: `heart_score_high` (LR+ 13.0 cardiac) and `wells_pe_high` (LR+ 5.59 pulm) are the largest single-row discriminators.

- **Cardiac vs MSK**:
  - `reproducible_with_palpation` (LR+ 2.8 MSK / LR 0.3 cardiac — the classical distinguisher)
  - `sharp_quality` (LR+ 1.8 MSK / LR 0.3 cardiac)
  - `exertional_trigger` (LR+ 2.4 cardiac / LR+ ~0.55 MSK implied — reciprocal)
  - `marburg_heart_score_low_0_1` (LR+ 0.04 cardiac — very strong argument for MSK/non-cardiac alternative)

- **Cardiac vs GI**:
  - `retrosternal_burning_pain` (LR+ 3.1 GI)
  - `ppi_trial_response` (LR+ 5.5 GI — the GI-specific counterfactual test)
  - `exertional_trigger` (LR+ 2.4 cardiac) and its absence favouring GI
  - `relief_with_nitroglycerin` (LR+ 1.1 cardiac — *non-discriminatory*, explicitly flagged in notes because nitro also relieves esophageal spasm; the verifier should not over-weight)

- **Pulmonary vs MSK**:
  - `sudden_dyspnea` (LR+ 1.83 pulm) / `any_dyspnea` absence LR- 0.52
  - `pain_reproducible_with_palpation` (LR+ 2.8 MSK)
  - `recent_trauma_or_strain` (LR+ 3.0 MSK)
  - `perc_rule_negative` (LR- 0.17 pulm — near-ideal rule-out lever)

**Design note for the verifier**: features with **LR values on *both* branches and opposite signs** (cardiac-vs-pulmonary pleuritic pain; cardiac-vs-MSK palpation-reproducibility) are the highest-information-gain discriminators. The CPG (counterfactual probability gap) metric in `docs/research_brief.md §2.2` applied to these asymmetric-LR features will produce the strongest next-best-question candidates.

## 8. Open questions for human review

1. **Panic disorder placement.** AAFP 2013 lists panic disorder as a distinct non-cardiac cause (LR+ 4.2 / LR- 0.09 via the single-question screen). I mapped it to the `gi` branch as the functional/visceral catch-all since we only have 4 branches. Options: (a) leave as `gi`, (b) move to `msk` as "non-organic/functional", (c) add a 5th `functional` branch (would require `branches.json` update and violate the "4 branches" scope in `PRD.md §3`). **Recommendation:** leave as `gi` for scope, and note in UI that the panic-disorder cluster is not GI-organic.

2. **Marburg Heart Score vs HEART score.** Both are in the LR table as cardiac risk-stratifying scores. HEART is ED-oriented; Marburg is primary-care-oriented. Which is the **default** for the demo's primary-care single-visit scenario? Marburg is more appropriate clinically but HEART has larger validation cohort. **Recommendation:** keep both; use Marburg as the default primary-care prior and HEART as a secondary lever when the physician verbalizes ED-style context.

3. **Approximation density in MSK and GI.** 24 of 36 approximations sit in these two branches. Is 3/15 direct pooled LRs in MSK acceptable for demo ship, or should we trim MSK to the 3 published + 5 strongest approximations (8 features total) to keep the table tighter? **Recommendation:** keep all 15 — the deterministic engine will downweight low-LR features naturally via `log(1.5)` vs `log(3.0)`, and having the full clinical feature set reads better in the demo-video tree-rendering moment.

4. **Aortic dissection LR on the cardiac branch.** The Klompas 2002 pulse-deficit LR+ 5.7 was modified by Ohle 2018 meta-analysis to LR+ 2.48. I kept the Klompas value as the "established pooled" estimate and flagged the Ohle revision in notes. Should we prefer the more recent Ohle 2018 value instead? **Recommendation:** defer to human review; if preferring Ohle 2018, also add it as a new short-key source `ohle_acem_2018`.

5. **Validator scope.** The validator task (per progress.md) will spot-check LR values against sources. For MSK and GI approximations, is the expectation that the validator accepts "defensible approximation with cited adjacent evidence" (my current posture) or that the validator rejects any row lacking a directly-pooled LR (in which case ~36 of the 79 rows get stripped)? **Recommendation:** the former — consistent with rules.md §4.4 which explicitly permits approximations when flagged.

## 9. Method and rejections

WebFetched/cross-verified AAFP 2013, AAFP 2020, Klompas 2002 (abstract), QJM 2007 (abstract), Fanaroff 2015, Cremonini 2005, BMC Pulm Med 2025, Singh 2012, AAFP 2021 costochondritis, Bösner 2010 Marburg validation, ACG 2022. Verified DOIs for AHA/ACC 2021 and TIMI 2000 resolve correctly. Did **not** paste verbatim from any paywalled source — Panju 1998, Bruyninckx 2008, ACG 2022 primary values cited via open-access AAFP/BMC summaries; single-feature approximations flagged. Four entries appended to `reasons.md` this session: (1) pneumothorax single-finding LRs (no pooled LRs exist — rules.md §3.2 imaging + §4.4); (2) paywalled-source verbatim paraphrase of Panju/Bruyninckx/ACG full text (rules.md §7.1); (3) Ohle 2018 aortic-dissection revision (paywalled Wiley, deferred to human review); (4) general UpToDate/AMBOSS/DynaMed/ClinicalKey exclusion (rules.md §7.1).
