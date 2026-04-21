# Sources — Chest Pain Differential

Every row in `lr_table.json` cites a peer-reviewed or guideline source. This file lists the canonical primary references and the short-form keys used in `source` fields.

## Primary references

| Short key | Full citation | URL / DOI | License |
|---|---|---|---|
| `aafp_2013_chestpain` | Cayley WE Jr. *Diagnosing the Cause of Chest Pain*. American Family Physician, 2005 + 2013 update. | https://www.aafp.org/pubs/afp/issues/2013/0201/p177.html | Open access (AAFP) |
| `aha_acc_2021_chestpain` | Gulati M et al. *2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation and Diagnosis of Chest Pain*. Circulation, 2021. | https://doi.org/10.1161/CIR.0000000000001029 | Open (AHA/ACC) |
| `heart_score_2008` | Six AJ et al. *Chest pain in the emergency room: value of the HEART score*. Netherlands Heart Journal, 2008. | https://doi.org/10.1007/BF03086144 | Peer-reviewed |
| `timi_risk_2000` | Antman EM et al. *The TIMI risk score for unstable angina/non–ST elevation MI*. JAMA, 2000. | https://doi.org/10.1001/jama.284.7.835 | Peer-reviewed |
| `wells_pe_2000` | Wells PS et al. *Derivation of a simple clinical model to categorize patients probability of pulmonary embolism*. Thromb Haemost, 2000. | https://doi.org/10.1055/s-0037-1613830 | Peer-reviewed |
| `perc_2004` | Kline JA et al. *Clinical criteria to prevent unnecessary diagnostic testing in emergency department patients with suspected pulmonary embolism*. J Thromb Haemost, 2004. | https://doi.org/10.1111/j.1538-7836.2004.00887.x | Peer-reviewed |
| `liu_jeccm_2021` | Liu et al. *A systematic review of HEART / TIMI / Wells / PERC performance*. Journal of Emergency and Critical Care Medicine, 2021. | https://jeccm.amegroups.org/article/view/4088/html | Open-access |

## Usage

In `lr_table.json`, each row's `source` field names a short key above, optionally with a `§` suffix pointing to a specific section or table:

```json
{
  "branch": "cardiac",
  "feature": "exertional_trigger",
  "predicate_path": "aggravating_factor=exertion",
  "lr_plus": 2.4,
  "lr_minus": 0.6,
  "source": "aha_acc_2021_chestpain §4.2",
  "approximation": false
}
```

## Rules

- **Never invent a number.** If a value isn't in a peer-reviewed source, don't add it. See `rules.md §4.4, §7.4`.
- **No paywalled-content verbatim.** UpToDate / AMBOSS / DynaMed / ClinicalKey cannot be paraphrased into rows with their specific wording. Reference only peer-reviewed or open guideline content. See `rules.md §7.1`.
- **Approximations** allowed if sourced and flagged: `"approximation": true` plus a visual indicator in the UI.
