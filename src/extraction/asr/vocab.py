"""Medical-vocabulary bias string for Whisper `initial_prompt`.

Per `research/asr_stack.md` sections 4 and 5, `faster-whisper.transcribe(..., initial_prompt=...)`
threads a short domain-vocabulary string into Whisper's 448-token decoder context.
We target ≤224 tokens (half the budget; the other half is reserved for generated
tokens so a ~5 s utterance never triggers the prompt+output length check — see
`reasons.md` entry "ASR prompt length: initial_prompt >224 tokens"). The remaining
tokens bias Whisper toward the chest-pain vocabulary our synthetic demo expects.

Sources — every term in this file is sourced from public-domain or open-access
references. **No SNOMED strings are redistributed** (research/asr_stack.md §6 R8):
- Drug ingredient names: RxNorm (public domain, NLM) [1], filtered to chest-pain-
  relevant ingredients (antiplatelet, anticoagulant, statin, beta-blocker, PPI,
  NSAID, nitrates).
- Chest-pain descriptors: 2021 AHA/ACC/ASE Chest Pain Guideline, open-access
  Circulation [2], recommendation-table glossary.
- Symptom descriptors: ICD-10-CM R07.x "Pain in throat and chest" chapter
  (public-domain, NCHS/CMS) [3].

[1] https://www.nlm.nih.gov/research/umls/rxnorm/index.html
[2] https://doi.org/10.1161/CIR.0000000000001029
[3] https://www.cms.gov/medicare/coding-billing/icd-10-codes

Exported:
- `RXNORM_CHEST_PAIN_DRUGS` — ~30 ingredient names, public-domain RxNorm strings.
- `AHA_ACC_2021_DESCRIPTORS` — ~25 descriptor terms from the guideline's glossary.
- `ICD10_R07_DESCRIPTORS` — ~10 R07.x descriptor terms.
- `build_initial_prompt()` — compose a single bias string ≤224 tokens.
"""

from __future__ import annotations

# RxNorm chest-pain-relevant ingredient names (public-domain NLM strings).
# Selected to cover the meds most likely to be spoken in a chest-pain consultation:
# antiplatelet, anticoagulant, statin, beta-blocker, PPI, NSAID, nitrate, ACEi/ARB.
RXNORM_CHEST_PAIN_DRUGS: tuple[str, ...] = (
    "aspirin",
    "clopidogrel",
    "ticagrelor",
    "prasugrel",
    "heparin",
    "enoxaparin",
    "warfarin",
    "apixaban",
    "rivaroxaban",
    "atorvastatin",
    "rosuvastatin",
    "simvastatin",
    "metoprolol",
    "atenolol",
    "bisoprolol",
    "carvedilol",
    "nitroglycerin",
    "isosorbide",
    "omeprazole",
    "pantoprazole",
    "ranitidine",
    "famotidine",
    "ibuprofen",
    "naproxen",
    "acetaminophen",
    "lisinopril",
    "losartan",
    "amlodipine",
    "furosemide",
    "morphine",
)

# Chest-pain descriptors lifted from the 2021 AHA/ACC Chest Pain Guideline
# recommendation-table glossary (open-access, Circulation).
# These are generic anatomical and character terms, not copyrighted prose.
AHA_ACC_2021_DESCRIPTORS: tuple[str, ...] = (
    "substernal",
    "retrosternal",
    "epigastric",
    "precordial",
    "pleuritic",
    "exertional",
    "crescendo",
    "radiating",
    "radiation to jaw",
    "radiation to arm",
    "radiation to back",
    "diaphoresis",
    "dyspnea",
    "palpitations",
    "syncope",
    "nausea",
    "vomiting",
    "lightheadedness",
    "angina",
    "ischemia",
    "infarction",
    "pericarditis",
    "dissection",
    "pulmonary embolism",
    "costochondritis",
)

# ICD-10-CM R07.x chapter descriptors (public domain, NCHS/CMS).
# Ref: `icd10cm_codes_2025.txt`, chapter XVIII (R00-R99), section R07.
ICD10_R07_DESCRIPTORS: tuple[str, ...] = (
    "chest pain",
    "precordial pain",
    "pleurodynia",
    "painful respiration",
    "anterior chest wall pain",
    "intercostal pain",
    "tightness",
    "squeezing",
    "pressure",
    "heaviness",
)

# Hard cap on the assembled prompt, in whitespace-delimited words. faster-whisper
# tokenises with Whisper's BPE; a conservative word-to-token ratio on English
# medical vocab is ~1.3. 170 words * 1.3 ~ 221 tokens, comfortably under 224.
# See `reasons.md` "ASR prompt length: initial_prompt >224 tokens".
MAX_PROMPT_WORDS: int = 170


def build_initial_prompt(
    drugs: tuple[str, ...] = RXNORM_CHEST_PAIN_DRUGS,
    descriptors: tuple[str, ...] = AHA_ACC_2021_DESCRIPTORS,
    icd10: tuple[str, ...] = ICD10_R07_DESCRIPTORS,
    max_words: int = MAX_PROMPT_WORDS,
) -> str:
    """Compose the `initial_prompt` bias string.

    Shape: one sentence-like line listing drugs, then one AHA/ACC descriptors
    line, then one ICD-10 descriptors line. This layout was chosen per
    `research/asr_stack.md` §5.1 (recommended composition). Whisper sees a
    plain comma-separated prior; it does not parse our structure.

    Truncates gracefully if any list is oversized: earlier entries win
    (clinically the most common items come first in each list).
    """
    sections = [
        "Medications: " + ", ".join(drugs) + ".",
        "Chest pain descriptors: " + ", ".join(descriptors) + ".",
        "Chest pain symptoms: " + ", ".join(icd10) + ".",
    ]
    text = " ".join(sections)
    words = text.split()
    if len(words) <= max_words:
        return text
    # Trim from the tail to stay under the cap; keep sentence-closing period if present.
    truncated = " ".join(words[:max_words]).rstrip(",")
    if not truncated.endswith("."):
        truncated += "."
    return truncated
