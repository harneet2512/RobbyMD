"""
Fuzzy medical term correction via rapidfuzz (MIT).

Zero-VRAM replacement for the BioMistral-7B-DARE cleanup LLM.
Vocabulary loaded from predicate pack files at runtime (no hardcoded terms).
"""
from __future__ import annotations

from typing import List, Tuple

from rapidfuzz import fuzz, process


def _normalize_token(s: str) -> str:
    return s.lower().strip(".,;:!?\"'()[]")


class MedicalCorrector:
    """Edit-distance corrector driven by an external vocabulary list.

    Loaded from a predicate pack's correction_vocab.txt by ShipPipeline.
    Two passes: bigram (multi-word terms) then single-word, with plural
    guard and minimum-length filter.
    """

    def __init__(self, vocabulary: List[str]):
        self._vocab_lookup = {term.lower(): term for term in vocabulary}
        self._vocab_words = list(self._vocab_lookup.keys())
        self._single_word_vocab = [t for t in self._vocab_words if " " not in t]
        self._multi_word_vocab = [t for t in self._vocab_words if " " in t]

    def correct(self, text: str, threshold: int = 88) -> Tuple[str, List[dict]]:
        """Fuzzy-match words/bigrams against the loaded vocabulary.

        Threshold 88 tuned to catch mangled terms like 'addorvastatin'
        (ratio 88.0) while suppressing 'giving' → 'IVIG' (ratio 80.0).
        """
        words = text.split()
        if not words:
            return text, []

        corrections: List[dict] = []
        corrected_words = list(words)
        used_indices: set[int] = set()

        # Pass 1 — bigrams against multi-word vocab only
        for i in range(len(words) - 1):
            if i in used_indices or (i + 1) in used_indices:
                continue
            bigram = _normalize_token(f"{words[i]} {words[i + 1]}")
            if len(bigram) < 6:
                continue
            match = process.extractOne(
                bigram, self._multi_word_vocab, scorer=fuzz.ratio, score_cutoff=threshold
            )
            if match and match[0] != bigram:
                original_form = self._vocab_lookup[match[0]]
                original_bigram = f"{words[i]} {words[i + 1]}"
                replacement_words = original_form.split()
                corrected_words[i] = replacement_words[0]
                corrected_words[i + 1] = " ".join(replacement_words[1:])
                used_indices.add(i)
                used_indices.add(i + 1)
                corrections.append({
                    "original": original_bigram,
                    "corrected": original_form,
                    "score": match[1],
                    "position": i,
                })

        # Pass 2 — single words
        for i in range(len(corrected_words)):
            if i in used_indices:
                continue
            raw = corrected_words[i]
            stripped = raw.lower().strip(".,;:!?")
            if len(stripped) < 4:
                continue
            match = process.extractOne(
                stripped, self._single_word_vocab, scorer=fuzz.ratio, score_cutoff=threshold
            )
            if match and match[0] != stripped:
                if stripped.endswith("s") and stripped[:-1] == match[0]:
                    continue
                if match[0].endswith("s") and match[0][:-1] == stripped:
                    continue
                original_form = self._vocab_lookup[match[0]]
                prefix = ""
                suffix = ""
                body = raw
                while body and not body[0].isalnum():
                    prefix += body[0]
                    body = body[1:]
                while body and not body[-1].isalnum():
                    suffix = body[-1] + suffix
                    body = body[:-1]
                corrected_words[i] = prefix + original_form + suffix
                corrections.append({
                    "original": words[i],
                    "corrected": corrected_words[i],
                    "score": match[1],
                    "position": i,
                })

        result = " ".join(w for w in corrected_words if w)
        return result, corrections


# Backwards-compatible module-level function for tests and measure.py imports.
# Uses the hardcoded default vocabulary (same terms as clinical_general pack).
_DEFAULT_VOCABULARY: List[str] = [
    "chest pain", "angina", "myocardial infarction", "troponin", "pericarditis",
    "tamponade", "aortic dissection", "STEMI", "NSTEMI", "arrhythmia",
    "atrial fibrillation", "tachycardia", "bradycardia", "palpitations",
    "nitroglycerin", "aspirin", "clopidogrel", "heparin", "enoxaparin",
    "HEART score", "TIMI", "electrocardiogram", "echocardiogram",
    "pulmonary embolism", "pneumonia", "pneumothorax", "dyspnea", "tachypnea",
    "pleuritic", "hemoptysis", "bronchospasm", "asthma", "COPD",
    "Wells criteria", "PERC rule", "D-dimer", "CT angiography",
    "oxygen saturation", "SpO2", "wheezing", "crackles", "rales",
    "cholecystitis", "appendicitis", "diverticulitis", "pancreatitis",
    "bowel obstruction", "peritonitis", "Murphy sign", "McBurney point",
    "guarding", "rebound tenderness", "rigidity",
    "omeprazole", "pantoprazole", "metoclopramide",
    "migraine", "subarachnoid hemorrhage", "subdural hematoma",
    "meningitis", "aphasia", "dysarthria", "photophobia", "phonophobia",
    "nuchal rigidity", "Kernig sign", "Brudzinski sign",
    "vasovagal", "orthostatic", "presyncope", "syncope", "BPPV",
    "Dix-Hallpike", "Epley maneuver",
    "Kawasaki", "Kawasaki disease", "strawberry tongue", "desquamation",
    "febrile", "febrile seizure", "immunoglobulin", "IVIG",
    "acetaminophen", "ibuprofen", "Tylenol", "Motrin",
    "hypertension", "hypotension", "auscultation", "palpation", "percussion",
    "metformin", "amlodipine", "atorvastatin", "lisinopril", "losartan",
    "metoprolol", "warfarin", "rivaroxaban", "apixaban",
    "prednisone", "dexamethasone", "albuterol",
    "differential diagnosis", "chief complaint", "history of present illness",
    "review of systems", "physical examination", "assessment", "plan",
    "bilateral", "unilateral", "proximal", "distal", "anterior", "posterior",
    "acute", "chronic", "exacerbation", "remission",
]

_DEFAULT_CORRECTOR = MedicalCorrector(vocabulary=_DEFAULT_VOCABULARY)

# Kept for backwards compatibility with measure.py MEDICAL_TERMS_SET
MEDICAL_VOCABULARY = _DEFAULT_VOCABULARY


def correct_medical_terms(text: str, threshold: int = 88) -> Tuple[str, List[dict]]:
    """Module-level convenience wrapper using the default vocabulary."""
    return _DEFAULT_CORRECTOR.correct(text, threshold=threshold)
