# TODO (L4 operator, Step 3): OVERWRITE this file verbatim with Bundle 4's
# src/extraction/flow/variant_a/medical_terms.py so Variants A and B score
# against the identical MEDICAL_TERMS set. The A-vs-B comparison is invalid
# if the medical-term WER sub-metric uses different term lists.
#
# This file exists as a placeholder so that measure.py imports cleanly on the
# laptop scaffold session. The terms below are a rough cross-section drawn
# from predicate_packs/clinical_general/differentials/chest_pain/branches.json
# and common ACI-Bench medical vocabulary. DO NOT ship the A-vs-B comparison
# with these placeholder terms — swap in Bundle 4's authoritative set first.

MEDICAL_TERMS: set[str] = {
    # Cardiac
    "chest", "pain", "angina", "ischemia", "infarction", "mi", "stemi",
    "nstemi", "acs", "troponin", "cad", "dissection", "aortic",
    "nitroglycerin", "diaphoresis", "diaphoretic", "radiation", "s3", "s4",
    "gallop",
    # Pulmonary
    "dyspnea", "dyspnoea", "pleuritic", "embolism", "pe", "pneumonia",
    "pneumothorax", "hemoptysis", "haemoptysis", "spo2", "oxygen",
    "respiratory", "tachypnea", "tachypnoea",
    # GI
    "abdominal", "nausea", "vomiting", "emesis", "diarrhea", "diarrhoea",
    "melena", "haematemesis",
    # Neuro
    "headache", "syncope", "dizziness", "vertigo", "seizure",
    # Vitals
    "bp", "sbp", "dbp", "hr", "heart-rate", "blood-pressure", "pulse",
    "temp", "temperature", "afebrile", "febrile",
    # Labs / imaging
    "bnp", "ecg", "ekg", "ct", "mri", "xray", "x-ray", "cbc", "bmp", "cmp",
    # History / comorbidity
    "htn", "hypertension", "dm", "dm2", "diabetes", "copd", "asthma",
    "chf", "smoker", "smoking", "pack-year", "pack-years",
    # Medications (common)
    "aspirin", "metoprolol", "lisinopril", "atorvastatin", "furosemide",
    "heparin", "warfarin", "insulin",
}
