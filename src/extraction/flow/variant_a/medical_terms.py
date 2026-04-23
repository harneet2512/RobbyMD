"""Fixed vocabulary used to compute medical-term WER.

Lower-case stems only — the measurement strips punctuation before comparing.
Additions tracked against predicate_packs/clinical_general/differentials/
feature names, so the term list biases towards what the differential engine
cares about.
"""

MEDICAL_TERMS = frozenset({
    # cardiac
    "chest", "pain", "pressure", "dyspnea", "dyspnoea", "palpitations",
    "troponin", "aspirin", "nitroglycerin", "myocardial", "infarction",
    "angina", "radiates", "radiation", "heart", "timi", "pericarditis",
    "tamponade", "ecg", "ekg",
    # pulmonary
    "embolism", "pneumonia", "asthma", "wheezing", "tachypnea", "tachypnoea",
    "pleuritic", "perc", "wells",
    # gi
    "cholecystitis", "appendicitis", "diverticulitis", "bowel", "obstruction",
    "murphy", "mcburney", "guarding", "rebound", "rigidity", "periumbilical",
    # neuro
    "migraine", "subarachnoid", "hemorrhage", "haemorrhage", "aphasia",
    "seizure",
    # vestibular / syncope
    "vasovagal", "orthostatic", "bppv", "arrhythmia", "syncope", "presyncope",
    "dizziness", "vertigo",
    # general
    "hypertension", "diabetes", "ibuprofen", "metformin", "amlodipine",
    "atorvastatin", "statin", "lisinopril", "losartan", "metoprolol",
    "clopidogrel", "heparin", "apixaban", "enoxaparin", "albuterol",
    "furosemide", "acetaminophen", "ceftriaxone", "metronidazole",
})
