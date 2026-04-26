RobbyMD is a doctor-steered diagnostic trace system for live clinical encounters. A clinical note preserves the conclusion; RobbyMD preserves the path.

During a doctor-patient conversation, RobbyMD converts speech into timestamped turns, extracts structured clinical claims, tracks corrections through an append-only supersession graph, projects active evidence into a working differential, suggests discriminating next questions, and generates SOAP only as a provenance-backed downstream artifact.

The physician remains in control. RobbyMD does not diagnose, decide, or replace clinical judgment. It keeps the relevant clues visible: what the patient said, which evidence is active, what was ruled down, what was corrected, what is missing, and why a question matters.

Claude Opus 4.7 is load-bearing but bounded: it extracts claims, phrases next questions, drafts SOAP with claim markers, and powers five post-encounter Managed Agents that query the same trace for aftercare, handoff, bias review, and provenance-preserving note edits.

The core invariant is simple: nothing disappears. Every claim, correction, question, and SOAP sentence traces back to the source turn and transcript span. The demo uses synthetic/anonymized clinical scripts only. RobbyMD is a research prototype, not a medical device; the physician makes every clinical decision.
