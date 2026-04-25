"""D11 differential compiler — pure-Python evidence pipeline.

Modules:
  types                             — shared dataclasses
  clinical_abstraction              — Qwen3-32B clinical case abstraction
  differential_board                — 3-role parallel reasoning board
  pairwise_discriminator_tournament — deterministic pair selection + discriminators
  targeted_discriminator_repair     — repair unresolved/missing discriminators
  reader                            — Gemini reader with compiled evidence bundle
  candidate_hypothesis_adapter      — MedXpertQA option → CandidateHypothesis
  evidence_attributor               — board role outputs → per-candidate evidence
  strict_sufficiency_auditor        — structural evidence-graph audit
  final_evidence_bundle             — assemble reader-facing text bundle
"""
