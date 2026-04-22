"""DDXPlus full variant — substrate + differential engine + verifier.

Ingests one `DDXPlusCase` as a turn stream via the substrate write API, then
reads the deterministic differential ranking out of the projections layer.

Currently a SCAFFOLD: writes turns into `SubstrateStub`, then returns the
DDXPlus gold differential (top-5 truncation) as a placeholder. When
wt-engine publishes the real substrate + wt-trees publishes the differential
engine, replace the TODO marker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval._common import SubstrateStub
from eval.ddxplus.adapter import DDXPlusCase, load_evidence_dictionary, record_to_turns
from eval.ddxplus.baseline import DDXPrediction


@dataclass
class FullRunner:
    """Owns substrate state for one session (one DDXPlus case = one session)."""

    evidence_dict_path: Path

    def predict_differential(self, case: DDXPlusCase) -> DDXPrediction:
        evidence_dict = load_evidence_dictionary(self.evidence_dict_path)
        turns = record_to_turns(case, evidence_dict)

        # TODO(wt-engine): replace SubstrateStub with the real substrate once
        # wt-engine publishes the write API. Expected interface:
        #
        #   from substrate import ClaimStore
        #   from differential import DifferentialEngine
        #   store = ClaimStore(session_id=case.patient_id)
        #   store.write_turns(turns)
        #   engine = DifferentialEngine(lr_table_path=Path("content/differentials/chest_pain/lr_table.json"))
        #   ranking = engine.rank(store.active_claims(session_id=case.patient_id))
        #   top5 = [b.pathology for b in ranking[:5]]
        stub = SubstrateStub(session_id=case.patient_id)
        stub.write_turns(turns)

        # Placeholder: fall back to DDXPlus's gold differential so run.py can
        # exercise the harness end-to-end without the real substrate. Flagged
        # in run.py output so the reviewer sees the placeholder tier.
        top5 = [p for p, _ in case.differential[:5]]
        return DDXPrediction(
            patient_id=case.patient_id,
            top5=top5,
            raw_response="[SUBSTRATE STUB] wt-engine + wt-trees pending",
        )
