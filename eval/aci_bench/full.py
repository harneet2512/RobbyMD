"""ACI-Bench full variant — substrate + provenance-validated note generator.

Dialogue → substrate writes → claim extraction → note generator. Every
generated sentence has `source_claim_ids` chain; sentences without valid
provenance are rejected by the validator and never shown
(Eng_doc.md §5.6, rules.md §4.2).

Currently a SCAFFOLD pending wt-engine + wt-extraction + wt-trees. The stub
writes turns into `SubstrateStub` and falls back to baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

from eval._common import SubstrateStub
from eval.aci_bench.adapter import ACIEncounter, encounter_to_turns
from eval.aci_bench.baseline import ACINotePrediction
from eval.aci_bench.baseline import predict_note as baseline_predict


@dataclass
class FullRunner:
    def predict_note(self, enc: ACIEncounter) -> ACINotePrediction:
        # TODO(wt-engine+wt-extraction+wt-trees): swap SubstrateStub for the
        # real pipeline once the three upstream worktrees publish interfaces.
        # Expected chain:
        #
        #   store = ClaimStore(session_id=enc.encounter_id)
        #   store.write_turns(turns)
        #   extractor = ClaimExtractor(opus_client)
        #   extractor.run_on(store)                 # populates claims from turns
        #   note_gen = NoteGenerator(opus_client)
        #   note = note_gen.compose(store.active_claims(session_id=enc.encounter_id))
        #   validator = ProvenanceValidator()
        #   final_note = validator.filter(note)     # drops sentences without provenance
        stub = SubstrateStub(session_id=enc.encounter_id)
        stub.write_turns(encounter_to_turns(enc))

        # Placeholder: fall back to baseline so the metrics pipeline has
        # something to score.
        pred = baseline_predict(enc)
        return ACINotePrediction(
            encounter_id=pred.encounter_id,
            predicted_note=pred.predicted_note,
            raw_response="[SUBSTRATE STUB] wt-engine+wt-extraction+wt-trees pending\n" + pred.raw_response,
        )
