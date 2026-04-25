"""LongMemEval-S full variant — Opus 4.7 + substrate-backed retrieval.

# STALE: This module uses SubstrateStub and falls back to baseline.
# The canonical substrate runtime path is:
#   eval/longmemeval/pipeline.py::run_substrate_case()
# Do not add new logic here. Wire through the canonical path instead.

Each question's `haystack_sessions` is written to the substrate; the
retrieval layer is queried for the top-k relevant claims given the question;
Opus 4.7 answers with those claims as context (not the full haystack).

Currently a SCAFFOLD pending wt-engine's substrate write + retrieval API.
The stub writes turns into `SubstrateStub` and returns the baseline answer so
run.py produces sensible outputs end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass

from eval._common import SubstrateStub
from eval.longmemeval.adapter import LongMemEvalQuestion, session_to_turns
from eval.longmemeval.baseline import LongMemEvalPrediction
from eval.longmemeval.baseline import predict_answer as baseline_predict


@dataclass
class FullRunner:
    """Per-question substrate + retrieval harness."""

    def predict_answer(self, q: LongMemEvalQuestion) -> LongMemEvalPrediction:
        # TODO(wt-engine): replace SubstrateStub with the real substrate once
        # wt-engine publishes the write + retrieval API. Expected interface:
        #
        #   store = ClaimStore(session_id=q.question_id)
        #   for sidx in range(len(q.haystack_sessions)):
        #       store.write_turns(session_to_turns(q, sidx))
        #   retriever = SubstrateRetriever(store)
        #   context = retriever.top_k(query=q.question, k=16)
        #   return opus_answer(context, q.question)
        stub = SubstrateStub(session_id=q.question_id)
        for sidx in range(len(q.haystack_sessions)):
            stub.write_turns(session_to_turns(q, sidx))

        # Placeholder: fall through to baseline (full-context Opus) so we have
        # something to score. The real variant will pass substrate-retrieved
        # claims instead of the full haystack.
        pred = baseline_predict(q)
        wrapped = LongMemEvalPrediction(
            question_id=pred.question_id,
            question_type=pred.question_type,
            predicted_answer=pred.predicted_answer,
            raw_response="[SUBSTRATE STUB] wt-engine retrieval pending\n" + pred.raw_response,
        )
        # Bypass detection (orchestrator CRITICAL CONSTRAINT, Worker 1):
        # `--variant full` MUST exercise the real substrate. The
        # `[SUBSTRATE STUB]` sentinel above means we degraded to baseline.
        # Failing loudly here prevents accidental publication of baseline
        # numbers labelled as substrate. Wire the real substrate (Worker 3
        # event tuples + shared backend in eval/_substrate_backend.py) to
        # remove the sentinel and let this assertion pass silently.
        assert "[SUBSTRATE STUB]" not in wrapped.raw_response, (
            "Bypass detected: --variant full fell back to baseline. "
            "Wire the real substrate (Worker 3 + shared backend) before "
            "running LongMemEval --variant full."
        )
        return wrapped
