"""Prompt templates for the counterfactual verifier.

One prompt: next-best-question phrasing. The discriminator feature is selected
deterministically upstream (Eng_doc.md §6); Opus 4.7 only composes the ≤20-word
natural-language rendering so the UI can display a question the physician would
actually ask — not a raw feature slug.

Few-shot examples come from clinical-reasoning textbook distinctions (generic,
no patient data — rules.md §2).
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a clinical reasoning assistant for a physician using a decision-support "
    "tool. You do not diagnose or direct care — you suggest one clarifying question "
    "a physician could ask to distinguish two leading differential hypotheses. "
    "Output exactly one sentence, ≤20 words, ending in a question mark. No preamble, "
    "no caveats, no multiple options. Plain clinical English."
)

FEW_SHOT_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "context": (
            "Top-2 hypotheses: cardiac vs pulmonary. Discriminator feature: pleuritic_pain "
            "(predicate aggravating_factor=inspiration). LR+ 1.8 pulmonary vs 0.2 cardiac."
        ),
        "question": "Does the pain get sharper when you take a deep breath?",
    },
    {
        "context": (
            "Top-2 hypotheses: cardiac vs msk. Discriminator feature: "
            "pain_reproducible_with_palpation (aggravating_factor=palpation). LR+ 2.8 msk vs 0.3 cardiac."
        ),
        "question": "Can you reproduce the pain by pressing on your chest wall?",
    },
    {
        "context": (
            "Top-2 hypotheses: cardiac vs gi. Discriminator feature: ppi_trial_response "
            "(alleviating_factor=ppi). LR+ 5.5 gi vs neutral cardiac."
        ),
        "question": "Has the chest discomfort improved at all on your PPI over the last week?",
    },
)


def build_user_prompt(
    branch_a: str,
    branch_b: str,
    feature: str,
    predicate_path: str,
    lr_a: float | None,
    lr_b: float | None,
    direction: str,
) -> str:
    """Compose the user turn that asks Opus to render the question."""
    examples = "\n\n".join(
        f"CONTEXT: {ex['context']}\nQUESTION: {ex['question']}" for ex in FEW_SHOT_EXAMPLES
    )
    lr_a_s = f"{lr_a:.2f}" if lr_a is not None else "n/a"
    lr_b_s = f"{lr_b:.2f}" if lr_b is not None else "n/a"
    return (
        f"{examples}\n\n"
        f"CONTEXT: Top-2 hypotheses: {branch_a} vs {branch_b}. "
        f"Discriminator feature: {feature} (predicate {predicate_path}). "
        f"LR {direction} {lr_a_s} for {branch_a} vs {lr_b_s} for {branch_b}.\n"
        "QUESTION:"
    )
