"""Real-time cost and token tracker for the LongMemEval final run.

Reads diagnostics and computes token usage + cost from actual data.
Run alongside the final runner to track spend.
"""
import json
import sys
from pathlib import Path

# OpenRouter pricing (per 1M tokens)
PRICING = {
    "openai/gpt-5-mini":          {"input": 1.10, "output": 4.40},
    "openai/gpt-4o-2024-11-20":   {"input": 2.50, "output": 10.00},
    "all-MiniLM-L6-v2":           {"input": 0.00, "output": 0.00},  # local, free
}

# Estimated tokens per call (from actual measurements)
# Reader: ~17,800 input (evidence) + ~400 output (notes + answer)
# Judge: ~300 input (prompt) + ~5 output (yes/no)
READER_OUTPUT_EST = 400
JUDGE_INPUT_EST = 300
JUDGE_OUTPUT_EST = 5


def analyze(diag_path: str = "eval/longmemeval/results/final_full_500_diagnostics.json"):
    path = Path(diag_path)
    if not path.exists():
        print("No diagnostics file found.")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    total = len(data)
    if total == 0:
        print("No cases yet.")
        return

    # Accumulate actual tokens
    total_reader_input = sum(d["reader_input_tokens"] for d in data)
    total_reader_output = total * READER_OUTPUT_EST
    total_judge_input = total * JUDGE_INPUT_EST
    total_judge_output = total * JUDGE_OUTPUT_EST
    total_elapsed = sum(d["elapsed_s"] for d in data)

    # Cost calculation
    reader_cost_in = total_reader_input / 1e6 * PRICING["openai/gpt-5-mini"]["input"]
    reader_cost_out = total_reader_output / 1e6 * PRICING["openai/gpt-5-mini"]["output"]
    judge_cost_in = total_judge_input / 1e6 * PRICING["openai/gpt-4o-2024-11-20"]["input"]
    judge_cost_out = total_judge_output / 1e6 * PRICING["openai/gpt-4o-2024-11-20"]["output"]
    total_cost = reader_cost_in + reader_cost_out + judge_cost_in + judge_cost_out

    # Project full 500
    projected_total = total_cost * 500 / total if total > 0 else 0

    # Per-type token breakdown
    type_tokens: dict[str, list[int]] = {}
    for d in data:
        type_tokens.setdefault(d["question_type"], []).append(d["reader_input_tokens"])

    correct = sum(1 for d in data if d["correct_final"])

    print(f"{'=' * 60}")
    print(f"  LongMemEval-S Cost & Token Report")
    print(f"{'=' * 60}")
    print()
    print(f"  Progress: {total}/500 ({100*total/500:.0f}%)")
    print(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()
    print(f"  TOKEN USAGE (actual reader input + estimated output/judge)")
    print(f"  {'-' * 50}")
    print(f"  Reader input tokens:  {total_reader_input:>12,}")
    print(f"  Reader output tokens: {total_reader_output:>12,} (est)")
    print(f"  Judge input tokens:   {total_judge_input:>12,} (est)")
    print(f"  Judge output tokens:  {total_judge_output:>12,} (est)")
    print(f"  Embedding tokens:     {'free (local)':>12}")
    print(f"  Total API tokens:     {total_reader_input + total_reader_output + total_judge_input + total_judge_output:>12,}")
    print()
    print(f"  COST BREAKDOWN")
    print(f"  {'-' * 50}")
    print(f"  Reader (gpt-5-mini):")
    print(f"    Input:  {total_reader_input/1e6:.2f}M × $1.10/M = ${reader_cost_in:.2f}")
    print(f"    Output: {total_reader_output/1e6:.3f}M × $4.40/M = ${reader_cost_out:.2f}")
    print(f"  Judge (gpt-4o):")
    print(f"    Input:  {total_judge_input/1e6:.3f}M × $2.50/M = ${judge_cost_in:.2f}")
    print(f"    Output: {total_judge_output/1e6:.4f}M × $10.00/M = ${judge_cost_out:.2f}")
    print(f"  Embeddings (local): $0.00")
    print(f"  {'-' * 50}")
    print(f"  Spent so far:     ${total_cost:.2f}")
    print(f"  Projected total:  ${projected_total:.2f}")
    print()
    print(f"  PER-TYPE TOKEN USAGE")
    print(f"  {'-' * 50}")
    for qt in sorted(type_tokens):
        tokens = type_tokens[qt]
        avg_t = sum(tokens) / len(tokens)
        print(f"  {qt}: {len(tokens)} cases, avg {avg_t:.0f} tokens/case, total {sum(tokens):,}")
    print()
    print(f"  EFFICIENCY")
    print(f"  {'-' * 50}")
    print(f"  Avg tokens/question: {total_reader_input/total:,.0f}")
    print(f"  Avg cost/question:   ${total_cost/total:.4f}")
    print(f"  Avg time/question:   {total_elapsed/total:.1f}s")
    print(f"  Cost per correct:    ${total_cost/max(1,correct):.4f}")

    # Save cost log
    cost_log = {
        "timestamp": str(Path(diag_path).stat().st_mtime),
        "cases_completed": total,
        "cases_correct": correct,
        "accuracy": round(correct / total, 4),
        "tokens": {
            "reader_input": total_reader_input,
            "reader_output_est": total_reader_output,
            "judge_input_est": total_judge_input,
            "judge_output_est": total_judge_output,
            "total_api": total_reader_input + total_reader_output + total_judge_input + total_judge_output,
            "embeddings": "local_free",
        },
        "cost_usd": {
            "reader_input": round(reader_cost_in, 4),
            "reader_output": round(reader_cost_out, 4),
            "judge_input": round(judge_cost_in, 4),
            "judge_output": round(judge_cost_out, 4),
            "embeddings": 0.0,
            "total_spent": round(total_cost, 4),
            "projected_500": round(projected_total, 4),
        },
        "pricing": PRICING,
        "per_type_avg_tokens": {qt: round(sum(t)/len(t)) for qt, t in sorted(type_tokens.items())},
        "avg_cost_per_question": round(total_cost / total, 4),
        "avg_time_per_question_s": round(total_elapsed / total, 1),
    }
    log_path = Path("eval/longmemeval/results/final_full_500_cost_log.json")
    log_path.write_text(json.dumps(cost_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Cost log saved: {log_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/longmemeval/results/final_full_500_diagnostics.json"
    analyze(path)
