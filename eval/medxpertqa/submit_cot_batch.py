"""Submit official CoT prompt batch for MedXpertQA Text."""
import anthropic
import ast
import json
import time
from datasets import load_dataset

ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")
print(f"Dataset: {len(ds)} cases", flush=True)

requests = []
for idx in range(len(ds)):
    case = ds[idx]
    question = case["question"]
    opts_raw = case["options"]
    if isinstance(opts_raw, str):
        opts = ast.literal_eval(opts_raw)
    else:
        opts = opts_raw
    n_opts = len(opts)

    cot_prompt = "Q: " + question + "\nA: Let's think step by step."
    end_letter = chr(64 + n_opts)

    requests.append({
        "custom_id": f"case-{idx}",
        "params": {
            "model": "claude-opus-4-7",
            "max_tokens": 4096,
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful medical assistant.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": cot_prompt},
            ],
        },
    })

print(f"Submitting official CoT batch: {len(requests)} requests", flush=True)
client = anthropic.Anthropic()
batch = client.messages.batches.create(requests=requests)
print(f"CoT Batch ID: {batch.id}", flush=True)
print(f"Status: {batch.processing_status}", flush=True)

with open(
    "eval/reports/medxpertqa/3stage/batch_cot_full_meta.json", "w", encoding="utf-8"
) as f:
    json.dump(
        {"batch_id": batch.id, "type": "cot_official", "n": len(requests), "ts": time.time()},
        f,
    )
print("Submitted.", flush=True)
