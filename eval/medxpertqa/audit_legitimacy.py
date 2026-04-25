"""Legitimacy audit: Sections 3, 4, 5b (contamination deep-dive)."""
import anthropic
import json
import math
import os
import re

# Load official data
official = []
with open("eval/data/medxpertqa_official/input/medxpertqa_text_input.jsonl", encoding="utf-8") as f:
    for line in f:
        official.append(json.loads(line))

gold_map = {}
type_map = {}
opts_map = {}
for item in official:
    idx = int(item["id"].split("-")[1])
    gold_map[idx] = item["label"][0] if isinstance(item["label"], list) else item["label"]
    type_map[idx] = item.get("question_type", "?")
    opts_map[idx] = len(item["options"])


def official_extract(raw, n_opts=10):
    prediction = raw
    trigger = "Therefore, among A through J, the answer is"
    try:
        prediction = prediction.split(trigger)[1].strip()
    except IndexError:
        try:
            prediction = prediction.split(" answer is ")[1].strip()
        except IndexError:
            pass
    for phrase in ["I understand", "A through J", "A through E", "A through D"]:
        prediction = prediction.replace(phrase, "")
    options = [chr(65 + i) for i in range(n_opts)]
    options_str = r"\b(" + "|".join(options) + r")\b"
    found = re.findall(options_str, prediction)
    return found[0] if found else ""


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def our_extract(text):
    cleaned = _THINK_RE.sub("", text).strip().upper()
    for p in [
        r"(?:FINAL\s+ANSWER)\s*(?:IS|:|=|-)*\s*\(?([A-J])\)?",
        r"(?:^|\n)\s*\*{0,2}ANSWER\*{0,2}\s*(?:IS|:|=|-)\s*\(?([A-J])\)?",
        r"(?:THE\s+ANSWER\s+IS|ANSWER\s*:|ANSWER\s+IS)\s+\(?([A-J])\)?",
        r"^\s*\(?([A-J])\)?\s*\.?\s*$",
        r"\b([A-J])\b",
    ]:
        matches = list(re.finditer(p, cleaned, re.MULTILINE))
        if matches:
            return matches[-1].group(1)
    for c in reversed(cleaned):
        if c in "ABCDEFGHIJ":
            return c
    return ""


client = anthropic.Anthropic()

# Collect raw predictions
baseline_raw = {}
rag_raw = {}

print("Loading baseline predictions...", flush=True)
for result in client.messages.batches.results("msgbatch_017DAmLJMWgLapTPGGtw2kB4"):
    idx = int(result.custom_id.split("-")[1])
    if result.result.type == "succeeded":
        text = ""
        for block in result.result.message.content:
            if getattr(block, "type", "") == "text":
                text = block.text
                break
        baseline_raw[idx] = text
    else:
        baseline_raw[idx] = ""

print("Loading RAG predictions...", flush=True)
for result in client.messages.batches.results("msgbatch_01Ku7RC9o8nwS1Lp4hAuyno3"):
    idx = int(result.custom_id.split("-")[1])
    if result.result.type == "succeeded":
        text = ""
        for block in result.result.message.content:
            if getattr(block, "type", "") == "text":
                text = block.text
                break
        rag_raw[idx] = text
    else:
        rag_raw[idx] = ""

print(f"Baseline predictions: {len(baseline_raw)}", flush=True)
print(f"RAG predictions: {len(rag_raw)}", flush=True)
print(f"Baseline empty: {sum(1 for v in baseline_raw.values() if not v)}", flush=True)
print(f"RAG empty: {sum(1 for v in rag_raw.values() if not v)}", flush=True)

# === SECTION 3 ===
print("\n===== SECTION 3: RAW PREDICTION RESCORING =====", flush=True)

for variant, raw_map in [("baseline", baseline_raw), ("rag", rag_raw)]:
    for method_name, extractor in [("our_extractor", our_extract), ("official_extractor", lambda t, idx=0: official_extract(t, opts_map.get(idx, 10)))]:
        r_c, r_t, u_c, u_t, total_c, empty = 0, 0, 0, 0, 0, 0
        for idx in range(2450):
            text = raw_map.get(idx, "")
            gold = gold_map[idx]
            qtype = type_map[idx]
            if method_name == "official_extractor":
                pred = official_extract(text, opts_map.get(idx, 10))
            else:
                pred = our_extract(text)
            ok = pred.upper() == gold.upper() if pred else False
            if ok:
                total_c += 1
            if not pred:
                empty += 1
            if qtype == "Reasoning":
                r_t += 1
                if ok: r_c += 1
            else:
                u_t += 1
                if ok: u_c += 1
        rp = r_c / r_t * 100 if r_t else 0
        up = u_c / u_t * 100 if u_t else 0
        op = total_c / 2450 * 100
        print(f"  {variant:8s} | {method_name:20s} | R={r_c:4d}/{r_t:4d} ({rp:5.1f}%) U={u_c:3d}/{u_t:3d} ({up:5.1f}%) Overall={total_c:4d}/2450 ({op:5.1f}%) empty={empty}", flush=True)

# Extraction disagreements
for variant, raw_map in [("baseline", baseline_raw), ("rag", rag_raw)]:
    disagree = 0
    flips = 0
    for idx in range(2450):
        text = raw_map.get(idx, "")
        gold = gold_map[idx]
        op = our_extract(text)
        tp = official_extract(text, opts_map.get(idx, 10))
        if op != tp.upper():
            disagree += 1
        o_ok = op == gold
        t_ok = tp.upper() == gold.upper() if tp else False
        if o_ok != t_ok:
            flips += 1
    print(f"  {variant:8s} | extraction disagreements: {disagree}, correctness flips: {flips}", flush=True)

diff_pp = abs(1354 / 2450 * 100 - 1348 / 2450 * 100)
print(f"\n  Extraction difference severity: {'MINOR' if diff_pp <= 0.5 else 'MATERIAL'} ({diff_pp:.2f}pp)", flush=True)

# === SECTION 4 ===
print("\n===== SECTION 4: PAIRED DELTA =====", flush=True)

both_correct = 0
both_wrong = 0
rag_wins = 0
base_wins = 0

for idx in range(2450):
    b_text = baseline_raw.get(idx, "")
    r_text = rag_raw.get(idx, "")
    gold = gold_map[idx]
    b_pred = official_extract(b_text, opts_map.get(idx, 10))
    r_pred = official_extract(r_text, opts_map.get(idx, 10))
    b_ok = b_pred.upper() == gold.upper() if b_pred else False
    r_ok = r_pred.upper() == gold.upper() if r_pred else False
    if b_ok and r_ok:
        both_correct += 1
    elif not b_ok and not r_ok:
        both_wrong += 1
    elif r_ok and not b_ok:
        rag_wins += 1
    elif b_ok and not r_ok:
        base_wins += 1

net = rag_wins - base_wins
n_disc = rag_wins + base_wins
if n_disc > 0:
    chi2 = (abs(rag_wins - base_wins) - 1) ** 2 / n_disc
    z = math.sqrt(chi2)
    p_approx = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
else:
    chi2 = 0
    p_approx = 1.0

baseline_acc = (both_correct + base_wins) / 2450 * 100
rag_acc = (both_correct + rag_wins) / 2450 * 100

print(json.dumps({
    "baseline_correct": both_correct + base_wins,
    "rag_correct": both_correct + rag_wins,
    "both_correct": both_correct,
    "both_wrong": both_wrong,
    "rag_wins": rag_wins,
    "baseline_wins": base_wins,
    "net_gain_cases": net,
    "delta_pp": round(rag_acc - baseline_acc, 2),
    "mcnemar_chi2": round(chi2, 3),
    "mcnemar_p_approx": f"{p_approx:.6f}",
    "n_discordant": n_disc,
    "statistically_significant_p005": p_approx < 0.05,
}, indent=2), flush=True)

# === SECTION 5b: Contamination deep-dive ===
print("\n===== SECTION 5b: CONTAMINATION ANALYSIS =====", flush=True)

with open("eval/data/rag_index/passages.json", encoding="utf-8") as f:
    passages = json.load(f)

contaminated_indices = set()
for idx in range(2450):
    q_prefix = official[idx]["question"][:60].lower()
    for p in passages:
        if q_prefix in p.lower():
            contaminated_indices.add(idx)
            break

print(f"Contaminated cases: {len(contaminated_indices)}/2450", flush=True)
clean_indices = set(range(2450)) - contaminated_indices

# Accuracy on contaminated vs clean
for subset_name, subset_idx in [("contaminated", contaminated_indices), ("clean", clean_indices)]:
    b_c = 0
    r_c = 0
    n = len(subset_idx)
    for idx in subset_idx:
        b_text = baseline_raw.get(idx, "")
        r_text = rag_raw.get(idx, "")
        gold = gold_map[idx]
        b_pred = official_extract(b_text, opts_map.get(idx, 10))
        r_pred = official_extract(r_text, opts_map.get(idx, 10))
        if b_pred.upper() == gold.upper():
            b_c += 1
        if r_pred.upper() == gold.upper():
            r_c += 1
    bp = b_c / n * 100 if n else 0
    rp = r_c / n * 100 if n else 0
    print(f"  {subset_name:13s}: n={n:4d}  baseline={b_c:4d} ({bp:5.1f}%)  rag={r_c:4d} ({rp:5.1f}%)  delta={rp-bp:+.1f}pp", flush=True)
