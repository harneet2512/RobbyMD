"""RAG index for medical knowledge retrieval using BM25.

Builds a BM25 index from MedQA training Q+A pairs. No dense embeddings,
no GPU, no model download. Builds in seconds. Medical text is keyword-rich
so BM25 performs well at 10K-scale retrieval.

No MedXpertQA data is used -- no leakage risk.

Usage:
    python -m eval.medxpertqa.rag_index --build
    python -m eval.medxpertqa.rag_index --test "chest pain differential"
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO / "eval" / "data" / "rag_index"

_bm25: object | None = None
_passages: list[str] | None = None


def build_index(index_dir: Path | None = None) -> int:
    """Build BM25 index from MedQA training data."""
    from rank_bm25 import BM25Okapi
    from datasets import load_dataset

    out = index_dir or INDEX_DIR
    out.mkdir(parents=True, exist_ok=True)

    print("Loading MedQA training split...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    print(f"  {len(ds)} training cases")

    passages = []
    for item in ds:  # type: ignore[union-attr]
        q = item["question"] if "question" in item else ""
        opts = item["options"] if "options" in item else {}
        answer_key = item.get("answer_idx", "") if hasattr(item, "get") else (item["answer_idx"] if "answer_idx" in item else "")

        answer_text = ""
        if isinstance(opts, dict) and answer_key in opts:
            answer_text = opts[answer_key]
        elif isinstance(opts, list) and isinstance(answer_key, int):
            answer_text = opts[answer_key] if answer_key < len(opts) else ""

        if q and answer_text:
            passages.append(f"Q: {q}\nA: {answer_text}")

    print(f"  {len(passages)} passages with Q+A")

    print("Building BM25 index...")
    tokenized = [doc.lower().split() for doc in passages]
    bm25 = BM25Okapi(tokenized)

    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(out / "passages.json", "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)

    print(f"BM25 index built: {len(passages)} passages")
    print(f"Saved to: {out}")
    return len(passages)


def load_index(index_dir: Path | None = None) -> bool:
    """Load BM25 index and passages from disk."""
    global _bm25, _passages
    d = index_dir or INDEX_DIR
    bp = d / "bm25.pkl"
    pp = d / "passages.json"

    if not bp.exists() or not pp.exists():
        return False

    with open(bp, "rb") as f:
        _bm25 = pickle.load(f)
    with open(pp, encoding="utf-8") as f:
        _passages = json.load(f)
    return True


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Retrieve top-k relevant medical passages for a query."""
    global _bm25, _passages
    if _bm25 is None or _passages is None:
        if not load_index():
            return []

    tokenized_query = query.lower().split()
    results = _bm25.get_top_n(tokenized_query, _passages, n=top_k)  # type: ignore[union-attr]
    return results


def format_rag_context(passages: list[str]) -> str:
    """Format retrieved passages for prompt injection."""
    if not passages:
        return "(No relevant medical knowledge retrieved)"
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"[Passage {i}]: {p}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_index()
    elif "--test" in sys.argv:
        idx = sys.argv.index("--test")
        query = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "chest pain"
        if not load_index():
            print("Index not found. Run --build first.")
            sys.exit(1)
        results = retrieve(query, top_k=3)
        print(f"Query: {query}")
        print(f"Results: {len(results)}")
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r[:200]}...")
