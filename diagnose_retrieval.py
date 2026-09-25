"""
Retrieval diagnostics -- run this BEFORE re-running evaluate.py.

The Phase 2 run returned "I don't have enough information" for all 50
questions, and the judge reported that the retrieved chunks came from an
unrelated paper (LLM calibration / LoRA ensembles / NeurIPS checklist).
This script checks each link in the chain to find where it breaks:

  1. Are the eval PDFs actually in data/raw/?
  2. Did ingest.py produce clean chunks for them (or garbled text)?
  3. Does the Chroma collection contain those chunks, or stale/foreign ones?
  4. Does retrieval return the right source for each eval question?
     -> reported as Hit@k and MRR, computed WITHOUT any LLM judge
        (free, instant, no Groq rate limit).

Usage:
    python diagnose_retrieval.py
"""

import json
from collections import Counter
from pathlib import Path

import chromadb
import ollama

RAW_DIR = Path("data/corpus")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EVAL_SET_PATH = Path("data/eval/eval_dataset.jsonl")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "personal_rag_assistant"
EMBED_MODEL = "bge-m3"
TOP_K = 5


def header(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def garbage_ratio(text: str) -> float:
    """Share of characters that are not letters, digits, whitespace or common
    punctuation. Clean English prose is usually under 0.05; a PDF with a
    broken font encoding often comes out well above 0.3."""
    if not text:
        return 1.0
    ok = sum(c.isalnum() or c.isspace() or c in ".,;:()[]-'\"%/=+*<>?!" for c in text)
    return 1 - ok / len(text)


def expected_sources(item: dict) -> set[str]:
    return {s.strip() for s in item.get("source", "").split("+") if s.strip().endswith(".pdf")}


def main():
    eval_items = [json.loads(l) for l in EVAL_SET_PATH.open(encoding="utf-8") if l.strip()]
    eval_sources = set().union(*(expected_sources(it) for it in eval_items))

    # --- 1. Raw files ----------------------------------------------------
    header("1. Files in data/raw/")
    raw_files = {p.name: p for p in RAW_DIR.rglob("*") if p.is_file()}
    print(f"{len(raw_files)} files found.")
    for src in sorted(eval_sources):
        status = f"OK  ({raw_files[src].relative_to(RAW_DIR)})" if src in raw_files else "MISSING"
        print(f"  {src:40s} {status}")
    others = sorted(set(raw_files) - eval_sources)
    if others:
        print(f"\nOther files in the corpus ({len(others)}): {others[:10]}{' ...' if len(others) > 10 else ''}")

    # --- 2. Chunks produced by ingest.py ---------------------------------
    header("2. data/processed/chunks.jsonl")
    chunks = [json.loads(l) for l in CHUNKS_PATH.open(encoding="utf-8") if l.strip()]
    by_src: dict[str, list[str]] = {}
    for c in chunks:
        by_src.setdefault(c["source"], []).append(c["text"])
    print(f"{len(chunks)} chunks from {len(by_src)} sources.")
    print(f"\n  {'source':40s} {'chunks':>7s} {'garbage':>8s}")
    for src, texts in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
        g = sum(garbage_ratio(t) for t in texts) / len(texts)
        flag = "  <-- likely broken text extraction" if g > 0.2 else ""
        print(f"  {src:40s} {len(texts):7d} {g:8.2f}{flag}")
    for src in sorted(eval_sources - set(by_src)):
        print(f"  {src:40s} {0:7d}      --  <-- NOT INGESTED")

    # --- 3. Chroma collection --------------------------------------------
    header("3. Chroma collection")
    collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
    stored = collection.get(include=["metadatas", "embeddings"])
    n_stored = len(stored["ids"])
    print(f"{n_stored} vectors stored (chunks.jsonl has {len(chunks)}).")
    if n_stored != len(chunks):
        print("  <-- MISMATCH: the collection holds chunks that are not in the current "
              "chunks.jsonl (stale data from an earlier ingest), or is missing some.")
    stored_src = Counter(m["source"] for m in stored["metadatas"])
    print("\nVectors per source (top 10):")
    for src, n in stored_src.most_common(10):
        tag = "eval" if src in eval_sources else "other"
        print(f"  {src:40s} {n:6d}  [{tag}]")
    if len(stored["embeddings"]):
        print(f"\nStored embedding dimension: {len(stored['embeddings'][0])}")
    q_dim = len(ollama.embed(model=EMBED_MODEL, input="test")["embeddings"][0])
    print(f"Query embedding dimension ({EMBED_MODEL}): {q_dim}")

    # --- 4. Retrieval probe: Hit@k and MRR, no LLM judge -----------------
    header(f"4. Retrieval quality on answerable eval questions (top {TOP_K})")
    hits, rr, seen_top1 = 0, 0.0, Counter()
    answerable = [it for it in eval_items if expected_sources(it)]
    for it in answerable:
        q_emb = ollama.embed(model=EMBED_MODEL, input=it["question"])["embeddings"][0]
        res = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
        got = [m["source"] for m in res["metadatas"][0]]
        seen_top1[res["ids"][0][0]] += 1
        want = expected_sources(it)
        rank = next((i + 1 for i, s in enumerate(got) if s in want), None)
        hits += rank is not None
        rr += 1 / rank if rank else 0
        mark = f"hit@{rank}" if rank else "MISS"
        print(f"  [{mark:6s}] want {sorted(want)} got {got[:3]}")

    n = len(answerable)
    print(f"\nHit@{TOP_K}: {hits}/{n} = {hits / n:.2f}    MRR: {rr / n:.3f}")
    most_common_id, count = seen_top1.most_common(1)[0]
    if n >= 5 and count > n * 0.3:
        print(f"  <-- The same chunk ({most_common_id}) is the #1 result for {count}/{n} "
              "different questions. That points to an embedding problem, not just a "
              "missing document.")


if __name__ == "__main__":
    main()
