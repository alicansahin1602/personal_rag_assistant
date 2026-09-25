import json
import math
import re
from collections import Counter
from pathlib import Path
import ollama
import chromadb

CHUNKS_PATH = Path(__file__).parent / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = 'chroma_db'
COLLECTION_NAME = 'personal_rag_assistant'
EMBEDDING_MODEL = 'bge-m3'
BATCH_SIZE = 32
MAX_GARBAGE_RATIO = 0.3   # chunks above this are skipped as unreadable (broken PDF text)
MIN_CHUNK_CHARS = 20      # chunks shorter than this after cleaning are skipped

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def load_chunks(path:Path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def make_id(chunk: dict) -> str:
    # chunk_id restarts at 0 for every file, so it is NOT unique on its own.
    # Using it alone made every file overwrite the previous file's chunks.
    return f"{chunk.get('category', 'unknown')}::{chunk.get('source', 'unknown')}::{chunk['chunk_id']}"


def clean(text: str) -> str:
    return CONTROL_CHARS.sub(" ", text).strip()


def garbage_ratio(text: str) -> float:
    """Share of characters that aren't letters, digits, whitespace or common
    punctuation. Clean prose is usually < 0.05; broken extractions are > 0.3."""
    if not text:
        return 1.0
    ok = sum(c.isalnum() or c.isspace() or c in ".,;:()[]-'\"%/=+*<>?!" for c in text)
    return 1 - ok / len(text)


def embed_batch(texts: list[str]):
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response['embeddings']


def store(collection, batch: list[dict], embeddings):
    if any(math.isnan(v) for e in embeddings for v in e):
        raise ValueError("embedding model returned NaN values")
    collection.add(
        ids=[make_id(chunk) for chunk in batch],
        embeddings=embeddings,
        metadatas=[{'source': chunk.get('source', 'unknown'),
                    'category': chunk.get('category', 'unknown'),
                    'chunk_id': chunk['chunk_id']} for chunk in batch],
        documents=[chunk['text'] for chunk in batch]
    )


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")

    raw_chunks = load_chunks(CHUNKS_PATH)

    if not raw_chunks:
        raise ValueError("No chunks found in the provided file.")

    # Guard against the id-collision bug ever coming back
    id_counts = Counter(make_id(chunk) for chunk in raw_chunks)
    duplicates = [i for i, n in id_counts.items() if n > 1]
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate chunk ids, e.g. {duplicates[:3]} -- "
                         "chunks would overwrite each other.")

    # Screen out unreadable chunks (judged on the ORIGINAL text: cleaning turns
    # control characters into spaces, which would hide how broken a chunk was)
    chunks, skipped = [], Counter()
    for chunk in raw_chunks:
        text = clean(chunk['text'])
        if len(text) < MIN_CHUNK_CHARS or garbage_ratio(chunk['text']) > MAX_GARBAGE_RATIO:
            skipped[chunk.get('source', 'unknown')] += 1
            continue
        chunks.append({**chunk, 'text': text})

    print(f"{len(raw_chunks)} chunks read, {len(chunks)} kept, {sum(skipped.values())} skipped.")
    for source, n in skipped.most_common():
        total = sum(1 for chunk in raw_chunks if chunk.get('source', 'unknown') == source)
        warn = "  <-- most of this file is unreadable; fix its parsing" if n > total / 2 else ""
        print(f"  skipped {n}/{total} chunks from {source}{warn}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Rebuild from scratch every run so the store always matches chunks.jsonl
    # (upsert alone never removes chunks from deleted or renamed files)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet
    collection = client.create_collection(name=COLLECTION_NAME, metadata={'hnsw:space' : 'cosine'})

    print(f"Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL}' and storing in Chroma...")

    failed = []  # (source, chunk_id, error)
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        try:
            store(collection, batch, embed_batch([chunk['text'] for chunk in batch]))
        except Exception:
            # one bad chunk shouldn't sink the whole batch: retry one by one
            for chunk in batch:
                try:
                    store(collection, [chunk], embed_batch([chunk['text']]))
                except Exception as e:
                    failed.append((chunk.get('source', 'unknown'), chunk['chunk_id'], repr(e)))
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks processed")

    # Completeness check
    stored = collection.count()
    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {stored}/{len(chunks)} chunks in {CHROMA_DIR}/")
    stored_sources = Counter(m['source'] for m in collection.get(include=['metadatas'])['metadatas'])
    for source, n in sorted(stored_sources.items()):
        print(f"  {source}: {n}")
    missing = sorted({chunk.get('source', 'unknown') for chunk in chunks} - set(stored_sources))
    if stored != len(chunks) or failed or missing:
        print("\nWARNING: the store is incomplete.")
        if missing:
            print(f"  Sources with nothing stored: {missing}")
        for source, chunk_id, err in failed[:20]:
            print(f"  Failed: {source} chunk {chunk_id}: {err}")
    else:
        print("OK: every kept chunk was embedded and stored.")

    ## Testing retrieval
    query_text = "How does asymmetric duo algorithm work?"
    query_embedding = embed_batch([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print(f"\nSanity check — top 3 results for: '{query_text}'")

    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{meta['category']}/{meta['source']}] (distance={dist:.3f}) {doc[:150]}...")


if __name__ == "__main__":
    main()