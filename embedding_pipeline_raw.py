import json
from pathlib import Path
import ollama
import chromadb

CHUNKS_PATH = Path(__file__).parent / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = 'chroma_db'
COLLECTION_NAME = 'personal_rag_assistant'
EMBEDDING_MODEL = 'bge-m3'
BATCH_SIZE = 32


def load_chunks(path:Path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def embed_batch(texts: list[str]):
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response['embeddings']

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")

    chunks = load_chunks(CHUNKS_PATH)

    if not chunks:
        raise ValueError("No chunks found in the provided file.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={'hnsw:space' : 'cosine'})

    print(f"Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL}' and storing in Chroma...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [chunk['text'] for chunk in batch]
        embeddings = embed_batch(texts)

        ids = [str(chunk['chunk_id']) for chunk in batch]
        metadatas = [{'source': chunk.get('source', 'unknown'), 'category': chunk.get('category', 'unknown')} for chunk in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks stored")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks in {CHROMA_DIR}/")

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
        print(f"  [{meta['category']}/{meta['source']}] (distance={dist:.3f}) {doc}...")


if __name__ == "__main__":
    main()