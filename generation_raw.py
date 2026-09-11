import os
import sys
from pathlib import Path
import chromadb
import ollama
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

CHUNKS_PATH = Path(__file__).parent / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "personal_rag_assistant"
EMBED_MODEL = "bge-m3"
GEN_MODEL = "qwen/qwen3.8-27b"  # hosted on Groq, free tier, no local download
TOP_K = 5

env_file = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=env_file)

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url = "https://api.groq.com/openai/v1")


def embed_query(query: str):
    response = ollama.embed(model=EMBED_MODEL, input=[query])
    return response['embeddings'][0]

def retrieve_relevant_chunks(query: str, top_k: int = TOP_K):
    query_embedding = embed_query(query)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={'hnsw:space' : 'cosine'})
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return [
        {"text": doc, "source": meta["source"], "category": meta["category"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

def number_unique_sources(relevant_chunks: list[dict]):
    """Assign one citation number to all chunks from the same source file."""
    source_numbers = {}
    numbered_chunks = []

    for chunk in relevant_chunks:
        source_key = (chunk["category"], chunk["source"])
        if source_key not in source_numbers:
            source_numbers[source_key] = len(source_numbers) + 1

        numbered_chunks.append({
            **chunk,
            "source_number": source_numbers[source_key],
        })

    return numbered_chunks, source_numbers

def build_prompt(query: str, relevant_chunks: list[dict]):
    context = "\n\n".join(
        f"Source [{chunk['source_number']}]: {chunk['source']} "
        f"(Category: {chunk['category']})\n{chunk['text']}"
        for chunk in relevant_chunks
    )
    
    prompt = (
        f"Answer the question using ONLY the context below. If the context doesn't \
            contain the answer, say you don't have enough information -- do not use outside knowledge.\
            Cite the source number(s) you used, like [1] or [1][3], right after each claim. "
        f"Context passages with the same source number come from the same source; reuse that number.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    
    return prompt


def generate_answer(query: str, model: str = GEN_MODEL):
    relevant_chunks = retrieve_relevant_chunks(query)
    if not relevant_chunks:
        return "No relevant chunks found for the query."

    relevant_chunks, source_numbers = number_unique_sources(relevant_chunks)
    prompt = build_prompt(query, relevant_chunks)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    
    answer = response.choices[0].message.content.strip()
    sources = "\n".join(
        f"  [{number}] {category}/{source}"
        for (category, source), number in source_numbers.items()
    )
    return f"{answer}\n\nSources:\n{sources}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generation_raw.py 'Your question here'")
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    print(f"Q: {user_query}\n")
    answer = generate_answer(user_query)
    print(answer)
