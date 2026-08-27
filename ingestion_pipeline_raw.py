import json
from pathlib import Path
from dataclasses import dataclass, asdict

from pypdf import PdfReader
from bs4 import BeautifulSoup
from tqdm import tqdm


RAW_DIR = Path('data/corpus')

## We are using jsonl because we do not have to read and process entire 
## json object everytime we want to read a single document. Instead, we can read it line by line.
OUT_PATH = Path('data/processed/chunks.jsonl')

## Characters per chunk. Token based split will be considered
CHUNK_SIZE = 800

## Overlap between chunks. Token based split will be considered
## Reason for this is increasing the overlap will help in retaining context between chunks, which is important for downstream tasks like information retrieval or question answering.
CHUNK_OVERLAP = 200

@dataclass
class Chunk:
    """A class to represent a chunk of text."""
    source: str
    category: str
    chunk_id: int
    text: str


def get_category(path: Path, raw_dir: Path = RAW_DIR) -> str:
    """Derive a category label from the file's immediate subfolder under data/raw.
 
    data/raw/statistics/book.pdf       -> "statistics"
    data/raw/machine-learning/notes.md -> "machine-learning"
    data/raw/some_file.pdf             -> "uncategorized" (no subfolder)
    """
    relative_parts = path.relative_to(raw_dir).parts
    return relative_parts[0] if len(relative_parts) > 1 else "uncategorized"

def extract_text(path:Path):
    """Extract raw text from a single source file based on its extension."""
    suffix = path.suffix.lower()

    if suffix == '.pdf':
        reader = PdfReader(str(path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    elif suffix in ('.md', '.txt'):
        return path.read_text(encoding='utf-8')
    
    elif suffix in ('.html', '.htm'):
        html_content = path.read_text(encoding='utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')

        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator='\n', strip=True)

    raise ValueError(f"Unsupported file type: {path.name}")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple recursive-ish character chunking with overlap.

    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        return []
 
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step forward, leaving overlap
    return chunks

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_files = [
        p for p in RAW_DIR.rglob("*")
        if p.suffix.lower() in (".pdf", ".md", ".txt", ".html", ".htm")
    ]
 
    if not source_files:
        print(f"No source files found in {RAW_DIR}/ — add some PDFs/notes/blog posts first.")
        return
 
    all_chunks: list[Chunk] = []
    for path in tqdm(source_files, desc= f"Ingesting {len(source_files)} files"):
        try:
            text = extract_text(path)
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
            continue
 
        category = get_category(path)
        for i, piece in enumerate(chunk_text(text)):
            all_chunks.append(Chunk(source=path.name, category=category, chunk_id=i, text=piece))
 
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
 
    counts_by_category: dict[str, int] = {}
    for c in all_chunks:
        counts_by_category[c.category] = counts_by_category.get(c.category, 0) + 1
 
    print(f"\nDone. {len(all_chunks)} chunks from {len(source_files)} files -> {OUT_PATH}")
    print("By category:")
    for category, count in sorted(counts_by_category.items()):
        print(f"  {category}: {count} chunks")
 
 
if __name__ == "__main__":
    main()
 