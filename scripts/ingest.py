"""
Ingestion pipeline — run once to index the PDF into ChromaDB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --pdf assets/other_book.pdf
"""
import sys
import argparse
import time
from pathlib import Path

# make src importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.embeddings.embedder import embed_chunks
from src.retrieval.vector_store import store_chunks

DEFAULT_PDF = str(
    Path(__file__).resolve().parents[1]
    / "assets"
    / "pdfcoffee.com_the-world-atlas-of-coffee-pdf-free.pdf"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Index a PDF into ChromaDB for RAG.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks in characters")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    print("\n=== RAG Coffee — Ingestion Pipeline ===\n")

    print("[1/4] Loading PDF...")
    pages = load_pdf(args.pdf)

    print("\n[2/4] Chunking text...")
    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)

    print("\n[3/4] Generating embeddings...")
    chunks = embed_chunks(chunks)

    print("\n[4/4] Storing in FAISS...")
    store_chunks(chunks)

    elapsed = time.time() - start
    print(f"\nDone! {len(chunks)} chunks indexed in {elapsed:.1f}s")
    print("You can now run: python app.py")


if __name__ == "__main__":
    main()
