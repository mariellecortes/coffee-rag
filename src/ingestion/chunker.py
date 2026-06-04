def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """
    Splits page texts into overlapping chunks of ~chunk_size characters.
    Keeps source metadata (page number, source file) on each chunk.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "page": page["page"],
                    "source": page["source"],
                })
                chunk_id += 1

            # move forward by chunk_size minus overlap
            start += chunk_size - overlap

    print(f"  Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks
