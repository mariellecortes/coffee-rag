import json
import faiss
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data"
INDEX_FILE = DB_PATH / "coffee.index"
META_FILE = DB_PATH / "coffee_meta.json"


def store_chunks(chunks: list[dict]) -> None:
    """
    Saves embeddings to a FAISS index and metadata to JSON.
    """
    embeddings = np.array([c["embedding"] for c in chunks], dtype="float32")
    dim = embeddings.shape[1]

    # cosine similarity via L2 on normalized vectors
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after normalization
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    metadata = [{"id": c["id"], "text": c["text"], "page": c["page"], "source": c["source"]} for c in chunks]
    META_FILE.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    print(f"  Stored {len(chunks)} chunks in FAISS index at '{DB_PATH}'")


def query_collection(embedding: list[float], n_results: int = 5) -> list[dict]:
    """
    Returns the top-n most similar chunks for a given query embedding.
    """
    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))

    query = np.array([embedding], dtype="float32")
    faiss.normalize_L2(query)

    scores, indices = index.search(query, n_results)

    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        hits.append({
            "text": meta["text"],
            "page": meta["page"],
            "source": meta["source"],
            "score": round(float(score), 4),
        })

    return hits
