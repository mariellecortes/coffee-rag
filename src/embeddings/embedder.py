from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading embedding model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Adds an 'embedding' field to each chunk dict.
    """
    model = get_model()
    texts = [c["text"] for c in chunks]

    print(f"  Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks


def embed_query(query: str) -> list[float]:
    model = get_model()
    return model.encode(query).tolist()
