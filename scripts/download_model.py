from sentence_transformers import SentenceTransformer

print("Downloading and caching embedding model...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model cached successfully.")
