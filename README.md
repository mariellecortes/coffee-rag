# rag-coffee

A RAG system that answers questions about coffee using *The World Atlas of Coffee* by James Hoffmann as its knowledge base.

---

## how it works

```
question -> embedding -> FAISS similarity search -> top-k chunks -> Llama 3.3 70B -> answer
```

1. The PDF is parsed and split into overlapping text chunks
2. Each chunk is embedded with `sentence-transformers` and stored in a FAISS index
3. At query time, the question is embedded and matched against the index
4. The most relevant excerpts are sent to Llama 3.3 70B (via Groq) as context
5. The model answers grounded strictly in the retrieved text

---

## stack

| layer | tool |
|---|---|
| PDF parsing | PyMuPDF |
| chunking | custom with overlap |
| embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| vector store | FAISS (persisted locally) |
| LLM | Llama 3.3 70B via Groq API |
| API | FastAPI + Uvicorn |

---

## project structure

```
rag-coffee/
|-- src/
|   |-- ingestion/
|   |   |-- pdf_loader.py       # extract text per page
|   |   |-- chunker.py          # split into overlapping chunks
|   |-- embeddings/
|   |   |-- embedder.py         # generate + query embeddings
|   |-- retrieval/
|   |   |-- vector_store.py     # FAISS store + similarity search
|   |-- generation/
|       |-- chain.py            # RAG pipeline + Groq call
|-- data/
|   |-- coffee.index            # FAISS index (pre-built)
|   |-- coffee_meta.json        # chunk metadata
|-- scripts/
|   |-- ingest.py               # run once to index the PDF
|-- api.py                      # FastAPI endpoint
|-- app.py                      # CLI interface
|-- render.yaml                 # Render deploy config
```

---

## setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=gsk_...
```

### ingest (run once)

Place the PDF in `assets/` and run:

```bash
python scripts/ingest.py
```

This generates `data/coffee.index` and `data/coffee_meta.json`.

### run the API

```bash
uvicorn api:app --reload
```

### run the CLI

```bash
python app.py
```

---

## API

**`POST /ask`**

```json
{
  "question": "What makes Ethiopian coffee unique?",
  "history": []
}
```

```json
{
  "answer": "Ethiopian coffee is...",
  "source": "HOFFMANN, James. The World Atlas of Coffee. London: Mitchell Beazley, 2014. p. 42, 87."
}
```

---

## deploy

Configured for [Render](https://render.com) via `render.yaml`. Set `GROQ_API_KEY` as an environment variable in the Render dashboard.
