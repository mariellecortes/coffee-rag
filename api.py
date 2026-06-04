import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.generation.chain import ask
from src.embeddings.embedder import get_model

app = FastAPI(title="RAG Coffee API")


@app.on_event("startup")
def preload_model():
    get_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask_question(request: dict):
    question = request.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = request.get("history", [])
    result = ask(question, history=history)

    pages = sorted({s["page"] for s in result["sources"]})
    pages_str = ", ".join(str(p) for p in pages)
    source = f"HOFFMANN, James. The World Atlas of Coffee. London: Mitchell Beazley, 2014. p. {pages_str}."

    return JSONResponse({"answer": result["answer"], "source": source})
