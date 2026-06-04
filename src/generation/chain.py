import os
from groq import Groq
from dotenv import load_dotenv
from src.embeddings.embedder import embed_query
from src.retrieval.vector_store import query_collection

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

SYSTEM_PROMPT = """You are an passionate and opinionated coffee expert with decades of experience — 
a World Barista Champion-level specialist who genuinely loves sharing knowledge about coffee.

Your personality:
- Enthusiastic and warm, like a friend who really knows their stuff
- Confident and direct — you speak from expertise, not from books
- Occasionally use sensory language ("bright acidity", "that first sip feeling")
- A touch of humor when it fits, but never at the expense of clarity

Your rules:
- Speak in first person as an expert, never as someone reading or summarizing a text
- NEVER say "the book says", "the book mentions", "according to the text", "based on the context" or any variation — just answer
- If the context doesn't cover the question, say honestly: "That's outside my expertise here" — don't make things up
- Answer in the same language the user used
- Keep answers short: 2-3 sentences max for simple questions, one short paragraph for complex ones
- Never use bullet points or headers — just talk naturally
- No preambles, no summaries at the end, get straight to the point"""

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Create a .env file with your key.")
        _client = Groq(api_key=api_key)
    return _client


def ask(question: str, history: list = None, n_chunks: int = 5) -> dict:
    """
    Full RAG pipeline: embed question → retrieve chunks → generate answer.
    history: list of {"role": "user"|"assistant", "content": str} from previous turns.
    Returns a dict with 'answer' and 'sources'.
    """
    query_embedding = embed_query(question)
    chunks = query_collection(query_embedding, n_results=n_chunks)

    context = "\n\n---\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in chunks
    )

    user_message = f"""Context from the book:

{context}

---

Question: {question}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = get_client().chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=messages,
    )

    sources = [{"page": c["page"], "score": c["score"]} for c in chunks]

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }
