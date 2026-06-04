"""
RAG Coffee — interactive Q&A CLI.

Usage:
    python app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.generation.chain import ask


def main():
    print("\n=== RAG Coffee ===")
    print("Ask anything about coffee (based on The World Atlas of Coffee).")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        print("\nSearching...\n")
        result = ask(question)

        print(f"Claude: {result['answer']}")

        pages = sorted({s["page"] for s in result["sources"]})
        pages_str = ", ".join(str(p) for p in pages)
        print(f"\n  Source: HOFFMANN, James. The World Atlas of Coffee. London: Mitchell Beazley, 2014. p. {pages_str}.\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
