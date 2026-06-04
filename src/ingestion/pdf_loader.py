import fitz  # PyMuPDF
from pathlib import Path


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from each page of the PDF.
    Returns a list of dicts with 'page' and 'text' keys.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = text.strip()

        if text:  # skip blank pages
            pages.append({
                "page": page_num + 1,
                "text": text,
                "source": path.name,
            })

    doc.close()
    print(f"  Extracted {len(pages)} pages with content from '{path.name}'")
    return pages
