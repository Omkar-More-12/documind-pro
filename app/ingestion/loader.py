import fitz  # PyMuPDF


def load_pdf(path):
    doc = fitz.open(path)
    return [{"page": i + 1, "text": p.get_text()} for i, p in enumerate(doc)]


def chunk_pages(pages, size=800, overlap=150):
    chunks = []
    for p in pages:
        text, start = p["text"], 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"text": piece, "page": p["page"]})
            start += size - overlap
    return chunks