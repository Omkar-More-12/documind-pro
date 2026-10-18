from app.ingestion.loader import chunk_pages


def test_chunking_keeps_page_numbers():
    pages = [{"page": 3, "text": "a" * 2000}]
    chunks = chunk_pages(pages, size=800, overlap=100)
    assert all(c["page"] == 3 for c in chunks)
    assert len(chunks) > 1