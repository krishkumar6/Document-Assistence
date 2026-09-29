from pathlib import Path

from pypdf import PdfReader


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return (1-based page number, text) for every page that has extractable text.

    Whitespace is collapsed so chunk sizes measure content, not layout. Scanned
    PDFs with no text layer come back empty; they would need OCR first.
    """
    reader = PdfReader(path)
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if text:
            pages.append((page_no, text))
    return pages
