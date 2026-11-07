"""Step 1: pull selectable text out of a PDF, page by page."""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pymupdf  # PyMuPDF (older releases expose it as `fitz`)


class Page(TypedDict):
    page: int
    text: str


def extract_pages(pdf_path: str | Path) -> tuple[list[Page], int]:
    """Return (pages_with_text, total_page_count).

    Pages with no selectable text are skipped, but the total count is returned
    so callers can warn when a PDF looks scanned (many pages, little text).
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[Page] = []
    with pymupdf.open(path) as document:
        total = document.page_count
        for number, page in enumerate(document, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page": number, "text": text})
    return pages, total
