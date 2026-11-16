"""Step 1: pull selectable text out of a PDF, page by page with metadata extraction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pymupdf  # PyMuPDF (older releases expose it as `fitz`)


class Page(TypedDict):
    page: int
    text: str


@dataclass
class DocumentMetadata:
    """Metadata extracted from a PDF document."""
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: datetime | None = None
    modification_date: datetime | None = None
    page_count: int = 0
    file_size: int = 0
    is_encrypted: bool = False
    pdf_version: str | None = None


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


def extract_pages_with_metadata(pdf_path: str | Path) -> tuple[list[Page], int, DocumentMetadata]:
    """Extract pages with text and document metadata.

    Returns:
        Tuple of (pages_with_text, total_page_count, document_metadata)
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[Page] = []
    metadata = DocumentMetadata()

    with pymupdf.open(path) as document:
        metadata.page_count = document.page_count
        metadata.file_size = path.stat().st_size
        metadata.is_encrypted = document.is_encrypted
        metadata.pdf_version = document.pdf_version()

        # Extract document info dictionary
        info = document.metadata
        if info:
            metadata.title = info.get("title") or None
            metadata.author = info.get("author") or None
            metadata.subject = info.get("subject") or None
            metadata.creator = info.get("creator") or None
            metadata.producer = info.get("producer") or None

            # Parse dates
            for date_field, attr in [("creationDate", "creation_date"), ("modDate", "modification_date")]:
                date_str = info.get(date_str := date_field)
                if date_str:
                    try:
                        # PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
                        date_str = date_str.removeprefix("D:")
                        # Parse basic format
                        dt = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
                        setattr(metadata, attr, dt)
                    except (ValueError, TypeError):
                        pass

        for number, page in enumerate(document, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page": number, "text": text})

    return pages, metadata.page_count, metadata
