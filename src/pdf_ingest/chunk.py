"""Step 2: split page text into overlapping character chunks."""
from __future__ import annotations

from typing import TypedDict

from .extract import Page


class Chunk(TypedDict):
    id: str
    page: int
    chunk: int
    text: str


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or more and smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break  # the last window already reached the end; avoid a tail that is pure overlap
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    result: list[Chunk] = []
    for page in pages:
        for index, piece in enumerate(split_text(page["text"], chunk_size, overlap)):
            result.append(
                {
                    "id": f"p{page['page']}-c{index}",
                    "page": page["page"],
                    "chunk": index,
                    "text": piece,
                }
            )
    return result
