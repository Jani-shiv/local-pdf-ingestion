"""Step 2: split page text into overlapping chunks (character, semantic, recursive).

This module provides text chunking functionality with three strategies:
- character: Fixed-size character chunks with overlap
- semantic: Sentence-aware chunking using embeddings (falls back to recursive)
- recursive: Hierarchical splitting by separators (paragraphs, sentences, words)
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, TypedDict, overload

from .config import settings

if TYPE_CHECKING:
    from .extract import Page


class Chunk(TypedDict):
    """A text chunk with metadata.

    Attributes:
        id: Unique identifier in format "p{page}-c{chunk}"
        page: Page number (1-indexed)
        chunk: Chunk index within page (0-indexed)
        text: The chunk text content
    """
    id: str
    page: int
    chunk: int
    text: str


@overload
def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    strategy: Literal["character", "semantic", "recursive"] = "semantic",
) -> list[str]: ...


def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    strategy: Literal["character", "semantic", "recursive"] = "semantic",
) -> list[str]:
    """Split text into overlapping chunks using specified strategy.

    Args:
        text: Input text to chunk
        chunk_size: Maximum chunk size in characters (default: 1000)
        overlap: Overlap between chunks in characters (default: 150)
        strategy: Chunking strategy - "character", "semantic", or "recursive"

    Returns:
        List of text chunks

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or more and smaller than chunk_size")

    if strategy == "character":
        return _split_character(text, chunk_size, overlap)
    if strategy == "semantic":
        return _split_semantic(text, chunk_size, overlap)
    if strategy == "recursive":
        return _split_recursive(text, chunk_size, overlap)
    raise ValueError(f"Unknown strategy: {strategy}")


def _split_character(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text by fixed character count with overlap.

    Args:
        text: Input text
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks

    Returns:
        List of character-based chunks
    """
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def _split_semantic(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text using semantic (embedding-based) sentence boundaries.

    Falls back to recursive splitting if sentence-transformers unavailable.

    Args:
        text: Input text
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks

    Returns:
        List of semantically-aware chunks
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.local_embed_model)
    except Exception:
        return _split_recursive(text, chunk_size, overlap)

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    embeddings = model.encode(sentences, show_progress_bar=False)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for i, sent in enumerate(sentences):
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            overlap_sentences = _get_overlap_sentences(
                current_chunk, overlap, model,
                embeddings[i - len(current_chunk):i] if i >= len(current_chunk) else [],
            )
            current_chunk = [*overlap_sentences, sent]
            current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(sent)
            current_len += sent_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return [c for c in chunks if c]


def _split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text recursively by hierarchical separators.

    Separators (in order): paragraphs, lines, sentences, clauses, words, characters

    Args:
        text: Input text
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks

    Returns:
        List of recursively split chunks
    """
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""]
    return _recursive_split(text, separators, chunk_size, overlap)


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Recursive splitting helper.

    Args:
        text: Input text
        separators: List of separators to try in order
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks

    Returns:
        List of chunks
    """
    if not text.strip():
        return []

    for sep in separators:
        if sep == "":
            return _split_character(text, chunk_size, overlap)
        parts = text.split(sep)
        if len(parts) > 1:
            chunks: list[str] = []
            current: list[str] = []
            current_len = 0

            for part in parts:
                part_with_sep = part + sep if part != parts[-1] else part
                part_len = len(part_with_sep)

                if current_len + part_len > chunk_size and current:
                    chunk_text = "".join(current).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    overlap_text = _get_overlap_text(chunk_text, overlap)
                    current = [overlap_text, part_with_sep] if overlap_text else [part_with_sep]
                    current_len = sum(len(p) for p in current)
                else:
                    current.append(part_with_sep)
                    current_len += part_len

            if current:
                chunk_text = "".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            return chunks

    return _split_character(text, chunk_size, overlap)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _get_overlap_sentences(
    sentences: list[str],
    overlap: int,
    model: Any,
    embeddings: Any,
) -> list[str]:
    """Get overlap sentences for semantic chunking.

    Args:
        sentences: List of sentences
        overlap: Overlap size in characters
        model: SentenceTransformer model
        embeddings: Pre-computed embeddings

    Returns:
        List of overlap sentences
    """
    if not sentences or overlap <= 0:
        return []
    total_len = sum(len(s) for s in sentences)
    if total_len <= overlap:
        return sentences
    overlap_text = " ".join(sentences)
    return _recursive_split(overlap_text, [" ", ""], overlap, 0)


def _get_overlap_text(text: str, overlap: int) -> str:
    """Get trailing overlap text.

    Args:
        text: Input text
        overlap: Overlap size in characters

    Returns:
        Overlap text (last `overlap` characters)
    """
    if overlap <= 0 or len(text) <= overlap:
        return ""
    return text[-overlap:]


def chunk_pages(
    pages: list[Page],
    chunk_size: int | None = None,
    overlap: int | None = None,
    strategy: Literal["character", "semantic", "recursive"] | None = None,
) -> list[Chunk]:
    """Chunk multiple pages into overlapping text chunks.

    Args:
        pages: List of pages with text and page numbers
        chunk_size: Maximum chunk size (default: settings.chunk_size)
        overlap: Overlap between chunks (default: settings.chunk_overlap)
        strategy: Chunking strategy (default: settings.chunk_strategy)

    Returns:
        List of Chunk objects with id, page, chunk index, and text
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    strategy = strategy or settings.chunk_strategy

    result: list[Chunk] = []
    for page in pages:
        pieces = split_text(page["text"], chunk_size, overlap, strategy)
        for index, piece in enumerate(pieces):
            result.append(
                {
                    "id": f"p{page['page']}-c{index}",
                    "page": page["page"],
                    "chunk": index,
                    "text": piece,
                },
            )
    return result