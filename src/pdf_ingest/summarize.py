"""Summarization: map-reduce summaries, key points, study guides.

This module provides:
- Summarizer: Map-reduce summarization with key point extraction
- Study guide generation (flashcards, mindmaps, quizzes)
- Direct and map-reduce strategies based on document length

Example:
    summarizer = create_summarizer(store, chat)
    summary = summarizer.summarize_all(chunks, max_length=500, style="bullet")
    print(summary.text)
    for kp in summary.key_points:
        print(f"  - {kp}")

    study = summarizer.generate_study_guide(chunks, format="flashcards")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .config import settings
from .chat import RAGChat, ChatResponse
from .chunk import Chunk
from .store import VectorStore


@dataclass
class Summary:
    """Summary result with key points and source tracking.

    Attributes:
        text: Summary text
        key_points: List of key point strings
        chunks_used: List of chunk IDs used in summary
    """
    text: str
    key_points: list[str]
    chunks_used: list[str]


class Summarizer:
    """Map-reduce summarizer with key point extraction.

    Uses two strategies:
    - Direct: For short texts (< 3000 chars), single-pass summarization
    - Map-reduce: For long texts, chunk summaries combined into final summary

    Example:
        summarizer = Summarizer(store, chat)
        summary = summarizer.summarize_all(chunks, max_length=500, style="concise")
    """

    def __init__(
        self,
        store: VectorStore,
        chat: RAGChat,
    ):
        """Initialize summarizer.

        Args:
            store: Vector store for retrieval
            chat: RAGChat instance for LLM calls
        """
        self.store = store
        self.chat = chat

    def summarize_all(
        self,
        chunks: list[Chunk],
        max_length: int = 500,
        style: Literal["concise", "detailed", "bullet"] = "concise",
    ) -> Summary:
        """Generate summary from chunks.

        Automatically selects direct or map-reduce based on total text length.

        Args:
            chunks: List of text chunks to summarize
            max_length: Maximum summary length in words (default: 500)
            style: Summary style - "concise", "detailed", "bullet" (default: "concise")

        Returns:
            Summary with text, key_points, and chunks_used
        """
        all_text = "\n\n".join(c["text"] for c in chunks)
        if len(all_text) < 3000:
            return self._summarize_direct(all_text, max_length, style)
        return self._summarize_map_reduce(chunks, max_length, style)

    def _summarize_direct(self, text: str, max_length: int, style: str) -> Summary:
        """Single-pass summarization for short texts.

        Args:
            text: Full text to summarize
            max_length: Maximum summary length in words
            style: Summary style

        Returns:
            Summary object
        """
        prompt = f"""Summarize the following text in a {style} manner (max {max_length} words).
Extract 5-7 key points as bullet points.

Text:
{text}

Format:
Summary: [your summary]

Key Points:
- [point 1]
- [point 2]
..."""
        response = self.chat._call_ollama(prompt, stream=False)
        return self._parse_summary(response, [])

    def _summarize_map_reduce(self, chunks: list[Chunk], max_length: int, style: str) -> Summary:
        """Map-reduce summarization for long texts.

        1. Map: Summarize each chunk individually
        2. Reduce: Combine chunk summaries into final summary

        Args:
            chunks: List of chunks to summarize
            max_length: Maximum summary length in words
            style: Summary style

        Returns:
            Summary object
        """
        chunk_summaries = []
        for chunk in chunks:
            prompt = f"Summarize this chunk in 2-3 sentences:\n\n{chunk['text']}"
            summary = self.chat._call_ollama(prompt, stream=False)
            chunk_summaries.append(f"[{chunk['id']}] {summary}")

        combined = "\n\n".join(chunk_summaries)
        prompt = f"""Combine these chunk summaries into a {style} overall summary (max {max_length} words).
Extract 7-10 key points covering the main topics.

Chunk Summaries:
{combined}

Format:
Summary: [your summary]

Key Points:
- [point 1]
- [point 2]
..."""
        response = self.chat._call_ollama(prompt, stream=False)
        return self._parse_summary(response, [c["id"] for c in chunks])

    def _parse_summary(self, text: str, chunk_ids: list[str]) -> Summary:
        """Parse LLM response into Summary object.

        Expected format:
        Summary: [summary text]
        Key Points:
        - [point 1]
        - [point 2]

        Args:
            text: Raw LLM response
            chunk_ids: List of chunk IDs used

        Returns:
            Parsed Summary object
        """
        summary_text = ""
        key_points = []
        in_summary = False
        in_points = False

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Summary:"):
                in_summary = True
                in_points = False
                summary_text = line[8:].strip()
            elif line.startswith("Key Points:"):
                in_summary = False
                in_points = True
            elif in_summary and line:
                summary_text += " " + line
            elif in_points and line.startswith("-"):
                key_points.append(line[1:].strip())
            elif in_points and line and not line.startswith("-"):
                if key_points:
                    key_points[-1] += " " + line

        if not summary_text:
            summary_text = text[:500]

        return Summary(
            text=summary_text.strip(),
            key_points=key_points[:10],
            chunks_used=chunk_ids,
        )

    def generate_study_guide(
        self,
        chunks: list[Chunk],
        format: Literal["flashcards", "mindmap", "quiz", "all"] = "flashcards",
    ) -> dict:
        """Generate study materials from chunks.

        Args:
            chunks: List of text chunks
            format: Output format - "flashcards", "mindmap", "quiz", "all"

        Returns:
            Dictionary with study materials in requested format

        Raises:
            ValueError: If format is unknown
        """
        all_text = "\n\n".join(c["text"] for c in chunks)
        text_excerpt = all_text[:8000]

        if format == "flashcards":
            prompt = (
                "Create 10-15 flashcards from this text. Each flashcard should have a question and answer.\n"
                'Format as JSON array: [{"question": "...", "answer": "..."}]\n\n'
                f"Text:\n{text_excerpt}"
            )
        elif format == "mindmap":
            prompt = (
                'Create a hierarchical mind map from this text. Output as nested JSON:\n'
                '{"topic": "main", "children": [{"topic": "subtopic", "children": [...]}]}\n\n'
                f"Text:\n{text_excerpt}"
            )
        elif format == "quiz":
            prompt = (
                'Create 10 multiple choice questions from this text. Format as JSON:\n'
                '[{"question": "...", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": "..."}]\n\n'
                f"Text:\n{text_excerpt}"
            )
        else:
            raise ValueError(f"Unknown format: {format}")

        response = self.chat._call_ollama(prompt, stream=False)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response", "raw": response}


def create_summarizer(store: VectorStore, chat: RAGChat) -> Summarizer:
    """Factory function to create Summarizer.

    Args:
        store: Vector store instance
        chat: RAGChat instance

    Returns:
        Configured Summarizer instance
    """
    return Summarizer(store, chat)