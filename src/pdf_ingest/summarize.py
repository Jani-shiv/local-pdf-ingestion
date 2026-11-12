"""Summarization: map-reduce summaries, key points, study guides."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .config import settings
from .chat import RAGChat, ChatResponse
from .chunk import Chunk
from .store import VectorStore


@dataclass
class Summary:
    text: str
    key_points: list[str]
    chunks_used: list[str]


class Summarizer:
    def __init__(
        self,
        store: VectorStore,
        chat: RAGChat,
    ):
        self.store = store
        self.chat = chat

    def summarize_all(
        self,
        chunks: list[Chunk],
        max_length: int = 500,
        style: str = "concise",
    ) -> Summary:
        all_text = "\n\n".join(c["text"] for c in chunks)
        if len(all_text) < 3000:
            return self._summarize_direct(all_text, max_length, style)
        return self._summarize_map_reduce(chunks, max_length, style)

    def _summarize_direct(self, text: str, max_length: int, style: str) -> Summary:
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
        format: str = "flashcards",
    ) -> dict:
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
    return Summarizer(store, chat)