"""Study guide generator: flashcards, mind maps, quizzes from educational content.

This module provides:
- StudyGuide, Flashcard, QuizQuestion, MindMapNode dataclasses
- StudyExtractor: LLM-based extraction from text chunks

Example:
    extractor = StudyExtractor(chat)
    guides = extractor.extract(chunks)
    for g in guides:
        print(g.to_anki_csv())
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdf_ingest.chat import RAGChat
    from pdf_ingest.chunk import Chunk


@dataclass
class Flashcard:
    """Study flashcard with question and answer.

    Attributes:
        question: Question text
        answer: Answer text
        topic: Subject topic
        difficulty: "easy", "medium", or "hard"
    """
    question: str
    answer: str
    topic: str | None = None
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class QuizQuestion:
    """Multiple choice quiz question.

    Attributes:
        question: Question text
        options: List of 4 answer options
        correct_index: Index of correct option (0-3)
        explanation: Explanation of correct answer
        topic: Subject topic
    """
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    topic: str | None = None


@dataclass
class MindMapNode:
    """Node in a hierarchical mind map.

    Attributes:
        topic: Node topic/label
        children: Child nodes
    """
    topic: str
    children: list[MindMapNode] = field(default_factory=list)


@dataclass
class StudyGuide:
    """Comprehensive study guide with multiple formats.

    Attributes:
        title: Guide title
        summary: Brief content summary
        topics: List of covered topics
        flashcards: List of Flashcard objects
        quiz_questions: List of QuizQuestion objects
        mind_map: Root MindMapNode for mind map
        key_concepts: Dictionary of concept definitions
        source_pages: Source document page numbers
    """
    title: str
    summary: str
    topics: list[str] = field(default_factory=list)
    flashcards: list[Flashcard] = field(default_factory=list)
    quiz_questions: list[QuizQuestion] = field(default_factory=list)
    mind_map: MindMapNode | None = None
    key_concepts: dict[str, str] = field(default_factory=dict)
    source_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.mind_map:
            data["mind_map"] = self._mind_map_to_dict(self.mind_map)
        return data

    def _mind_map_to_dict(self, node: MindMapNode) -> dict:
        """Recursively convert mind map to dict."""
        return {
            "topic": node.topic,
            "children": [self._mind_map_to_dict(c) for c in node.children],
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_anki_csv(self) -> str:
        """Convert flashcards to Anki-compatible CSV.

        Returns:
            CSV string with Front,Back,Tags columns
        """
        lines = ["Front,Back,Tags"]
        for fc in self.flashcards:
            tags = f"study;{fc.topic or 'general'};{fc.difficulty}"
            lines.append(f'"{fc.question}","{fc.answer}","{tags}"')
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Convert to Markdown for display."""
        lines = [f"# Study Guide: {self.title}"]
        lines.append(f"\n## Summary\n{self.summary}")

        if self.key_concepts:
            lines.append("\n## Key Concepts")
            for concept, definition in self.key_concepts.items():
                lines.append(f"### {concept}\n{definition}")

        if self.flashcards:
            lines.append("\n## Flashcards")
            for i, fc in enumerate(self.flashcards, 1):
                lines.append(f"\n**Q{i}:** {fc.question}")
                lines.append(f"**A{i}:** {fc.answer}")

        if self.quiz_questions:
            lines.append("\n## Practice Quiz")
            for i, q in enumerate(self.quiz_questions, 1):
                lines.append(f"\n**{i}.** {q.question}")
                for j, opt in enumerate(q.options):
                    marker = "✓" if j == q.correct_index else "○"
                    lines.append(f"  {marker} {opt}")
                lines.append(f"  *Explanation: {q.explanation}*")

        if self.mind_map:
            lines.append("\n## Mind Map")
            lines.append(self._mind_map_to_markdown(self.mind_map))

        return "\n".join(lines)

    def _mind_map_to_markdown(self, node: MindMapNode, level: int = 0) -> str:
        """Convert mind map to Markdown bullet list."""
        lines = ["  " * level + f"- {node.topic}"]
        for child in node.children:
            lines.append(self._mind_map_to_markdown(child, level + 1))
        return "\n".join(lines)


class StudyExtractor:
    """Extract structured study guides from text chunks using LLM.

    Uses RAGChat to generate flashcards, quizzes, mind maps, and summaries.

    Example:
        extractor = StudyExtractor(chat)
        guides = extractor.extract(chunks)
    """

    def __init__(self, chat: RAGChat) -> None:
        """Initialize extractor.

        Args:
            chat: RAGChat instance for LLM calls
        """
        self.chat = chat

    def extract(self, chunks: list[Chunk]) -> list[StudyGuide]:
        """Extract study guides from list of chunks.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of extracted StudyGuide objects
        """
        guides = []
        for chunk in chunks:
            guide = self._extract_from_chunk(chunk)
            if guide:
                guides.append(guide)
        return guides

    def _extract_from_chunk(self, chunk: Chunk) -> StudyGuide | None:
        """Extract study guide from single chunk using LLM.

        Args:
            chunk: Text chunk to process

        Returns:
            StudyGuide object or None if extraction fails
        """
        prompt = f"""Create a comprehensive study guide from this text. Return ONLY valid JSON matching this schema:
{{
  "title": "string",
  "summary": "string (3-5 sentences)",
  "topics": ["string"],
  "flashcards": [
    {{"question": "string", "answer": "string", "topic": "string or null", "difficulty": "easy|medium|hard"}}
  ],
  "quiz_questions": [
    {{"question": "string", "options": ["string", "string", "string", "string"], "correct_index": "integer 0-3", "explanation": "string", "topic": "string or null"}}
  ],
  "mind_map": {{"topic": "string", "children": [{{"topic": "string", "children": []}}]}} or null,
  "key_concepts": {{"concept": "definition"}}
}}

Text (page {chunk['page']}, chunk {chunk['chunk']}):
{chunk['text'][:4000]}"""

        response = self.chat._call_ollama(prompt, stream=False)
        try:
            data = json.loads(response)
            data["source_pages"] = [chunk["page"]]
            return StudyGuide(**data)
        except (json.JSONDecodeError, TypeError):
            return self._fallback_extract(chunk)

    def _fallback_extract(self, chunk: Chunk) -> StudyGuide | None:
        """Fallback extraction when LLM fails.

        Args:
            chunk: Text chunk to process

        Returns:
            StudyGuide with basic flashcards
        """
        text = chunk["text"]
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

        flashcards = []
        for i in range(0, min(len(sentences), 10), 2):
            if i + 1 < len(sentences):
                flashcards.append(Flashcard(
                    question=f"What is: {sentences[i][:80]}...?",
                    answer=sentences[i + 1],
                ))

        return StudyGuide(
            title=f"Study Notes - Page {chunk['page']}",
            summary=" ".join(sentences[:3]) + ".",
            flashcards=flashcards,
            source_pages=[chunk["page"]],
        )