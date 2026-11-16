"""Recipe extractor: structured output from cookbooks/recipe PDFs.

This module provides:
- Recipe, Ingredient dataclasses with serialization
- RecipeExtractor: LLM-based extraction from text chunks

Example:
    extractor = RecipeExtractor(chat)
    recipes = extractor.extract(chunks)
    for r in recipes:
        print(r.to_markdown())
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdf_ingest.chat import RAGChat
    from pdf_ingest.chunk import Chunk


@dataclass
class Ingredient:
    """Recipe ingredient with quantity and notes.

    Attributes:
        name: Ingredient name (e.g., "all-purpose flour")
        amount: Quantity as string (e.g., "2", "1/2")
        unit: Unit of measure (e.g., "cups", "tbsp", "g")
        notes: Preparation notes (e.g., "sifted", "room temperature")
    """
    name: str
    amount: str | None = None
    unit: str | None = None
    notes: str | None = None


@dataclass
class Recipe:
    """Structured recipe with ingredients, instructions, and metadata.

    Attributes:
        title: Recipe name
        description: Brief description
        servings: Number of servings
        prep_time_minutes: Preparation time in minutes
        cook_time_minutes: Cooking time in minutes
        ingredients: List of Ingredient objects
        instructions: Step-by-step instructions
        nutrition: Per-serving nutrition info
        tags: Category tags (e.g., "dessert", "vegetarian")
        source_page: Source document page number
    """
    title: str
    description: str | None = None
    servings: int | None = None
    prep_time_minutes: int | None = None
    cook_time_minutes: int | None = None
    ingredients: list[Ingredient] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    nutrition: dict | None = None
    tags: list[str] = field(default_factory=list)
    source_page: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        """Convert to Markdown for display."""
        lines = [f"# {self.title}"]
        if self.description:
            lines.append(f"\n{self.description}\n")
        if self.servings:
            lines.append(f"**Servings:** {self.servings}")
        if self.prep_time_minutes:
            lines.append(f"**Prep Time:** {self.prep_time_minutes} min")
        if self.cook_time_minutes:
            lines.append(f"**Cook Time:** {self.cook_time_minutes} min")
        if self.tags:
            lines.append(f"**Tags:** {', '.join(self.tags)}")

        lines.append("\n## Ingredients")
        for ing in self.ingredients:
            parts = []
            if ing.amount:
                parts.append(ing.amount)
            if ing.unit:
                parts.append(ing.unit)
            parts.append(ing.name)
            if ing.notes:
                parts.append(f"({ing.notes})")
            lines.append(f"- {' '.join(parts)}")

        lines.append("\n## Instructions")
        for i, step in enumerate(self.instructions, 1):
            lines.append(f"{i}. {step}")

        if self.nutrition:
            lines.append("\n## Nutrition (per serving)")
            for k, v in self.nutrition.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)


class RecipeExtractor:
    """Extract structured recipes from text chunks using LLM.

    Uses RAGChat to extract recipe information with fallback to regex parsing.

    Example:
        extractor = RecipeExtractor(chat)
        recipes = extractor.extract(chunks)
    """

    def __init__(self, chat: RAGChat) -> None:
        """Initialize extractor.

        Args:
            chat: RAGChat instance for LLM calls
        """
        self.chat = chat

    def extract(self, chunks: list[Chunk]) -> list[Recipe]:
        """Extract recipes from list of chunks.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of extracted Recipe objects
        """
        recipes = []
        for chunk in chunks:
            recipe = self._extract_from_chunk(chunk)
            if recipe:
                recipes.append(recipe)
        return recipes

    def _extract_from_chunk(self, chunk: Chunk) -> Recipe | None:
        """Extract recipe from single chunk using LLM.

        Args:
            chunk: Text chunk to process

        Returns:
            Recipe object or None if extraction fails
        """
        prompt = f"""Extract recipe information from this text. Return ONLY valid JSON matching this schema:
{{
  "title": "string",
  "description": "string or null",
  "servings": "integer or null",
  "prep_time_minutes": "integer or null",
  "cook_time_minutes": "integer or null",
  "ingredients": [{{"name": "string", "amount": "string or null", "unit": "string or null", "notes": "string or null"}}],
  "instructions": ["string"],
  "nutrition": {{"calories": "string", "protein": "string", "carbs": "string", "fat": "string"}} or null,
  "tags": ["string"]
}}

Text (page {chunk['page']}, chunk {chunk['chunk']}):
{chunk['text'][:3000]}"""

        response = self.chat._call_ollama(prompt, stream=False)
        try:
            data = json.loads(response)
            data["source_page"] = chunk["page"]
            return Recipe(**data)
        except (json.JSONDecodeError, TypeError):
            return self._fallback_extract(chunk)

    def _fallback_extract(self, chunk: Chunk) -> Recipe | None:
        """Fallback regex-based extraction when LLM fails.

        Args:
            chunk: Text chunk to process

        Returns:
            Recipe object with best-effort extraction
        """
        text = chunk["text"]
        title_match = re.search(r"(?i)^(?:recipe\s*[:|-]\s*)?([A-Z][A-Za-z\s]+)", text)
        title = title_match.group(1).strip() if title_match else f"Recipe from page {chunk['page']}"

        ingredients = []
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r"^[\d\/\.\-\s]+(?:cup|tbsp|tsp|oz|lb|g|kg|ml|l|pinch|dash|clove)s?\b", line, re.IGNORECASE):
                ingredients.append(Ingredient(name=line))

        instructions = []
        in_instructions = False
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r"(?i)^(?:instructions?|directions?|method|steps?)\s*[:|-]?$", line):
                in_instructions = True
                continue
            if in_instructions and line:
                if re.match(r"^\d+[\.\)]\s+", line):
                    instructions.append(re.sub(r"^\d+[\.\)]\s+", "", line))
                elif len(line) > 20:
                    instructions.append(line)

        return Recipe(
            title=title,
            ingredients=ingredients[:20],
            instructions=instructions[:15],
            source_page=chunk["page"],
        )