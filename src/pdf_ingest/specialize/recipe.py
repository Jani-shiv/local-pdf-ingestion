"""Recipe extractor: structured output from cookbooks/recipe PDFs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

from ..chat import RAGChat
from ..chunk import Chunk


@dataclass
class Ingredient:
    name: str
    amount: Optional[str] = None
    unit: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Recipe:
    title: str
    description: Optional[str] = None
    servings: Optional[int] = None
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    ingredients: list[Ingredient] = None
    instructions: list[str] = None
    nutrition: Optional[dict] = None
    tags: list[str] = None
    source_page: Optional[int] = None

    def __post_init__(self):
        if self.ingredients is None:
            self.ingredients = []
        if self.instructions is None:
            self.instructions = []
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
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
    def __init__(self, chat: RAGChat):
        self.chat = chat

    def extract(self, chunks: list[Chunk]) -> list[Recipe]:
        recipes = []
        for chunk in chunks:
            recipe = self._extract_from_chunk(chunk)
            if recipe:
                recipes.append(recipe)
        return recipes

    def _extract_from_chunk(self, chunk: Chunk) -> Optional[Recipe]:
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

    def _fallback_extract(self, chunk: Chunk) -> Optional[Recipe]:
        text = chunk["text"]
        title_match = re.search(r"(?i)^(?:recipe\s*[:|-]\s*)?([A-Z][A-Za-z\s]+)", text)
        title = title_match.group(1).strip() if title_match else f"Recipe from page {chunk['page']}"

        ingredients = []
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r"^[\d\/\.\-\s]+(?:cup|tbsp|tsp|oz|lb|g|kg|ml|l|pinch|dash|clove)s?\b", line, re.I):
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