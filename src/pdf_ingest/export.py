"""Export functionality for knowledge base (JSON, CSV, Markdown, Anki).

This module provides export functions for:
- Knowledge base (JSON, CSV, Markdown)
- Study guides (Anki CSV)
- Domain objects (recipes, invoices, etc.)

Example:
    export_knowledge_base_json("./data/vs", 768, "./export/kb.json")
    export_study_guide_anki(guide, "./export/flashcards.csv")
"""
from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .chunk import Chunk
from .store import VectorStore, load_vector_store
from .specialize import StudyGuide


def export_knowledge_base_json(
    vector_store_path: str | Path,
    dimension: int,
    output_path: str | Path,
) -> Path:
    """Export knowledge base to JSON format.

    Args:
        vector_store_path: Path to vector store directory
        dimension: Embedding vector dimension
        output_path: Output file path

    Returns:
        Path to exported file
    """
    store = load_vector_store(vector_store_path, dimension)
    all_results = store.search(np.zeros(dimension, dtype=np.float32), store.count)

    data = {
        "meta": {
            "chunks": len(all_results),
            "dimension": dimension,
            "vector_store_path": str(vector_store_path),
        },
        "chunks": [
            {
                "id": meta["id"],
                "text": meta["text"],
                "page": meta["page"],
                "chunk": meta["chunk"],
            }
            for _, meta in all_results
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def export_knowledge_base_markdown(
    vector_store_path: str | Path,
    dimension: int,
    output_path: str | Path,
) -> Path:
    """Export knowledge base to Markdown format.

    Args:
        vector_store_path: Path to vector store directory
        dimension: Embedding vector dimension
        output_path: Output file path

    Returns:
        Path to exported file
    """
    store = load_vector_store(vector_store_path, dimension)
    all_results = store.search(np.zeros(dimension, dtype=np.float32), store.count)

    md = "# Knowledge Base Export\n\n"
    for _, meta in all_results:
        md += f"## [{meta['id']}] Page {meta['page']}\n{meta['text']}\n\n---\n\n"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    return output_path


def export_knowledge_base_csv(
    vector_store_path: str | Path,
    dimension: int,
    output_path: str | Path,
) -> Path:
    """Export knowledge base to CSV format.

    Args:
        vector_store_path: Path to vector store directory
        dimension: Embedding vector dimension
        output_path: Output file path

    Returns:
        Path to exported file
    """
    store = load_vector_store(vector_store_path, dimension)
    all_results = store.search(np.zeros(dimension, dtype=np.float32), store.count)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "page", "chunk", "text"])
        for _, meta in all_results:
            writer.writerow([meta["id"], meta["page"], meta["chunk"], meta["text"]])

    return output_path


def export_knowledge_base(
    vector_store_path: str | Path,
    dimension: int,
    output_path: str | Path,
    format: Literal["json", "markdown", "csv"] = "json",
) -> Path:
    """Export knowledge base in specified format.

    Args:
        vector_store_path: Path to vector store directory
        dimension: Embedding vector dimension
        output_path: Output file path
        format: Export format - "json", "markdown", or "csv"

    Returns:
        Path to exported file

    Raises:
        ValueError: If format is unsupported
    """
    if format == "json":
        return export_knowledge_base_json(vector_store_path, dimension, output_path)
    elif format == "markdown":
        return export_knowledge_base_markdown(vector_store_path, dimension, output_path)
    elif format == "csv":
        return export_knowledge_base_csv(vector_store_path, dimension, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_study_guide_anki(
    study_guide: StudyGuide,
    output_path: str | Path,
) -> Path:
    """Export study guide as Anki-compatible CSV.

    Args:
        study_guide: StudyGuide object to export
        output_path: Output file path

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_data = study_guide.to_anki_csv()
    output_path.write_text(csv_data, encoding="utf-8")
    return output_path


def export_domain_objects(
    objects: list[Any],
    output_path: str | Path,
    format: Literal["json", "markdown", "csv"] = "json",
) -> Path:
    """Export domain objects (recipes, invoices, etc.) to various formats.

    Args:
        objects: List of domain objects with to_dict(), to_markdown(), to_csv_row()
        output_path: Output file path
        format: Export format - "json", "markdown", or "csv"

    Returns:
        Path to exported file

    Raises:
        ValueError: If format is unsupported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        output_path.write_text(
            json.dumps([obj.to_dict() for obj in objects], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    elif format == "markdown":
        output_path.write_text(
            "\n\n---\n\n".join(obj.to_markdown() for obj in objects),
            encoding="utf-8",
        )
    elif format == "csv" and objects and hasattr(objects[0], "to_csv_row"):
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header from first object
            writer.writerow(objects[0].csv_headers())
            for obj in objects:
                writer.writerow(obj.to_csv_row())
    else:
        raise ValueError(f"Unsupported format: {format}")

    return output_path