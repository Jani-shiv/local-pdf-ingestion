"""Step 4: save chunks, vectors and the settings that produced them."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .chunk import Chunk


def save_dataset(
    path: str | Path,
    source: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    *,
    model: str,
    chunk_size: int,
    overlap: int,
) -> Path:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    records = [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]
    payload = {
        "meta": {
            "source": source,
            "model": model,
            "dimensions": len(vectors[0]) if vectors else 0,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunk_count": len(records),
        },
        "chunks": records,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def load_dataset(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
