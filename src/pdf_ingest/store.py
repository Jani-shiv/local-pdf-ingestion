"""Step 4: save chunks, vectors and the settings that produced them. Vector store abstraction."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .chunk import Chunk
from .config import settings


class VectorStore:
    def __init__(self, path: Path | str, dimension: int):
        self.path = Path(path)
        self.dimension = dimension
        self.path.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._metadata: list[dict] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError


class FAISSStore(VectorStore):
    def __init__(self, path: Path | str, dimension: int):
        super().__init__(path, dimension)
        self.index_path = self.path / "faiss.index"
        self.meta_path = self.path / "metadata.pkl"
        self._load_or_create()

    def _load_or_create(self) -> None:
        try:
            import faiss
            if self.index_path.exists() and self.meta_path.exists():
                self._index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "rb") as f:
                    self._metadata = pickle.load(f)
            else:
                self._index = faiss.IndexFlatIP(self.dimension)
                self._metadata = []
        except ImportError:
            raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        import faiss
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        for chunk in chunks:
            self._metadata.append({
                "id": chunk["id"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "text": chunk["text"],
            })

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        import faiss
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        scores, indices = self._index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                results.append((float(score), self._metadata[idx]))
        return results

    def save(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self._metadata, f)

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0


class ChromaStore(VectorStore):
    def __init__(self, path: Path | str, dimension: int):
        super().__init__(path, dimension)
        self.client = None
        self.collection = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.client = chromadb.PersistentClient(
                path=str(self.path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name="pdf_chunks",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [{
            "page": chunk["page"],
            "chunk": chunk["chunk"],
        } for chunk in chunks]
        embeddings = vectors.tolist()
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k
        )
        output = []
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                score = 1.0 - distance
                output.append((score, {
                    "id": doc_id,
                    "text": doc,
                    "page": metadata["page"],
                    "chunk": metadata["chunk"],
                }))
        return output

    def save(self) -> None:
        pass

    @property
    def count(self) -> int:
        return self.collection.count() if self.collection else 0


def get_vector_store(path: Path | str, dimension: int) -> VectorStore:
    if settings.vector_store == "chroma":
        return ChromaStore(path, dimension)
    return FAISSStore(path, dimension)


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


def save_vector_store(
    chunks: list[Chunk],
    vectors: np.ndarray,
    store_path: Path | str,
) -> VectorStore:
    dimension = vectors.shape[1] if len(vectors) > 0 else 0
    store = get_vector_store(store_path, dimension)
    store.add(vectors, chunks)
    store.save()
    return store


def load_vector_store(store_path: Path | str, dimension: int) -> VectorStore:
    return get_vector_store(store_path, dimension)