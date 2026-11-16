"""Step 4: save chunks, vectors and the settings that produced them. Vector store abstraction.

This module provides:
- VectorStore: Abstract base class for vector storage backends
- FAISSStore: FAISS-based vector store (default, fast, file-based)
- ChromaStore: ChromaDB-based vector store (persistent, feature-rich)
- Dataset I/O: JSON serialization for chunks and embeddings
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .chunk import Chunk
from .config import settings


class VectorStore(Protocol):
    """Protocol defining the vector store interface.

    All vector store implementations must provide these methods.
    """

    def __init__(self, path: Path | str, dimension: int) -> None:
        """Initialize vector store.

        Args:
            path: Directory path for store persistence
            dimension: Embedding vector dimension
        """
        ...

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Add vectors and associated chunk metadata.

        Args:
            vectors: Embedding vectors of shape (n_chunks, dimension)
            chunks: List of Chunk metadata matching vectors
        """
        ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding of shape (dimension,)
            top_k: Number of results to return

        Returns:
            List of (score, metadata) tuples sorted by score descending
        """
        ...

    def save(self) -> None:
        """Persist store to disk."""
        ...

    @property
    def count(self) -> int:
        """Return number of vectors in store."""
        ...


class FAISSStore:
    """FAISS-based vector store using IndexFlatIP (inner product).

    Features:
    - Fast in-memory search with disk persistence
    - Cosine similarity via L2 normalization + inner product
    - Lightweight, no external dependencies beyond faiss-cpu

    Example:
        store = FAISSStore("./data/vs", dimension=768)
        store.add(vectors, chunks)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self, path: Path | str, dimension: int):
        """Initialize FAISS store.

        Args:
            path: Directory for index and metadata files
            dimension: Embedding vector dimension
        """
        self.path = Path(path)
        self.dimension = dimension
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.path / "faiss.index"
        self.meta_path = self.path / "metadata.pkl"
        self._index: Any = None
        self._metadata: list[dict] = []
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing index or create new one."""
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
        """Add vectors and chunks to the store.

        Args:
            vectors: Embedding vectors of shape (n_chunks, dimension)
            chunks: List of Chunk metadata matching vectors
        """
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
        """Search for similar vectors using cosine similarity.

        Args:
            query_vector: Query embedding of shape (dimension,)
            top_k: Number of results to return

        Returns:
            List of (score, metadata) tuples sorted by score descending
        """
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
        """Persist index and metadata to disk."""
        import faiss
        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self._metadata, f)

    @property
    def count(self) -> int:
        """Return number of vectors in store."""
        return self._index.ntotal if self._index else 0


class ChromaStore:
    """ChromaDB-based vector store with persistent client.

    Features:
    - Persistent storage with ChromaDB
    - Built-in HNSW indexing
    - Rich metadata filtering
    - HTTP API available separately

    Example:
        store = ChromaStore("./data/vs", dimension=768)
        store.add(vectors, chunks)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self, path: Path | str, dimension: int):
        """Initialize ChromaDB store.

        Args:
            path: Directory for ChromaDB persistence
            dimension: Embedding vector dimension
        """
        self.path = Path(path)
        self.dimension = dimension
        self.path.mkdir(parents=True, exist_ok=True)
        self.client: Any = None
        self.collection: Any = None
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing collection or create new one."""
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
        """Add vectors and chunks to the store.

        Args:
            vectors: Embedding vectors of shape (n_chunks, dimension)
            chunks: List of Chunk metadata matching vectors
        """
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
        """Search for similar vectors using cosine similarity.

        Args:
            query_vector: Query embedding of shape (dimension,)
            top_k: Number of results to return

        Returns:
            List of (score, metadata) tuples sorted by score descending
        """
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
        """ChromaDB auto-persists, no explicit save needed."""
        pass

    @property
    def count(self) -> int:
        """Return number of vectors in store."""
        return self.collection.count() if self.collection else 0


def get_vector_store(path: Path | str, dimension: int) -> FAISSStore | ChromaStore:
    """Get vector store instance based on configuration.

    Args:
        path: Store directory path
        dimension: Embedding vector dimension

    Returns:
        FAISSStore or ChromaStore instance based on settings.vector_store
    """
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
    """Save chunks and embeddings to JSON dataset file.

    Args:
        path: Output file path
        source: Source document identifier
        chunks: List of chunks with metadata
        vectors: List of embedding vectors matching chunks
        model: Embedding model name
        chunk_size: Chunk size used
        overlap: Overlap used

    Returns:
        Path to saved file

    Raises:
        ValueError: If chunks and vectors length mismatch
    """
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
    """Load dataset from JSON file.

    Args:
        path: Dataset file path

    Returns:
        Parsed dataset dictionary with meta and chunks
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_vector_store(
    chunks: list[Chunk],
    vectors: np.ndarray,
    store_path: Path | str,
) -> FAISSStore | ChromaStore:
    """Create and save vector store from chunks and vectors.

    Args:
        chunks: List of chunks with metadata
        vectors: Embedding vectors of shape (n_chunks, dimension)
        store_path: Directory for vector store

    Returns:
        Initialized and saved vector store instance
    """
    dimension = vectors.shape[1] if len(vectors) > 0 else 0
    store = get_vector_store(store_path, dimension)
    store.add(vectors, chunks)
    store.save()
    return store


def load_vector_store(store_path: Path | str, dimension: int) -> FAISSStore | ChromaStore:
    """Load existing vector store.

    Args:
        store_path: Store directory path
        dimension: Embedding vector dimension

    Returns:
        Loaded vector store instance
    """
    return get_vector_store(store_path, dimension)