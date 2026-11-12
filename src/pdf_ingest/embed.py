"""Step 3: turn chunks into vectors with Ollama or local sentence-transformers."""
from __future__ import annotations

import requests
import numpy as np
from tqdm import tqdm

from .config import settings


DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"


class EmbeddingError(RuntimeError):
    pass


class LocalEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.local_embed_model
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_URL,
    batch_size: int = 16,
    timeout: int = 120,
    use_local: bool | None = None,
) -> list[list[float]]:
    use_local = use_local if use_local is not None else settings.use_local_embeddings

    if use_local:
        embedder = LocalEmbedder()
        vectors = embedder.embed(texts, batch_size=batch_size)
        return vectors.tolist()

    return _embed_with_ollama(texts, model, base_url, batch_size, timeout)


def _embed_with_ollama(
    texts: list[str],
    model: str,
    base_url: str,
    batch_size: int,
    timeout: int,
) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", disable=len(texts) < batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": batch},
                timeout=timeout,
            )
        except requests.ConnectionError as exc:
            raise EmbeddingError(
                f"Could not reach Ollama at {base_url}. Is it running? (try: ollama serve)"
            ) from exc

        if response.status_code == 404:
            raise EmbeddingError(
                f"Model '{model}' not found. Run: ollama pull {model}"
            )
        response.raise_for_status()

        batch_vectors = response.json()["embeddings"]
        if len(batch_vectors) != len(batch):
            raise EmbeddingError(
                f"Expected {len(batch)} embeddings, got {len(batch_vectors)}"
            )
        vectors.extend(batch_vectors)
    return vectors


def embed_query(
    query: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_URL,
    use_local: bool | None = None,
) -> np.ndarray:
    vectors = embed_texts([query], model, base_url, batch_size=1, use_local=use_local)
    return np.array(vectors[0], dtype=np.float32)