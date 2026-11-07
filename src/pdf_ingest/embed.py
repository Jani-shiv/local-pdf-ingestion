"""Step 3: turn chunks into vectors with a local Ollama server."""
from __future__ import annotations

import requests

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"


class EmbeddingError(RuntimeError):
    pass


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_URL,
    batch_size: int = 16,
    timeout: int = 120,
) -> list[list[float]]:
    """Embed texts in small batches so large PDFs do not become one huge request."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
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
