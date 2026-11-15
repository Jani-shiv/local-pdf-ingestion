"""Step 3: turn chunks into vectors with Ollama or local sentence-transformers."""
from __future__ import annotations

import asyncio
import time
import httpx
import requests
import numpy as np
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from .config import settings


DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class RetryableError(EmbeddingError):
    """Error that should trigger a retry."""

    pass


class NonRetryableError(EmbeddingError):
    """Error that should not trigger a retry."""

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

    async def embed_async(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Async version of embed using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, texts, batch_size, show_progress)


def _is_retryable_error(exc: Exception) -> bool:
    """Determine if an error is retryable."""
    if isinstance(exc, (requests.ConnectionError, httpx.ConnectError)):
        return True
    if isinstance(exc, (requests.Timeout, httpx.TimeoutException)):
        return True
    if isinstance(exc, (requests.HTTPError, httpx.HTTPStatusError)):
        response = getattr(exc, "response", None)
        return response is not None and response.status_code >= 500
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RetryableError),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _embed_batch_with_retry(
    batch: list[str],
    model: str,
    base_url: str,
    timeout: int,
) -> list[list[float]]:
    """Embed a single batch with retry logic."""
    try:
        response = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": batch},
            timeout=timeout,
        )
    except requests.ConnectionError as exc:
        raise RetryableError(
            f"Could not reach Ollama at {base_url}. Is it running? (try: ollama serve)"
        ) from exc
    except requests.Timeout as exc:
        raise RetryableError(f"Request timed out after {timeout}s") from exc

    if response.status_code == 404:
        raise NonRetryableError(
            f"Model '{model}' not found. Run: ollama pull {model}"
        )
    if response.status_code >= 500:
        raise RetryableError(f"Server error: {response.status_code}")
    response.raise_for_status()

    batch_vectors = response.json()["embeddings"]
    if len(batch_vectors) != len(batch):
        raise NonRetryableError(
            f"Expected {len(batch)} embeddings, got {len(batch_vectors)}"
        )
    return batch_vectors


async def _embed_batch_async_with_retry(
    batch: list[str],
    model: str,
    base_url: str,
    timeout: int,
    client: httpx.AsyncClient,
) -> list[list[float]]:
    """Embed a single batch asynchronously with retry logic."""
    for attempt in range(3):
        try:
            response = await client.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": batch},
                timeout=timeout,
            )
        except httpx.ConnectError as exc:
            if attempt == 2:
                raise RetryableError(
                    f"Could not reach Ollama at {base_url}. Is it running? (try: ollama serve)"
                ) from exc
            await asyncio.sleep(2 ** attempt)
            continue
        except httpx.TimeoutException as exc:
            if attempt == 2:
                raise RetryableError(f"Request timed out after {timeout}s") from exc
            await asyncio.sleep(2 ** attempt)
            continue

        if response.status_code == 404:
            raise NonRetryableError(
                f"Model '{model}' not found. Run: ollama pull {model}"
            )
        if response.status_code >= 500:
            if attempt == 2:
                raise RetryableError(f"Server error: {response.status_code}")
            await asyncio.sleep(2 ** attempt)
            continue

        response.raise_for_status()

        batch_vectors = response.json()["embeddings"]
        if len(batch_vectors) != len(batch):
            raise NonRetryableError(
                f"Expected {len(batch)} embeddings, got {len(batch_vectors)}"
            )
        return batch_vectors

    raise RetryableError("Max retries exceeded")


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_URL,
    batch_size: int = 16,
    timeout: int = 120,
    use_local: bool | None = None,
    max_retries: int = 3,
) -> list[list[float]]:
    use_local = use_local if use_local is not None else settings.use_local_embeddings

    if use_local:
        embedder = LocalEmbedder()
        vectors = embedder.embed(texts, batch_size=batch_size)
        return vectors.tolist()

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", disable=len(texts) < batch_size):
        batch = texts[start : start + batch_size]
        try:
            batch_vectors = _embed_batch_with_retry(batch, model, base_url, timeout)
            vectors.extend(batch_vectors)
        except RetryableError as exc:
            log.error(f"Failed to embed batch after retries: {exc}")
            raise EmbeddingError(f"Embedding failed after {max_retries} retries: {exc}") from exc
        except NonRetryableError as exc:
            raise EmbeddingError(str(exc)) from exc
    return vectors


async def embed_texts_async(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_URL,
    batch_size: int = 16,
    timeout: int = 120,
    use_local: bool | None = None,
    max_concurrent: int = 4,
) -> list[list[float]]:
    """Asynchronously embed multiple texts with concurrent batch processing."""
    use_local = use_local if use_local is not None else settings.use_local_embeddings

    if use_local:
        embedder = LocalEmbedder()
        vectors = await embedder.embed_async(texts, batch_size=batch_size)
        return vectors.tolist()

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    async with httpx.AsyncClient(timeout=timeout) as client:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await _embed_batch_async_with_retry(batch, model, base_url, timeout, client)

        tasks = [
            embed_batch(texts[start : start + batch_size])
            for start in range(0, len(texts), batch_size)
        ]

        results = await asyncio.gather(*tasks)
        vectors: list[list[float]] = []
        for batch_vectors in results:
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


async def embed_query_async(
    query: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_URL,
    use_local: bool | None = None,
) -> np.ndarray:
    vectors = await embed_texts_async([query], model, base_url, batch_size=1, use_local=use_local)
    return np.array(vectors[0], dtype=np.float32)