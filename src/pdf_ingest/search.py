"""Hybrid search: vector similarity + keyword (BM25) search.

This module provides hybrid search combining:
- Vector similarity search (cosine similarity via FAISS/Chroma)
- Keyword search (BM25 via rank_bm25)

Features:
- Configurable hybrid fusion
- Async and sync interfaces
- Threshold filtering
- Source attribution (vector/keyword)

Example:
    searcher = HybridSearcher("./data/vs", dimension=768)
    results = searcher.search("machine learning", top_k=5)
    for r in results:
        print(f"[{r.id}] page {r.page} score={r.score:.3f} ({r.source})")
        print(r.text[:200])
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .embed import embed_query, embed_query_async
from .store import load_vector_store

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SearchResult:
    """A single search result with metadata.

    Attributes:
        id: Unique chunk identifier
        text: Chunk text content
        page: Page number (1-indexed)
        chunk: Chunk index within page
        score: Relevance score (0-1 for vector, BM25 score for keyword)
        source: "vector" or "keyword" indicating search method
    """
    id: str
    text: str
    page: int
    chunk: int
    score: float
    source: str = "vector"


class HybridSearcher:
    """Hybrid vector + keyword searcher.

    Builds BM25 index on initialization for keyword search.
    Combines vector similarity (cosine) with BM25 scores.

    Example:
        searcher = HybridSearcher("./data/vs", dimension=768)
        results = searcher.search("query", top_k=5, threshold=0.3)
    """

    def __init__(self, store_path: Path | str, dimension: int) -> None:
        """Initialize hybrid searcher.

        Args:
            store_path: Path to vector store directory
            dimension: Embedding vector dimension
        """
        self.store = load_vector_store(store_path, dimension)
        self._bm25: BM25Okapi | None = None
        self._corpus: list[list[str]] = []
        self._metadata: list[dict] = []
        self._build_bm25()

    def _build_bm25(self) -> None:
        """Build BM25 index from all stored chunks."""
        if self.store.count == 0:
            return
        all_results = self.store.search(np.zeros(self.store.dimension, dtype=np.float32), self.store.count)
        self._metadata = [meta for _, meta in all_results]
        self._corpus = [self._tokenize(meta["text"]) for meta in self._metadata]
        self._bm25 = BM25Okapi(self._corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of lowercase word tokens
        """
        return re.findall(r"\b\w+\b", text.lower())

    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        use_hybrid: bool | None = None,
    ) -> list[SearchResult]:
        """Search using hybrid vector + keyword search.

        Args:
            query: Search query string
            top_k: Number of results (default: settings.top_k)
            threshold: Minimum similarity score (default: settings.similarity_threshold)
            use_hybrid: Enable hybrid search (default: settings.use_hybrid_search)

        Returns:
            List of SearchResult sorted by score descending
        """
        top_k = top_k or settings.top_k
        threshold = threshold if threshold is not None else settings.similarity_threshold
        use_hybrid = use_hybrid if use_hybrid is not None else settings.use_hybrid_search

        query_vector = embed_query(query)
        vector_results = self.store.search(query_vector, top_k * 2 if use_hybrid else top_k)

        results: list[SearchResult] = []
        for score, meta in vector_results:
            if score >= threshold:
                results.append(SearchResult(
                    id=meta["id"],
                    text=meta["text"],
                    page=meta["page"],
                    chunk=meta["chunk"],
                    score=score,
                    source="vector",
                ))

        if use_hybrid and self._bm25 and self._metadata:
            bm25_scores = self._bm25.get_scores(self._tokenize(query))
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
            for idx in top_bm25_indices:
                if bm25_scores[idx] > 0:
                    meta = self._metadata[idx]
                    if not any(r.id == meta["id"] for r in results):
                        results.append(SearchResult(
                            id=meta["id"],
                            text=meta["text"],
                            page=meta["page"],
                            chunk=meta["chunk"],
                            score=float(bm25_scores[idx]),
                            source="keyword",
                        ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def search_async(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        use_hybrid: bool | None = None,
    ) -> list[SearchResult]:
        """Async version of search using async embedding.

        Args:
            query: Search query string
            top_k: Number of results
            threshold: Minimum similarity score
            use_hybrid: Enable hybrid search

        Returns:
            List of SearchResult sorted by score descending
        """
        top_k = top_k or settings.top_k
        threshold = threshold if threshold is not None else settings.similarity_threshold
        use_hybrid = use_hybrid if use_hybrid is not None else settings.use_hybrid_search

        query_vector = await embed_query_async(query)
        vector_results = self.store.search(query_vector, top_k * 2 if use_hybrid else top_k)

        results: list[SearchResult] = []
        for score, meta in vector_results:
            if score >= threshold:
                results.append(SearchResult(
                    id=meta["id"],
                    text=meta["text"],
                    page=meta["page"],
                    chunk=meta["chunk"],
                    score=score,
                    source="vector",
                ))

        if use_hybrid and self._bm25 and self._metadata:
            bm25_scores = self._bm25.get_scores(self._tokenize(query))
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
            for idx in top_bm25_indices:
                if bm25_scores[idx] > 0:
                    meta = self._metadata[idx]
                    if not any(r.id == meta["id"] for r in results):
                        results.append(SearchResult(
                            id=meta["id"],
                            text=meta["text"],
                            page=meta["page"],
                            chunk=meta["chunk"],
                            score=float(bm25_scores[idx]),
                            source="keyword",
                        ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


def search(
    query: str,
    store_path: Path | str,
    dimension: int,
    top_k: int | None = None,
    threshold: float | None = None,
    use_hybrid: bool | None = None,
) -> list[SearchResult]:
    """Convenience function for hybrid search.

    Args:
        query: Search query string
        store_path: Path to vector store directory
        dimension: Embedding vector dimension
        top_k: Number of results (default: settings.top_k)
        threshold: Minimum similarity score (default: settings.similarity_threshold)
        use_hybrid: Enable hybrid search (default: settings.use_hybrid_search)

    Returns:
        List of SearchResult sorted by score descending
    """
    searcher = HybridSearcher(store_path, dimension)
    return searcher.search(query, top_k, threshold, use_hybrid)


async def search_async(
    query: str,
    store_path: Path | str,
    dimension: int,
    top_k: int | None = None,
    threshold: float | None = None,
    use_hybrid: bool | None = None,
) -> list[SearchResult]:
    """Async convenience function for hybrid search.

    Args:
        query: Search query string
        store_path: Path to vector store directory
        dimension: Embedding vector dimension
        top_k: Number of results
        threshold: Minimum similarity score
        use_hybrid: Enable hybrid search

    Returns:
        List of SearchResult sorted by score descending
    """
    searcher = HybridSearcher(store_path, dimension)
    return await searcher.search_async(query, top_k, threshold, use_hybrid)