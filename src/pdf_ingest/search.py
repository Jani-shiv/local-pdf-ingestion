"""Hybrid search: vector similarity + keyword (BM25) search."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .store import VectorStore, load_vector_store
from .embed import embed_query


@dataclass
class SearchResult:
    id: str
    text: str
    page: int
    chunk: int
    score: float
    source: str = "vector"


class HybridSearcher:
    def __init__(self, store_path: Path | str, dimension: int):
        self.store = load_vector_store(store_path, dimension)
        self._bm25: BM25Okapi | None = None
        self._corpus: list[list[str]] = []
        self._metadata: list[dict] = []
        self._build_bm25()

    def _build_bm25(self) -> None:
        if self.store.count == 0:
            return
        all_results = self.store.search(np.zeros(self.store.dimension, dtype=np.float32), self.store.count)
        self._metadata = [meta for _, meta in all_results]
        self._corpus = [self._tokenize(meta["text"]) for meta in self._metadata]
        self._bm25 = BM25Okapi(self._corpus)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        use_hybrid: bool | None = None,
    ) -> list[SearchResult]:
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


def search(
    query: str,
    store_path: Path | str,
    dimension: int,
    top_k: int | None = None,
    threshold: float | None = None,
    use_hybrid: bool | None = None,
) -> list[SearchResult]:
    searcher = HybridSearcher(store_path, dimension)
    return searcher.search(query, top_k, threshold, use_hybrid)