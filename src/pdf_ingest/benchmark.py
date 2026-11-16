"""Performance benchmarking for pdf-ingest."""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .chat import RAGChat
from .chunk import chunk_pages
from .embed import embed_texts, embed_texts_async
from .pipeline import IngestionPipeline
from .search import HybridSearcher


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time: float
    min_time: float
    max_time: float
    mean_time: float
    median_time: float
    std_dev: float
    ops_per_second: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _time_it(func, *args, iterations: int = 10, **kwargs) -> BenchmarkResult:
    """Time a function over multiple iterations."""
    times = []
    result = None

    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=sum(times),
        min_time=min(times),
        max_time=max(times),
        mean_time=statistics.mean(times),
        median_time=statistics.median(times),
        std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
        ops_per_second=iterations / sum(times) if sum(times) > 0 else 0.0,
        metadata={"result": str(type(result))},
    )


async def _time_it_async(func, *args, iterations: int = 10, **kwargs) -> BenchmarkResult:
    """Time an async function over multiple iterations."""
    times = []
    result = None

    for _ in range(iterations):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=sum(times),
        min_time=min(times),
        max_time=max(times),
        mean_time=statistics.mean(times),
        median_time=statistics.median(times),
        std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
        ops_per_second=iterations / sum(times) if sum(times) > 0 else 0.0,
        metadata={"result": str(type(result))},
    )


def benchmark_chunking(text: str, iterations: int = 100) -> list[BenchmarkResult]:
    """Benchmark text chunking strategies."""
    strategies = ["character", "semantic", "recursive"]
    results = []

    for strategy in strategies:
        def chunk_func():
            return chunk_pages(
                [{"page": 1, "text": text}],
                chunk_size=1000,
                overlap=150,
                strategy=strategy,
            )

        result = _time_it(chunk_func, iterations=iterations)
        result.metadata["strategy"] = strategy
        result.metadata["text_length"] = len(text)
        results.append(result)

    return results


def benchmark_embeddings(texts: list[str], iterations: int = 10) -> list[BenchmarkResult]:
    """Benchmark embedding generation (local only)."""
    results = []

    # Sync embeddings
    result = _time_it(embed_texts, texts, use_local=True, iterations=iterations)
    result.metadata["mode"] = "sync_local"
    results.append(result)

    # Async embeddings
    async def async_embed():
        return await embed_texts_async(texts, use_local=True)

    result = asyncio.run(_time_it_async(async_embed, iterations=iterations))
    result.metadata["mode"] = "async_local"
    results.append(result)

    return results


def benchmark_search(
    vector_store_path: str | Path,
    dimension: int,
    queries: list[str],
    iterations: int = 50,
) -> list[BenchmarkResult]:
    """Benchmark search operations."""
    results = []

    searcher = HybridSearcher(vector_store_path, dimension)

    # Sync search
    def search_func(query: str):
        return searcher.search(query, top_k=5)

    for query in queries:
        result = _time_it(search_func, query, iterations=iterations)
        result.metadata["query"] = query[:50]
        result.metadata["mode"] = "sync"
        results.append(result)

    # Async search
    async def async_search_func(query: str):
        return await searcher.search_async(query, top_k=5)

    for query in queries:
        result = asyncio.run(_time_it_async(async_search_func, query, iterations=iterations))
        result.metadata["query"] = query[:50]
        result.metadata["mode"] = "async"
        results.append(result)

    return results


def benchmark_ingestion(
    files: list[str | Path],
    vector_store_path: str | Path,
    iterations: int = 5,
) -> BenchmarkResult:
    """Benchmark document ingestion."""
    pipeline = IngestionPipeline(vector_store_path=vector_store_path)

    def ingest_func():
        return pipeline.ingest_batch(files)

    result = _time_it(ingest_func, iterations=iterations)
    result.metadata["file_count"] = len(files)
    return result


def benchmark_chat(
    vector_store_path: str | Path,
    dimension: int,
    queries: list[str],
    iterations: int = 10,
) -> list[BenchmarkResult]:
    """Benchmark chat operations (requires Ollama)."""
    results = []

    chat = RAGChat(vector_store_path, dimension)

    for query in queries:
        def chat_func():
            return chat.chat(query, top_k=5)

        result = _time_it(chat_func, iterations=iterations)
        result.metadata["query"] = query[:50]
        result.metadata["mode"] = "sync"
        results.append(result)

    return results


def run_benchmark(
    output_path: str | Path | None = None,
    include_chat: bool = False,
) -> dict[str, list[BenchmarkResult]]:
    """Run comprehensive benchmarks and return results.

    Args:
        output_path: Optional path to save JSON results
        include_chat: Whether to include chat benchmarks (requires Ollama)

    Returns:
        Dictionary of benchmark categories to results
    """

    all_results = {}

    # Generate test data
    test_text = "This is a test sentence. " * 200  # ~5000 chars
    test_texts = [f"Document {i}: " + test_text for i in range(10)]
    test_queries = [
        "What is the main topic?",
        "Summarize the key points",
        "Find information about testing",
        "Extract all numbers",
        "What are the conclusions?",
    ]

    # Create temp vector store for search/chat benchmarks
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "bench_vs"

        # Build test vector store
        IngestionPipeline(vector_store_path=tmp_path)
        for _i, text in enumerate(test_texts[:3]):
            chunks = chunk_pages([{"page": 1, "text": text}], chunk_size=500, overlap=50)
            from .embed import embed_texts
            vectors = embed_texts([c["text"] for c in chunks], use_local=True)
            from .store import save_vector_store
            save_vector_store(chunks, np.array(vectors, dtype=np.float32), tmp_path)

        dimension = 384  # all-MiniLM-L6-v2 dimension

        # Run benchmarks
        all_results["chunking"] = benchmark_chunking(test_text)

        all_results["embeddings"] = benchmark_embeddings(test_texts)

        all_results["search"] = benchmark_search(tmp_path, dimension, test_queries)

        all_results["ingestion"] = [benchmark_ingestion(test_texts[:3], tmp_path)]

        if include_chat:
            all_results["chat"] = benchmark_chat(tmp_path, dimension, test_queries)

    # Print summary
    for results in all_results.values():
        for _r in results:
            pass

    # Save to file if requested
    if output_path:
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            cat: [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "total_time": r.total_time,
                    "min_time": r.min_time,
                    "max_time": r.max_time,
                    "mean_time": r.mean_time,
                    "median_time": r.median_time,
                    "std_dev": r.std_dev,
                    "ops_per_second": r.ops_per_second,
                    "metadata": r.metadata,
                }
                for r in results
            ]
            for cat, results in all_results.items()
        }
        output_path.write_text(json.dumps(serializable, indent=2))

    return all_results


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else None
    include_chat = "--chat" in sys.argv
    run_benchmark(output, include_chat=include_chat)