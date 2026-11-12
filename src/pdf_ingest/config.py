"""Configuration management for pdf-ingest."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:8b"

    # Embeddings fallback
    use_local_embeddings: bool = False
    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunk_strategy: Literal["character", "semantic", "recursive"] = "semantic"

    # Vector store
    vector_store: Literal["faiss", "chroma"] = "faiss"
    vector_store_path: Path = Path("data/vector_store")

    # Search
    top_k: int = 5
    similarity_threshold: float = 0.3
    use_hybrid_search: bool = True

    # LLM
    temperature: float = 0.1
    max_tokens: int = 2048
    system_prompt: str = (
        "You are a helpful assistant that answers questions based only on the provided context. "
        "Cite your sources using [chunk_id] format. If the context doesn't contain the answer, "
        "say 'I don't have enough information to answer this question.'"
    )

    # Processing
    batch_size: int = 16
    num_workers: int = 4

    # Output
    output_dir: Path = Path("output")
    export_formats: list[str] = ["json", "markdown", "txt"]


settings = Settings()