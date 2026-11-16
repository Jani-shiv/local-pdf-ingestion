"""Configuration management for pdf-ingest."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(ValueError):
    """Raised when configuration validation fails."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    embed_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name",
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        description="LLM model name for chat",
    )

    # Embeddings fallback
    use_local_embeddings: bool = Field(
        default=False,
        description="Use local SentenceTransformers instead of Ollama",
    )
    local_embed_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local embedding model name",
    )

    # Chunking
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=2000,
        description="Overlap between chunks in characters",
    )
    chunk_strategy: Literal["character", "semantic", "recursive"] = Field(
        default="semantic",
        description="Chunking strategy",
    )

    # Vector store
    vector_store: Literal["faiss", "chroma"] = Field(
        default="faiss",
        description="Vector store backend",
    )
    vector_store_path: Path = Field(
        default=Path("data/vector_store"),
        description="Path to vector store directory",
    )

    # Search
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to return",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for results",
    )
    use_hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid vector + keyword search",
    )

    # LLM
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=2048,
        ge=100,
        le=8192,
        description="Maximum tokens in response",
    )
    system_prompt: str = Field(
        default=(
            "You are a helpful assistant that answers questions based only on the provided context. "
            "Cite your sources using [chunk_id] format. If the context doesn't contain the answer, "
            "say 'I don't have enough information to answer this question.'"
        ),
        description="System prompt for LLM",
    )

    # Processing
    batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Batch size for embedding requests",
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker processes",
    )

    # Output
    output_dir: Path = Field(
        default=Path("output"),
        description="Output directory for exports",
    )
    export_formats: list[str] = Field(
        default=["json", "markdown", "txt"],
        description="Supported export formats",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    json_logs: bool = Field(
        default=False,
        description="Output logs in JSON format",
    )
    log_file: Path | None = Field(
        default=None,
        description="Path to log file (optional)",
    )

    @field_validator("ollama_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ConfigurationError("OLLAMA_URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ConfigurationError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ConfigurationError(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")
        return v_upper

    @field_validator("export_formats", mode="before")
    @classmethod
    def parse_export_formats(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(",")]
        return v

    @model_validator(mode="after")
    def validate_paths(self) -> Settings:
        self.vector_store_path = self.vector_store_path.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        if self.log_file:
            self.log_file = self.log_file.expanduser().resolve()
        return self


def validate_settings() -> Settings:
    """Validate settings and raise ConfigurationError with clear messages."""
    try:
        return Settings()
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid configuration: {e}") from e


settings = validate_settings()
