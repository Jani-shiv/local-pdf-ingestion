"""Local PDF ingestion pipeline: extract, chunk, embed, store, search, chat, summarize."""
from __future__ import annotations

from .chunk import Chunk, chunk_pages, split_text
from .config import settings, ConfigurationError
from .embed import (
    embed_texts,
    embed_query,
    embed_texts_async,
    embed_query_async,
    EmbeddingError,
)
from .extract import extract_pages, Page
from .extract_audio import extract_audio_text, extract_audio_segments, AudioSegment
from .extract_web import extract_web_content, extract_web_multiple, WebContent
from .extract_ocr import extract_pdf_ocr, extract_image_ocr, OCRResult
from .extract_office import extract_office, OfficeContent
from .store import (
    save_dataset,
    load_dataset,
    save_vector_store,
    load_vector_store,
    VectorStore,
    FAISSStore,
    ChromaStore,
    get_vector_store,
)
from .search import search, search_async, SearchResult, HybridSearcher
from .chat import RAGChat, ChatResponse, create_chat
from .summarize import create_summarizer, Summarizer, Summary
from .pipeline import IngestionPipeline, KnowledgeBase, create_pipeline, create_knowledge_base, IngestionResult
from .specialize import (
    RecipeExtractor, Recipe,
    InvoiceExtractor, Invoice,
    MedicalExtractor, MedicalReport,
    ContractExtractor, Contract,
    StudyExtractor, StudyGuide,
)
from .logging import setup_logging, get_logger, LogContext, JSONFormatter, ColoredFormatter
from .validation import InputValidator, ConfigurationError as ValidationConfigurationError
from .progress import (
    ProgressBar,
    progress_bar,
    track,
    track_range,
    MultiProgress,
    Spinner,
    format_duration,
    format_bytes,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "Page", "extract_pages",
    "Chunk", "chunk_pages", "split_text",
    "embed_texts", "embed_query", "embed_texts_async", "embed_query_async", "EmbeddingError",
    "save_dataset", "load_dataset",
    "save_vector_store", "load_vector_store", "VectorStore", "FAISSStore", "ChromaStore", "get_vector_store",
    "search", "search_async", "SearchResult", "HybridSearcher",
    "RAGChat", "ChatResponse", "create_chat",
    "create_summarizer", "Summarizer", "Summary",
    "IngestionPipeline", "KnowledgeBase", "create_pipeline", "create_knowledge_base", "IngestionResult",
    # Multi-source
    "extract_audio_text", "extract_audio_segments", "AudioSegment",
    "extract_web_content", "extract_web_multiple", "WebContent",
    "extract_pdf_ocr", "extract_image_ocr", "OCRResult",
    "extract_office", "OfficeContent",
    # Domain extractors
    "RecipeExtractor", "Recipe",
    "InvoiceExtractor", "Invoice",
    "MedicalExtractor", "MedicalReport",
    "ContractExtractor", "Contract",
    "StudyExtractor", "StudyGuide",
    # Config
    "settings", "ConfigurationError",
    # Logging
    "setup_logging", "get_logger", "LogContext", "JSONFormatter", "ColoredFormatter",
    # Validation
    "InputValidator", "ValidationConfigurationError",
    # Progress
    "ProgressBar", "progress_bar", "track", "track_range", "MultiProgress", "Spinner",
    "format_duration", "format_bytes",
]