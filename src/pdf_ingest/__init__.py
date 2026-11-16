"""Local PDF ingestion pipeline: extract, chunk, embed, store, search, chat, summarize."""

from __future__ import annotations

from .benchmark import BenchmarkResult, run_benchmark
from .chat import ChatResponse, RAGChat, create_chat
from .chunk import Chunk, chunk_pages, split_text
from .config import ConfigurationError, settings
from .embed import (
    EmbeddingError,
    embed_query,
    embed_query_async,
    embed_texts,
    embed_texts_async,
)
from .extract import (
    DocumentMetadata,
    Page,
    extract_pages,
    extract_pages_with_metadata,
)
from .extract_audio import AudioSegment, extract_audio_segments, extract_audio_text
from .extract_ocr import OCRResult, extract_image_ocr, extract_pdf_ocr
from .extract_office import OfficeContent, extract_office
from .extract_web import WebContent, extract_web_content, extract_web_multiple
from .logging import ColoredFormatter, JSONFormatter, LogContext, get_logger, setup_logging
from .pipeline import (
    BatchIngestionResult,
    IngestionPipeline,
    IngestionResult,
    KnowledgeBase,
    create_knowledge_base,
    create_pipeline,
)
from .plugins import (
    ExtractorPlugin,
    PluginRegistry,
    get_extractor,
    register_extractor,
)
from .progress import (
    MultiProgress,
    ProgressBar,
    Spinner,
    format_bytes,
    format_duration,
    progress_bar,
    track,
    track_range,
)
from .search import HybridSearcher, SearchResult, search, search_async
from .specialize import (
    Contract,
    ContractExtractor,
    Invoice,
    InvoiceExtractor,
    MedicalExtractor,
    MedicalReport,
    Recipe,
    RecipeExtractor,
    StudyExtractor,
    StudyGuide,
)
from .store import (
    ChromaStore,
    FAISSStore,
    VectorStore,
    get_vector_store,
    load_dataset,
    load_vector_store,
    save_dataset,
    save_vector_store,
)
from .summarize import Summarizer, Summary, create_summarizer
from .validation import ConfigurationError as ValidationConfigurationError
from .validation import InputValidator

__version__ = "0.3.0"

__all__ = [
    "AudioSegment",
    "BatchIngestionResult",
    "BenchmarkResult",
    "ChatResponse",
    "ChromaStore",
    "Chunk",
    "ColoredFormatter",
    "ConfigurationError",
    "Contract",
    "ContractExtractor",
    "DocumentMetadata",
    "EmbeddingError",
    # Plugins
    "ExtractorPlugin",
    "FAISSStore",
    "HybridSearcher",
    "IngestionPipeline",
    "IngestionResult",
    # Validation
    "InputValidator",
    "Invoice",
    "InvoiceExtractor",
    "JSONFormatter",
    "KnowledgeBase",
    "LogContext",
    "MedicalExtractor",
    "MedicalReport",
    "MultiProgress",
    "OCRResult",
    "OfficeContent",
    # Core
    "Page",
    "PluginRegistry",
    # Progress
    "ProgressBar",
    "RAGChat",
    "Recipe",
    # Domain extractors
    "RecipeExtractor",
    "SearchResult",
    "Spinner",
    "StudyExtractor",
    "StudyGuide",
    "Summarizer",
    "Summary",
    "ValidationConfigurationError",
    "VectorStore",
    "WebContent",
    "chunk_pages",
    "create_chat",
    "create_knowledge_base",
    "create_pipeline",
    "create_summarizer",
    "embed_query",
    "embed_query_async",
    "embed_texts",
    "embed_texts_async",
    "extract_audio_segments",
    # Multi-source
    "extract_audio_text",
    "extract_image_ocr",
    "extract_office",
    "extract_pages",
    "extract_pages_with_metadata",
    "extract_pdf_ocr",
    "extract_web_content",
    "extract_web_multiple",
    "format_bytes",
    "format_duration",
    "get_extractor",
    "get_logger",
    "get_vector_store",
    "load_dataset",
    "load_vector_store",
    "progress_bar",
    "register_extractor",
    # Benchmark
    "run_benchmark",
    "save_dataset",
    "save_vector_store",
    "search",
    "search_async",
    # Config
    "settings",
    # Logging
    "setup_logging",
    "split_text",
    "track",
    "track_range",
]
