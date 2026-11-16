"""Comprehensive edge case tests for pdf-ingest."""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pdf_ingest import (
    BatchIngestionResult,
    ChatResponse,
    ChromaStore,
    FAISSStore,
    HybridSearcher,
    IngestionPipeline,
    KnowledgeBase,
    RAGChat,
    SearchResult,
    chunk_pages,
    embed_query_async,
    embed_texts_async,
    get_vector_store,
    settings,
    split_text,
)
from pdf_ingest.config import ConfigurationError, Settings
from pdf_ingest.logging import ColoredFormatter, JSONFormatter, LogContext, get_logger
from pdf_ingest.progress import MultiProgress, Spinner, progress_bar, track, track_range
from pdf_ingest.validation import (
    sanitize_string,
    validate_chunk_overlap,
    validate_chunk_size,
    validate_chunk_strategy,
    validate_directory_path,
    validate_extract_format,
    validate_extract_type,
    validate_file_path,
    validate_log_level,
    validate_max_tokens,
    validate_model_name,
    validate_output_path,
    validate_positive_int,
    validate_query,
    validate_store_path,
    validate_study_format,
    validate_summarize_style,
    validate_temperature,
    validate_threshold,
    validate_top_k,
    validate_url,
    validate_urls,
    validate_vector_store,
)

# ===== Config validation tests =====


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_settings(self) -> None:
        """Test that default settings are valid."""
        s = Settings()
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 150
        assert s.vector_store in ("faiss", "chroma")

    def test_invalid_ollama_url(self) -> None:
        """Test invalid Ollama URL validation."""
        with pytest.raises(ConfigurationError, match="must start with http"):
            Settings(ollama_url="invalid-url")

    def test_chunk_overlap_exceeds_chunk_size(self) -> None:
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ConfigurationError, match="must be less than CHUNK_SIZE"):
            Settings(chunk_size=100, chunk_overlap=150)

    def test_invalid_log_level(self) -> None:
        """Test invalid log level."""
        with pytest.raises(ConfigurationError, match="LOG_LEVEL must be one of"):
            Settings(log_level="INVALID")

    def test_valid_log_levels(self) -> None:
        """Test all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            s = Settings(log_level=level)
            assert s.log_level == level

    def test_export_formats_parsing(self) -> None:
        """Test export formats parsing from string."""
        s = Settings(export_formats="json,markdown,txt")
        assert s.export_formats == ["json", "markdown", "txt"]

    def test_path_resolution(self) -> None:
        """Test that paths are resolved."""
        s = Settings(vector_store_path="./data/vs")
        assert s.vector_store_path.is_absolute()


# ===== Input validation tests =====


class TestInputValidation:
    """Tests for input validation and sanitization."""

    def test_sanitize_string_basic(self) -> None:
        """Test basic string sanitization."""
        assert sanitize_string("hello world") == "hello world"
        assert sanitize_string("  hello  world  ") == "hello world"

    def test_sanitize_string_control_chars(self) -> None:
        """Test removal of control characters."""
        assert sanitize_string("hello\x00world") == "helloworld"
        assert sanitize_string("hello\nworld") == "hello world"

    def test_sanitize_string_truncation(self) -> None:
        """Test string truncation."""
        long_str = "x" * 15000
        result = sanitize_string(long_str, max_length=100)
        assert len(result) == 100

    def test_validate_file_path_exists(self, tmp_path) -> None:
        """Test file path validation."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")
        result = validate_file_path(test_file)
        assert result == test_file.resolve()

    def test_validate_file_path_not_exists(self) -> None:
        """Test file path validation for non-existent file."""
        with pytest.raises(ConfigurationError, match="File not found"):
            validate_file_path("/nonexistent/file.pdf")

    def test_validate_file_path_extension(self, tmp_path) -> None:
        """Test file extension validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        with pytest.raises(ConfigurationError, match="Unsupported file type"):
            validate_file_path(test_file, allowed_extensions={".pdf"})

    def test_validate_directory_path(self, tmp_path) -> None:
        """Test directory path validation."""
        result = validate_directory_path(tmp_path)
        assert result == tmp_path.resolve()

    def test_validate_directory_path_not_exists(self) -> None:
        """Test directory path validation for non-existent dir."""
        with pytest.raises(ConfigurationError, match="Directory not found"):
            validate_directory_path("/nonexistent/dir")

    def test_validate_url_valid(self) -> None:
        """Test valid URL validation."""
        assert validate_url("https://example.com") == "https://example.com"
        assert validate_url("http://localhost:8080/path") == "http://localhost:8080/path"

    def test_validate_url_invalid(self) -> None:
        """Test invalid URL validation."""
        with pytest.raises(ConfigurationError, match="Invalid URL format"):
            validate_url("not-a-url")

    def test_validate_url_ftp_not_allowed(self) -> None:
        """Test that FTP URLs are rejected."""
        with pytest.raises(ConfigurationError, match="must use http or https"):
            validate_url("ftp://example.com")

    def test_validate_urls_list(self) -> None:
        """Test URL list validation."""
        urls = ["https://a.com", "https://b.com"]
        result = validate_urls(urls)
        assert result == urls

    def test_validate_urls_empty(self) -> None:
        """Test empty URL list."""
        with pytest.raises(ConfigurationError, match="At least one URL"):
            validate_urls([])

    def test_validate_chunk_size(self) -> None:
        """Test chunk size validation."""
        assert validate_chunk_size(1000) == 1000
        with pytest.raises(ConfigurationError, match="between 100 and 10000"):
            validate_chunk_size(50)

    def test_validate_chunk_overlap(self) -> None:
        """Test chunk overlap validation."""
        assert validate_chunk_overlap(150, 1000) == 150
        with pytest.raises(ConfigurationError, match="must be >= 0 and < chunk_size"):
            validate_chunk_overlap(1000, 1000)

    def test_validate_top_k(self) -> None:
        """Test top_k validation."""
        assert validate_top_k(5) == 5
        with pytest.raises(ConfigurationError, match="between 1 and 50"):
            validate_top_k(100)

    def test_validate_threshold(self) -> None:
        """Test threshold validation."""
        assert validate_threshold(0.5) == 0.5
        with pytest.raises(ConfigurationError, match="between 0.0 and 1.0"):
            validate_threshold(1.5)

    def test_validate_temperature(self) -> None:
        """Test temperature validation."""
        assert validate_temperature(0.7) == 0.7
        with pytest.raises(ConfigurationError, match="between 0.0 and 2.0"):
            validate_temperature(3.0)

    def test_validate_max_tokens(self) -> None:
        """Test max_tokens validation."""
        assert validate_max_tokens(2048) == 2048
        with pytest.raises(ConfigurationError, match="between 100 and 8192"):
            validate_max_tokens(50)

    def test_validate_model_name(self) -> None:
        """Test model name validation."""
        assert validate_model_name("llama3.1:8b") == "llama3.1:8b"
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            validate_model_name("")
        with pytest.raises(ConfigurationError, match="Invalid model name"):
            validate_model_name("model@name")

    def test_validate_output_path(self, tmp_path) -> None:
        """Test output path validation."""
        output = tmp_path / "out.json"
        result = validate_output_path(output)
        assert result == output.resolve()

    def test_validate_output_path_invalid_ext(self, tmp_path) -> None:
        """Test invalid output extension."""
        output = tmp_path / "out.xyz"
        with pytest.raises(ConfigurationError, match="Unsupported output format"):
            validate_output_path(output)

    def test_validate_store_path(self, tmp_path) -> None:
        """Test store path validation."""
        store = tmp_path / "store"
        result = validate_store_path(store)
        assert result == store.resolve()
        assert store.exists()

    def test_validate_query(self) -> None:
        """Test query validation."""
        assert validate_query("hello world") == "hello world"
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            validate_query("")

    def test_validate_extract_type(self) -> None:
        """Test extract type validation."""
        assert validate_extract_type("recipes") == "recipes"
        with pytest.raises(ConfigurationError, match="Invalid extract type"):
            validate_extract_type("invalid")

    def test_validate_extract_format(self) -> None:
        """Test extract format validation."""
        assert validate_extract_format("json", "recipes") == "json"
        assert validate_extract_format("csv", "invoices") == "csv"
        with pytest.raises(ConfigurationError, match="Invalid format"):
            validate_extract_format("csv", "recipes")

    def test_validate_study_format(self) -> None:
        """Test study format validation."""
        assert validate_study_format("flashcards") == "flashcards"
        with pytest.raises(ConfigurationError, match="Invalid study format"):
            validate_study_format("invalid")

    def test_validate_summarize_style(self) -> None:
        """Test summarize style validation."""
        assert validate_summarize_style("concise") == "concise"
        with pytest.raises(ConfigurationError, match="Invalid style"):
            validate_summarize_style("invalid")

    def test_validate_chunk_strategy(self) -> None:
        """Test chunk strategy validation."""
        assert validate_chunk_strategy("semantic") == "semantic"
        with pytest.raises(ConfigurationError, match="Invalid chunk strategy"):
            validate_chunk_strategy("invalid")

    def test_validate_vector_store(self) -> None:
        """Test vector store validation."""
        assert validate_vector_store("faiss") == "faiss"
        with pytest.raises(ConfigurationError, match="Invalid vector store"):
            validate_vector_store("invalid")

    def test_validate_log_level(self) -> None:
        """Test log level validation."""
        assert validate_log_level("DEBUG") == "DEBUG"
        with pytest.raises(ConfigurationError, match="Invalid log level"):
            validate_log_level("INVALID")

    def test_validate_positive_int(self) -> None:
        """Test positive integer validation."""
        assert validate_positive_int(5, "test") == 5
        with pytest.raises(ConfigurationError, match="must be an integer"):
            validate_positive_int("5", "test")
        with pytest.raises(ConfigurationError, match="between 1 and 10"):
            validate_positive_int(15, "test", min_val=1, max_val=10)


# ===== Async embedding tests =====


class TestAsyncEmbeddings:
    """Tests for async embedding functions."""

    @pytest.mark.asyncio
    async def test_embed_texts_async_local(self, monkeypatch) -> None:
        """Test async local embeddings."""
        monkeypatch.setattr("pdf_ingest.embed.settings.use_local_embeddings", True)

        # Mock the LocalEmbedder.embed_async
        from pdf_ingest.embed import LocalEmbedder

        mock_embed = AsyncMock(return_value=np.array([[0.1, 0.2], [0.3, 0.4]]))
        monkeypatch.setattr(LocalEmbedder, "embed_async", mock_embed)

        result = await embed_texts_async(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_embed_query_async_local(self, monkeypatch) -> None:
        """Test async query embedding with local embeddings."""
        monkeypatch.setattr("pdf_ingest.embed.settings.use_local_embeddings", True)

        from pdf_ingest.embed import LocalEmbedder

        mock_embed = AsyncMock(return_value=np.array([[0.1, 0.2]]))
        monkeypatch.setattr(LocalEmbedder, "embed_async", mock_embed)

        result = await embed_query_async("test query")
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


# ===== Async search tests =====


class TestAsyncSearch:
    """Tests for async search functions."""

    @pytest.mark.asyncio
    async def test_search_async(self, tmp_path, monkeypatch) -> None:
        """Test async search function."""
        from pdf_ingest.chunk import Chunk
        from pdf_ingest.search import HybridSearcher
        from pdf_ingest.store import FAISSStore

        # Create a test store
        store = FAISSStore(tmp_path / "faiss", dimension=3)
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        chunks = [Chunk(id="c1", page=1, chunk=0, text="test")]
        store.add(vectors, chunks)
        store.save()

        # Mock embed_query_async
        async def mock_embed(q):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        monkeypatch.setattr("pdf_ingest.search.embed_query_async", mock_embed)

        searcher = HybridSearcher(tmp_path / "faiss", dimension=3)
        results = await searcher.search_async("test", top_k=1)

        assert len(results) == 1
        assert results[0].id == "c1"


# ===== Progress bar tests =====


class TestProgressBar:
    """Tests for progress bar utilities."""

    def test_progress_bar_context_manager(self) -> None:
        """Test progress bar as context manager."""
        with progress_bar(total=10, desc="Test") as pbar:
            pbar.update(5)
            assert pbar._bar.n == 5

    def test_track_iterable(self) -> None:
        """Test tracking an iterable."""
        items = list(track(range(5), desc="Test", disable=True))
        assert items == [0, 1, 2, 3, 4]

    def test_track_range(self) -> None:
        """Test tracking a range."""
        items = list(track_range(5, desc="Test", disable=True))
        assert items == [0, 1, 2, 3, 4]

    def test_multi_progress(self) -> None:
        """Test multiple progress bars."""
        mp = MultiProgress()
        bar1 = mp.add_bar("bar1", total=10, desc="Bar 1")
        bar2 = mp.add_bar("bar2", total=5, desc="Bar 2")

        bar1.update(5)
        bar2.update(3)

        assert bar1._bar.n == 5
        assert bar2._bar.n == 3

        mp.close_all()

    def test_spinner(self) -> None:
        """Test spinner (just ensure it doesn't crash)."""
        spinner = Spinner(desc="Test", disable=True)
        spinner.start()
        spinner.stop()


# ===== Logging tests =====


class TestLogging:
    """Tests for logging configuration."""

    def test_setup_logging(self) -> None:
        """Test logging setup."""
        logger = get_logger("test")
        assert logger.name == "pdf_ingest.test"

    def test_log_context(self) -> None:
        """Test log context manager."""
        logger = get_logger("test")
        with LogContext(logger, key="value") as ctx:
            assert ctx == logger

    def test_json_formatter(self) -> None:
        """Test JSON formatter."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test_module"
        record.funcName = "test_func"
        result = formatter.format(record)
        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["module"] == "test_module"

    def test_colored_formatter(self) -> None:
        """Test colored formatter."""
        formatter = ColoredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "Test" in result


# ===== Batch ingestion tests =====


class TestBatchIngestion:
    """Tests for batch ingestion functionality."""

    def test_batch_ingestion_result(self) -> None:
        """Test BatchIngestionResult dataclass."""
        result = BatchIngestionResult(
            total_files=3,
            successful=2,
            failed=1,
            results=[],
            errors=[{"file": "test.pdf", "error": "Failed"}],
        )
        assert result.total_files == 3
        assert result.successful == 2
        assert result.failed == 1

    def test_ingest_batch_with_progress(self, tmp_path, sample_pdf, monkeypatch) -> None:
        """Test batch ingestion with progress."""
        monkeypatch.setattr("pdf_ingest.pipeline.settings.use_local_embeddings", True)

        pl = IngestionPipeline(vector_store_path=tmp_path / "vs", chunk_size=500, overlap=50)
        pl.ingest_pdf(sample_pdf)

        # Test batch ingest
        result = pl.ingest_batch_with_progress([sample_pdf])
        assert result.total_files == 1
        assert result.successful == 1
        assert result.failed == 0


# ===== Vector store edge cases =====


class TestVectorStoreEdgeCases:
    """Tests for vector store edge cases."""

    def test_faiss_store_empty_search(self, tmp_path) -> None:
        """Test searching empty FAISS store."""
        store = FAISSStore(tmp_path / "faiss", dimension=3)
        results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=5)
        assert results == []

    def test_faiss_store_add_empty_vectors(self, tmp_path) -> None:
        """Test adding empty vectors."""
        store = FAISSStore(tmp_path / "faiss", dimension=3)
        store.add(np.array([], dtype=np.float32).reshape(0, 3), [])
        assert store.count == 0

    def test_chroma_store_creation(self, tmp_path, monkeypatch) -> None:
        """Test ChromaStore creation (mocked)."""
        # Mock chromadb to avoid actual dependency
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("pdf_ingest.store.chromadb.PersistentClient", return_value=mock_client):
            store = ChromaStore(tmp_path / "chroma", dimension=3)
            assert store.collection == mock_collection

    def test_get_vector_store_faiss(self, monkeypatch) -> None:
        """Test get_vector_store returns FAISSStore by default."""
        monkeypatch.setattr("pdf_ingest.store.settings.vector_store", "faiss")
        store = get_vector_store("/tmp/test", 768)
        assert isinstance(store, FAISSStore)

    def test_get_vector_store_chroma(self, monkeypatch) -> None:
        """Test get_vector_store returns ChromaStore when configured."""
        monkeypatch.setattr("pdf_ingest.store.settings.vector_store", "chroma")
        with patch("pdf_ingest.store.chromadb"):
            store = get_vector_store("/tmp/test", 768)
            assert isinstance(store, ChromaStore)


# ===== Pipeline edge cases =====


class TestPipelineEdgeCases:
    """Tests for pipeline edge cases."""

    def test_ingest_directory_no_files(self, tmp_path) -> None:
        """Test ingesting empty directory."""
        pl = IngestionPipeline(vector_store_path=tmp_path / "vs")
        results = pl.ingest_directory(tmp_path)
        assert results == []

    def test_ingest_directory_with_unsupported(self, tmp_path) -> None:
        """Test ingesting directory with unsupported files."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("test")

        pl = IngestionPipeline(vector_store_path=tmp_path / "vs")
        results = pl.ingest_directory(tmp_path)
        assert results == []

    def test_ingest_batch_unsupported(self, tmp_path) -> None:
        """Test batch ingestion with unsupported file types."""
        pl = IngestionPipeline(vector_store_path=tmp_path / "vs")
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("test")

        result = pl.ingest_batch([unsupported])
        assert result.total_files == 1
        assert result.failed == 1
        assert len(result.errors) == 1


# ===== Chat edge cases =====


class TestChatEdgeCases:
    """Tests for chat edge cases."""

    def test_rag_chat_init_defaults(self, tmp_path, monkeypatch) -> None:
        """Test RAGChat initialization with defaults."""
        from pdf_ingest.chunk import Chunk
        from pdf_ingest.store import FAISSStore

        # Create a minimal store
        store = FAISSStore(tmp_path / "faiss", dimension=3)
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        chunks = [Chunk(id="c1", page=1, chunk=0, text="test")]
        store.add(vectors, chunks)
        store.save()

        chat = RAGChat(tmp_path / "faiss", dimension=3)
        assert chat.model == settings.llm_model
        assert chat.base_url == settings.ollama_url

    def test_rag_chat_clear_history(self, tmp_path) -> None:
        """Test clearing chat history."""
        from pdf_ingest.chunk import Chunk
        from pdf_ingest.store import FAISSStore

        store = FAISSStore(tmp_path / "faiss", dimension=3)
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        chunks = [Chunk(id="c1", page=1, chunk=0, text="test")]
        store.add(vectors, chunks)
        store.save()

        chat = RAGChat(tmp_path / "faiss", dimension=3)
        chat.history.append(chat.ChatMessage(role="user", content="test"))
        chat.clear_history()
        assert len(chat.history) == 0

    def test_chat_response_dataclass(self) -> None:
        """Test ChatResponse dataclass."""
        response = ChatResponse(
            content="Test answer",
            sources=[],
            model="test-model",
        )
        assert response.content == "Test answer"
        assert response.sources == []
        assert response.model == "test-model"


# ===== Search edge cases =====


class TestSearchEdgeCases:
    """Tests for search edge cases."""

    def test_hybrid_searcher_empty_store(self, tmp_path) -> None:
        """Test HybridSearcher with empty store."""
        searcher = HybridSearcher(tmp_path / "faiss", dimension=3)
        results = searcher.search("test")
        assert results == []

    def test_search_result_dataclass(self) -> None:
        """Test SearchResult dataclass."""
        result = SearchResult(
            id="test1",
            text="Test text",
            page=1,
            chunk=0,
            score=0.95,
            source="vector",
        )
        assert result.id == "test1"
        assert result.score == 0.95


# ===== Integration tests =====


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_pipeline_ingest_and_search(self, tmp_path, sample_pdf, monkeypatch) -> None:
        """Test complete ingest and search workflow."""
        monkeypatch.setattr("pdf_ingest.pipeline.settings.use_local_embeddings", True)

        pl = IngestionPipeline(vector_store_path=tmp_path / "vs", chunk_size=500, overlap=50)
        result = pl.ingest_pdf(sample_pdf)

        assert result.chunks
        assert len(result.vectors) == len(result.chunks)

        # Search
        kb = KnowledgeBase(tmp_path / "vs")
        from pdf_ingest.search import search

        search_results = search("alpha", tmp_path / "vs", kb.store.dimension, top_k=5)

        assert len(search_results) > 0


# ===== Chunk edge cases =====


class TestChunkEdgeCases:
    """Tests for chunking edge cases."""

    def test_chunk_pages_empty(self) -> None:
        """Test chunking empty pages."""
        chunks = chunk_pages([])
        assert chunks == []

    def test_chunk_pages_blank_text(self) -> None:
        """Test chunking pages with blank text."""
        chunks = chunk_pages([{"page": 1, "text": ""}])
        assert chunks == []

    def test_chunk_pages_whitespace_only(self) -> None:
        """Test chunking pages with whitespace only."""
        chunks = chunk_pages([{"page": 1, "text": "   \n\t  "}])
        assert chunks == []

    def test_split_text_empty(self) -> None:
        """Test splitting empty text."""
        chunks = split_text("", chunk_size=1000, overlap=150)
        assert chunks == []

    def test_split_text_smaller_than_chunk(self) -> None:
        """Test splitting text smaller than chunk size."""
        chunks = split_text("short text", chunk_size=1000, overlap=150)
        assert chunks == ["short text"]


# ===== Extract edge cases =====


class TestExtractEdgeCases:
    """Tests for extraction edge cases."""

    def test_recipe_extractor_empty_chunks(self) -> None:
        """Test recipe extractor with empty chunks."""
        from pdf_ingest.specialize import RecipeExtractor

        class MockChat:
            def _call_ollama(self, prompt, stream=False):
                return json.dumps({"title": "Test", "ingredients": [], "instructions": []})

        extractor = RecipeExtractor(MockChat())
        recipes = extractor.extract([])
        assert recipes == []

    def test_invoice_extractor_empty_chunks(self) -> None:
        """Test invoice extractor with empty chunks."""
        from pdf_ingest.specialize import InvoiceExtractor

        class MockChat:
            def _call_ollama(self, prompt, stream=False):
                return json.dumps({"vendor": "Test", "total": 0})

        extractor = InvoiceExtractor(MockChat())
        invoices = extractor.extract([])
        assert invoices == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
