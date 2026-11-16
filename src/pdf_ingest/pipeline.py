"""Unified ingestion pipeline: multi-source, multi-format, with domain extraction.

This module provides:
- IngestionPipeline: High-level API for ingesting various document types
- KnowledgeBase: Query interface for chat, search, summarization, extraction
- Batch ingestion with progress tracking
- Domain-specific extraction (recipes, invoices, medical, contracts, study)

Example:
    pipeline = IngestionPipeline(vector_store_path="./data/kb")
    pipeline.ingest_pdf("document.pdf")
    pipeline.ingest_audio("meeting.wav")
    pipeline.ingest_web(["https://example.com/article"])

    kb = KnowledgeBase("./data/kb")
    response = kb.ask("What are the key findings?")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .chat import RAGChat
from .chunk import Chunk, chunk_pages
from .config import settings
from .embed import embed_texts
from .extract import extract_pages
from .extract_audio import extract_audio_text
from .extract_ocr import extract_pdf_ocr
from .extract_office import extract_office
from .extract_web import extract_web_multiple
from .progress import progress_bar
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
from .store import get_vector_store, save_vector_store
from .summarize import Summary, create_summarizer

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class IngestionResult:
    """Result of a single document ingestion.

    Attributes:
        source: Source identifier (filename, URL, etc.)
        chunks: List of text chunks created
        vectors: Embedding vectors for chunks
        vector_store_path: Path to vector store
        metadata: Additional metadata (page count, chunk count, etc.)
    """
    source: str
    chunks: list[Chunk]
    vectors: np.ndarray
    vector_store_path: Path
    metadata: dict


@dataclass
class BatchIngestionResult:
    """Result of batch ingestion operation.

    Attributes:
        total_files: Total files processed
        successful: Number of successfully ingested files
        failed: Number of failed files
        results: List of successful IngestionResults
        errors: List of error dicts with file and error message
    """
    total_files: int
    successful: int
    failed: int
    results: list[IngestionResult]
    errors: list[dict]


class IngestionPipeline:
    """Unified pipeline for ingesting documents from multiple sources.

    Supports: PDF, audio, web pages, Office documents (DOCX, XLSX, PPTX)

    Example:
        pipeline = IngestionPipeline("./data/kb")
        result = pipeline.ingest_pdf("report.pdf")
        print(f"Ingested {result.metadata['chunks']} chunks")
    """

    def __init__(
        self,
        vector_store_path: Path | str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        strategy: str | None = None,
    ) -> None:
        """Initialize ingestion pipeline.

        Args:
            vector_store_path: Directory for vector store (default: settings.vector_store_path)
            chunk_size: Chunk size in characters (default: settings.chunk_size)
            overlap: Chunk overlap in characters (default: settings.chunk_overlap)
            strategy: Chunking strategy - "character", "semantic", "recursive" (default: settings.chunk_strategy)
        """
        self.vector_store_path = Path(vector_store_path or settings.vector_store_path)
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.strategy = strategy or settings.chunk_strategy

    def ingest_pdf(self, pdf_path: str | Path, use_ocr: bool = False) -> IngestionResult:
        """Ingest a PDF document.

        Args:
            pdf_path: Path to PDF file
            use_ocr: Use OCR for scanned PDFs (default: False)

        Returns:
            IngestionResult with chunks, vectors, and metadata
        """
        pdf_path = Path(pdf_path)
        if use_ocr:
            ocr_results = extract_pdf_ocr(pdf_path)
            pages = [{"page": r.page, "text": r.text} for r in ocr_results]
        else:
            pages, _ = extract_pages(pdf_path)

        chunks = chunk_pages(pages, self.chunk_size, self.overlap, self.strategy)
        texts = [c["text"] for c in chunks]
        vectors_list = embed_texts(texts)
        vectors = np.array(vectors_list, dtype=np.float32)

        save_vector_store(chunks, vectors, self.vector_store_path)

        return IngestionResult(
            source=pdf_path.name,
            chunks=chunks,
            vectors=vectors,
            vector_store_path=self.vector_store_path,
            metadata={"source": str(pdf_path), "pages": len(pages), "chunks": len(chunks)},
        )

    def ingest_audio(self, audio_path: str | Path) -> IngestionResult:
        """Ingest an audio file (transcribes with faster-whisper).

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, flac, ogg)

        Returns:
            IngestionResult with transcribed chunks
        """
        text = extract_audio_text(audio_path)
        pages = [{"page": 1, "text": text}]
        chunks = chunk_pages(pages, self.chunk_size, self.overlap, self.strategy)
        texts = [c["text"] for c in chunks]
        vectors_list = embed_texts(texts)
        vectors = np.array(vectors_list, dtype=np.float32)
        save_vector_store(chunks, vectors, self.vector_store_path)

        return IngestionResult(
            source=Path(audio_path).name,
            chunks=chunks,
            vectors=vectors,
            vector_store_path=self.vector_store_path,
            metadata={"source": str(audio_path), "chunks": len(chunks)},
        )

    def ingest_web(self, urls: list[str]) -> list[IngestionResult]:
        """Ingest web pages from URLs.

        Args:
            urls: List of URLs to fetch and ingest

        Returns:
            List of IngestionResults for each URL
        """
        results = []
        web_contents = extract_web_multiple(urls)
        for wc in web_contents:
            pages = [{"page": 1, "text": wc.text}]
            chunks = chunk_pages(pages, self.chunk_size, self.overlap, self.strategy)
            texts = [c["text"] for c in chunks]
            vectors_list = embed_texts(texts)
            vectors = np.array(vectors_list, dtype=np.float32)
            save_vector_store(chunks, vectors, self.vector_store_path)

            results.append(IngestionResult(
                source=wc.url,
                chunks=chunks,
                vectors=vectors,
                vector_store_path=self.vector_store_path,
                metadata={"source": wc.url, "title": wc.title, "chunks": len(chunks)},
            ))
        return results

    def ingest_office(self, file_path: str | Path) -> IngestionResult:
        """Ingest Office document (DOCX, XLSX, PPTX).

        Args:
            file_path: Path to Office file

        Returns:
            IngestionResult with extracted chunks
        """
        contents = extract_office(file_path)
        all_chunks = []
        for content in contents:
            pages = [{"page": content.page, "text": content.text}]
            chunks = chunk_pages(pages, self.chunk_size, self.overlap, self.strategy)
            all_chunks.extend(chunks)

        texts = [c["text"] for c in all_chunks]
        vectors_list = embed_texts(texts)
        vectors = np.array(vectors_list, dtype=np.float32)
        save_vector_store(all_chunks, vectors, self.vector_store_path)

        return IngestionResult(
            source=Path(file_path).name,
            chunks=all_chunks,
            vectors=vectors,
            vector_store_path=self.vector_store_path,
            metadata={"source": str(file_path), "chunks": len(all_chunks)},
        )

    def ingest_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[IngestionResult]:
        """Ingest all supported files in a directory.

        Args:
            dir_path: Directory path
            recursive: Recursively scan subdirectories (default: True)
            progress_callback: Optional callback(filename, current, total)

        Returns:
            List of IngestionResults for each file
        """
        dir_path = Path(dir_path)
        results = []
        pattern = "**/*" if recursive else "*"

        # Collect all supported files first
        supported_extensions = {
            ".pdf", ".mp3", ".wav", ".m4a", ".flac", ".ogg",
            ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
        }
        files = [
            f for f in dir_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(file_path.name, i + 1, len(files))

            suffix = file_path.suffix.lower()
            try:
                if suffix == ".pdf":
                    results.append(self.ingest_pdf(file_path))
                elif suffix in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
                    results.append(self.ingest_audio(file_path))
                elif suffix in (".docx", ".xlsx", ".pptx"):
                    results.append(self.ingest_office(file_path))
                elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                    # Could add image OCR here
                    pass
            except Exception:
                pass

        return results

    def ingest_batch(
        self,
        file_paths: list[str | Path],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> BatchIngestionResult:
        """Ingest multiple files with progress tracking.

        Args:
            file_paths: List of file paths to ingest
            progress_callback: Optional callback(filename, current, total)

        Returns:
            BatchIngestionResult with counts and errors
        """
        results: list[IngestionResult] = []
        errors: list[dict] = []
        successful = 0
        failed = 0

        for i, file_path in enumerate(file_paths):
            file_path = Path(file_path)
            if progress_callback:
                progress_callback(file_path.name, i + 1, len(file_paths))

            suffix = file_path.suffix.lower()
            try:
                if suffix == ".pdf":
                    results.append(self.ingest_pdf(file_path))
                    successful += 1
                elif suffix in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
                    results.append(self.ingest_audio(file_path))
                    successful += 1
                elif suffix in (".docx", ".xlsx", ".pptx"):
                    results.append(self.ingest_office(file_path))
                    successful += 1
                elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                    errors.append({"file": str(file_path), "error": "Image OCR not yet supported"})
                    failed += 1
                else:
                    errors.append({"file": str(file_path), "error": f"Unsupported file type: {suffix}"})
                    failed += 1
            except Exception as e:
                errors.append({"file": str(file_path), "error": str(e)})
                failed += 1

        return BatchIngestionResult(
            total_files=len(file_paths),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
        )

    def ingest_batch_with_progress(
        self,
        file_paths: list[str | Path],
        desc: str = "Batch ingesting",
    ) -> BatchIngestionResult:
        """Ingest multiple files with a progress bar.

        Args:
            file_paths: List of file paths
            desc: Progress bar description

        Returns:
            BatchIngestionResult with counts and errors
        """
        with progress_bar(total=len(file_paths), desc=desc, unit="file") as pbar:
            def callback(name: str, current: int, total: int) -> None:
                pbar.set_description(f"{desc} {name}")

            return self.ingest_batch(file_paths, progress_callback=callback)


class KnowledgeBase:
    """Query interface for a populated vector store.

    Provides high-level methods for chat, search, summarization, and extraction.

    Example:
        kb = KnowledgeBase("./data/kb")
        response = kb.ask("Summarize the document")
        summary = kb.summarize_all(style="bullet")
        recipes = kb.extract_recipes()
    """

    def __init__(self, vector_store_path: Path | str, dimension: int | None = None) -> None:
        """Initialize knowledge base.

        Args:
            vector_store_path: Path to existing vector store
            dimension: Embedding dimension (auto-detected if None)
        """
        self.vector_store_path = Path(vector_store_path)
        # Try to detect dimension from existing store, default to 768
        if dimension is None:
            try:
                import faiss
                index_path = self.vector_store_path / "faiss.index"
                if index_path.exists():
                    index = faiss.read_index(str(index_path))
                    dimension = index.d
            except Exception:
                pass
        if dimension is None:
            dimension = 768
        self.store = get_vector_store(self.vector_store_path, dimension)
        self.chat: RAGChat | None = None
        self.summarizer: Any = None

    def initialize_chat(self, **kwargs) -> RAGChat:
        """Initialize RAG chat interface.

        Args:
            **kwargs: Passed to RAGChat constructor

        Returns:
            Initialized RAGChat instance
        """
        self.chat = RAGChat(self.vector_store_path, self.store.dimension, **kwargs)
        return self.chat

    def initialize_summarizer(self) -> Any:
        """Initialize summarizer (requires chat).

        Returns:
            Initialized Summarizer instance
        """
        if not self.chat:
            self.initialize_chat()
        self.summarizer = create_summarizer(self.store, self.chat)
        return self.summarizer

    def ask(self, query: str, **kwargs) -> Any:
        """Ask a question (non-streaming).

        Args:
            query: Question to ask
            **kwargs: Passed to chat (top_k, etc.)

        Returns:
            ChatResponse with answer and sources
        """
        if not self.chat:
            self.initialize_chat()
        return self.chat.chat(query, **kwargs)

    def ask_stream(self, query: str, **kwargs) -> Any:
        """Ask a question (streaming).

        Args:
            query: Question to ask
            **kwargs: Passed to chat (top_k, etc.)

        Returns:
            Generator of response chunks
        """
        if not self.chat:
            self.initialize_chat()
        return self.chat.chat_stream(query, **kwargs)

    def summarize_all(self, **kwargs) -> Summary:
        """Generate summary of all documents.

        Args:
            **kwargs: Passed to summarizer (max_length, style)

        Returns:
            Summary with text, key_points, and chunks_used
        """
        if not self.summarizer:
            self.initialize_summarizer()
        # Get all chunks from store
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        return self.summarizer.summarize_all(chunks, **kwargs)

    def generate_study_guide(self, format: str = "flashcards", **kwargs) -> dict:
        """Generate study materials.

        Args:
            format: Output format - "flashcards", "mindmap", "quiz", "all"
            **kwargs: Passed to summarizer

        Returns:
            Study guide dictionary
        """
        if not self.summarizer:
            self.initialize_summarizer()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        return self.summarizer.generate_study_guide(chunks, format, **kwargs)

    def extract_recipes(self) -> list[Recipe]:
        """Extract recipes from knowledge base.

        Returns:
            List of Recipe objects
        """
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = RecipeExtractor(self.chat)
        return extractor.extract(chunks)

    def extract_invoices(self) -> list[Invoice]:
        """Extract invoices from knowledge base.

        Returns:
            List of Invoice objects
        """
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = InvoiceExtractor(self.chat)
        return extractor.extract(chunks)

    def extract_medical(self) -> list[MedicalReport]:
        """Extract medical reports from knowledge base.

        Returns:
            List of MedicalReport objects
        """
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = MedicalExtractor(self.chat)
        return extractor.extract(chunks)

    def extract_contracts(self) -> list[Contract]:
        """Extract contracts from knowledge base.

        Returns:
            List of Contract objects
        """
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = ContractExtractor(self.chat)
        return extractor.extract(chunks)

    def extract_study_guides(self) -> list[StudyGuide]:
        """Extract study guides from knowledge base.

        Returns:
            List of StudyGuide objects
        """
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = StudyExtractor(self.chat)
        return extractor.extract(chunks)


def create_pipeline(**kwargs) -> IngestionPipeline:
    """Factory function to create IngestionPipeline.

    Args:
        **kwargs: Passed to IngestionPipeline constructor

    Returns:
        Configured IngestionPipeline instance
    """
    return IngestionPipeline(**kwargs)


def create_knowledge_base(vector_store_path: Path | str) -> KnowledgeBase:
    """Factory function to create KnowledgeBase.

    Args:
        vector_store_path: Path to vector store

    Returns:
        Configured KnowledgeBase instance
    """
    return KnowledgeBase(vector_store_path)