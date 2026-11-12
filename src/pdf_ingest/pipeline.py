"""Unified ingestion pipeline: multi-source, multi-format, with domain extraction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
from tqdm import tqdm

from .chunk import chunk_pages, Chunk
from .config import settings
from .embed import embed_texts
from .extract import extract_pages, Page
from .extract_audio import extract_audio_text, extract_audio_segments, AudioSegment
from .extract_web import extract_web_content, extract_web_multiple, WebContent
from .extract_ocr import extract_pdf_ocr, extract_image_ocr, OCRResult
from .extract_office import extract_office, OfficeContent
from .store import save_vector_store, get_vector_store
from .chat import RAGChat
from .summarize import create_summarizer, Summary
from .specialize import (
    RecipeExtractor, Recipe,
    InvoiceExtractor, Invoice,
    MedicalExtractor, MedicalReport,
    ContractExtractor, Contract,
    StudyExtractor, StudyGuide,
)


@dataclass
class IngestionResult:
    source: str
    chunks: list[Chunk]
    vectors: np.ndarray
    vector_store_path: Path
    metadata: dict


class IngestionPipeline:
    def __init__(
        self,
        vector_store_path: Path | str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        strategy: str | None = None,
    ):
        self.vector_store_path = Path(vector_store_path or settings.vector_store_path)
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.strategy = strategy or settings.chunk_strategy

    def ingest_pdf(self, pdf_path: str | Path, use_ocr: bool = False) -> IngestionResult:
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

        store = save_vector_store(chunks, vectors, self.vector_store_path)

        return IngestionResult(
            source=pdf_path.name,
            chunks=chunks,
            vectors=vectors,
            vector_store_path=self.vector_store_path,
            metadata={"source": str(pdf_path), "pages": len(pages), "chunks": len(chunks)},
        )

    def ingest_audio(self, audio_path: str | Path) -> IngestionResult:
        text = extract_audio_text(audio_path)
        pages = [{"page": 1, "text": text}]
        chunks = chunk_pages(pages, self.chunk_size, self.overlap, self.strategy)
        texts = [c["text"] for c in chunks]
        vectors_list = embed_texts(texts)
        vectors = np.array(vectors_list, dtype=np.float32)
        store = save_vector_store(chunks, vectors, self.vector_store_path)

        return IngestionResult(
            source=Path(audio_path).name,
            chunks=chunks,
            vectors=vectors,
            vector_store_path=self.vector_store_path,
            metadata={"source": str(audio_path), "chunks": len(chunks)},
        )

    def ingest_web(self, urls: list[str]) -> list[IngestionResult]:
        results = []
        web_contents = extract_web_multiple(urls)
        for wc in web_contents:
            pages = [{"page": 1, "text": wc.text}]
            chunks = chunk_pages(pages, self.chunk_size, self.overlap, self.strategy)
            texts = [c["text"] for c in chunks]
            vectors_list = embed_texts(texts)
            vectors = np.array(vectors_list, dtype=np.float32)
            store = save_vector_store(chunks, vectors, self.vector_store_path)

            results.append(IngestionResult(
                source=wc.url,
                chunks=chunks,
                vectors=vectors,
                vector_store_path=self.vector_store_path,
                metadata={"source": wc.url, "title": wc.title, "chunks": len(chunks)},
            ))
        return results

    def ingest_office(self, file_path: str | Path) -> IngestionResult:
        contents = extract_office(file_path)
        all_chunks = []
        for content in contents:
            pages = [{"page": content.page, "text": content.text}]
            chunks = chunk_pages(pages, self.chunk_size, self.overlap, self.strategy)
            all_chunks.extend(chunks)

        texts = [c["text"] for c in all_chunks]
        vectors_list = embed_texts(texts)
        vectors = np.array(vectors_list, dtype=np.float32)
        store = save_vector_store(all_chunks, vectors, self.vector_store_path)

        return IngestionResult(
            source=Path(file_path).name,
            chunks=all_chunks,
            vectors=vectors,
            vector_store_path=self.vector_store_path,
            metadata={"source": str(file_path), "chunks": len(all_chunks)},
        )

    def ingest_directory(self, dir_path: str | Path, recursive: bool = True) -> list[IngestionResult]:
        dir_path = Path(dir_path)
        results = []
        pattern = "**/*" if recursive else "*"

        for file_path in dir_path.glob(pattern):
            if not file_path.is_file():
                continue
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
            except Exception as e:
                print(f"Failed to ingest {file_path}: {e}")

        return results


class KnowledgeBase:
    def __init__(self, vector_store_path: Path | str, dimension: int | None = None):
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
        self.chat = None
        self.summarizer = None

    def initialize_chat(self, **kwargs) -> RAGChat:
        self.chat = RAGChat(self.vector_store_path, self.store.dimension, **kwargs)
        return self.chat

    def initialize_summarizer(self) -> "Summarizer":
        if not self.chat:
            self.initialize_chat()
        self.summarizer = create_summarizer(self.store, self.chat)
        return self.summarizer

    def ask(self, query: str, **kwargs) -> "ChatResponse":
        if not self.chat:
            self.initialize_chat()
        return self.chat.chat(query, **kwargs)

    def ask_stream(self, query: str, **kwargs):
        if not self.chat:
            self.initialize_chat()
        return self.chat.chat_stream(query, **kwargs)

    def summarize_all(self, **kwargs) -> Summary:
        if not self.summarizer:
            self.initialize_summarizer()
        # Get all chunks from store
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        return self.summarizer.summarize_all(chunks, **kwargs)

    def generate_study_guide(self, format: str = "flashcards", **kwargs) -> dict:
        if not self.summarizer:
            self.initialize_summarizer()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        return self.summarizer.generate_study_guide(chunks, format, **kwargs)

    def extract_recipes(self) -> list[Recipe]:
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = RecipeExtractor(self.chat)
        return extractor.extract(chunks)

    def extract_invoices(self) -> list[Invoice]:
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = InvoiceExtractor(self.chat)
        return extractor.extract(chunks)

    def extract_medical(self) -> list[MedicalReport]:
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = MedicalExtractor(self.chat)
        return extractor.extract(chunks)

    def extract_contracts(self) -> list[Contract]:
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = ContractExtractor(self.chat)
        return extractor.extract(chunks)

    def extract_study_guides(self) -> list[StudyGuide]:
        if not self.chat:
            self.initialize_chat()
        all_results = self.store.search(np.zeros(self.store.dimension), self.store.count)
        from .chunk import Chunk
        chunks = [Chunk(id=m["id"], page=m["page"], chunk=m["chunk"], text=m["text"]) for _, m in all_results]
        extractor = StudyExtractor(self.chat)
        return extractor.extract(chunks)


def create_pipeline(**kwargs) -> IngestionPipeline:
    return IngestionPipeline(**kwargs)


def create_knowledge_base(vector_store_path: Path | str) -> KnowledgeBase:
    return KnowledgeBase(vector_store_path)