"""FastAPI backend for pdf-ingest."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pdf_ingest import (
    IngestionPipeline, KnowledgeBase, create_pipeline, create_knowledge_base,
    settings, Recipe, Invoice, MedicalReport, Contract, StudyGuide,
)

app = FastAPI(
    title="pdf-ingest API",
    description="Local-first RAG API for document ingestion, search, and chat",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pipeline: Optional[IngestionPipeline] = None
kb: Optional[KnowledgeBase] = None


def get_pipeline() -> IngestionPipeline:
    global pipeline
    if pipeline is None:
        pipeline = create_pipeline(vector_store_path=settings.vector_store_path)
    return pipeline


def get_kb() -> KnowledgeBase:
    global kb
    if kb is None:
        kb = create_knowledge_base(settings.vector_store_path)
        kb.initialize_chat(model=settings.llm_model)
    return kb


class IngestResponse(BaseModel):
    source: str
    chunks: int
    metadata: dict


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class ChatResponse(BaseModel):
    content: str
    sources: list[dict]
    model: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    id: str
    text: str
    page: int
    chunk: int
    score: float
    source: str


class SummarizeRequest(BaseModel):
    max_length: int = Field(default=500, ge=100, le=2000)
    style: str = Field(default="concise", pattern="^(concise|detailed|bullet)$")


class SummarizeResponse(BaseModel):
    summary: str
    key_points: list[str]
    chunks_used: list[str]


class ExtractRequest(BaseModel):
    type: str = Field(pattern="^(recipes|invoices|medical|contracts|study)$")
    format: str = Field(default="json", pattern="^(json|markdown)$")


@app.on_event("startup")
async def startup():
    get_pipeline()
    get_kb()


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/config")
async def get_config():
    return settings.model_dump()


@app.post("/ingest/file", response_model=list[IngestResponse])
async def ingest_file(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    use_ocr: bool = False,
):
    pl = get_pipeline()
    results = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            if suffix == ".pdf":
                result = pl.ingest_pdf(tmp_path, use_ocr=use_ocr)
            elif suffix in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
                result = pl.ingest_audio(tmp_path)
            elif suffix in (".docx", ".xlsx", ".pptx"):
                result = pl.ingest_office(tmp_path)
            else:
                raise HTTPException(400, f"Unsupported file type: {suffix}")

            results.append(IngestResponse(
                source=result.source,
                chunks=len(result.chunks),
                metadata=result.metadata,
            ))
        finally:
            tmp_path.unlink(missing_ok=True)

    return results


@app.post("/ingest/urls", response_model=list[IngestResponse])
async def ingest_urls(urls: list[str]):
    pl = get_pipeline()
    results = pl.ingest_web(urls)
    return [IngestResponse(source=r.source, chunks=len(r.chunks), metadata=r.metadata) for r in results]


@app.post("/ingest/directory", response_model=list[IngestResponse])
async def ingest_directory(path: str, recursive: bool = True):
    pl = get_pipeline()
    results = pl.ingest_directory(path, recursive=recursive)
    return [IngestResponse(source=r.source, chunks=len(r.chunks), metadata=r.metadata) for r in results]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    kb = get_kb()
    response = kb.ask(request.query, top_k=request.top_k)
    return ChatResponse(
        content=response.content,
        sources=[{"id": s.id, "text": s.text, "page": s.page, "score": s.score, "source": s.source} for s in response.sources],
        model=response.model,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    from fastapi.responses import StreamingResponse
    kb = get_kb()

    async def generate():
        sources = None
        for chunk in kb.ask_stream(request.query, top_k=request.top_k):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    kb = get_kb()
    from pdf_ingest import search as search_fn
    results = search_fn(request.query, settings.vector_store_path, kb.store.dimension, request.top_k, request.threshold)
    return [SearchResult(id=r.id, text=r.text, page=r.page, chunk=r.chunk, score=r.score, source=r.source) for r in results]


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    kb = get_kb()
    summary = kb.summarize_all(max_length=request.max_length, style=request.style)
    return SummarizeResponse(
        summary=summary.text,
        key_points=summary.key_points,
        chunks_used=summary.chunks_used,
    )


@app.post("/study")
async def study(format: str = "flashcards"):
    kb = get_kb()
    result = kb.generate_study_guide(format)
    return result


@app.post("/extract")
async def extract(request: ExtractRequest):
    kb = get_kb()
    items = getattr(kb, f"extract_{request.type}")()

    if request.format == "json":
        return [item.to_dict() for item in items]
    else:
        return "\n\n---\n\n".join(item.to_markdown() for item in items)


@app.get("/kb/stats")
async def kb_stats():
    kb = get_kb()
    return {
        "total_chunks": kb.store.count,
        "dimension": kb.store.dimension,
        "vector_store_path": str(settings.vector_store_path),
    }


@app.delete("/kb")
async def delete_kb():
    import shutil
    shutil.rmtree(settings.vector_store_path, ignore_errors=True)
    global pipeline, kb
    pipeline = None
    kb = None
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)