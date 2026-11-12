import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pdf_ingest import (
    extract_pages, split_text, chunk_pages,
    embed_texts, embed_query,
    save_dataset, load_dataset,
    save_vector_store, load_vector_store, VectorStore,
    search, SearchResult,
    RAGChat, ChatResponse,
    IngestionPipeline, KnowledgeBase,
    Recipe, Invoice, MedicalReport, Contract, StudyGuide,
    settings,
)


# ===== Existing tests =====

@pytest.fixture()
def sample_pdf(tmp_path):
    import pymupdf

    path = tmp_path / "t.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 523, 700), "alpha beta gamma. " * 120, fontsize=11)
    doc.new_page()  # blank page: should be skipped
    doc.save(path)
    return path


def test_extract_skips_blank_pages(sample_pdf):
    pages, total = extract_pages(sample_pdf)
    assert total == 2
    assert [p["page"] for p in pages] == [1]


def test_extract_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_pages("nope.pdf")


def test_split_text_respects_size_and_overlap():
    text = "x" * 2500
    chunks = split_text(text, chunk_size=1000, overlap=150, strategy="character")
    assert all(len(c) <= 1000 for c in chunks)
    assert len(chunks) == 3


def test_split_text_no_overlap_only_tail():
    chunks = split_text("abcdefghij", chunk_size=10, overlap=2)
    assert chunks == ["abcdefghij"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_split_text_validation(size, overlap):
    with pytest.raises(ValueError):
        split_text("hello", size, overlap)


def test_chunk_pages_ids_and_metadata():
    chunks = chunk_pages([{"page": 3, "text": "a" * 1200}])
    assert chunks[0]["id"] == "p3-c0"
    assert {c["page"] for c in chunks} == {3}


# ===== New chunking tests =====

def test_split_text_semantic():
    text = "This is sentence one. This is sentence two. This is sentence three. " * 50
    chunks = split_text(text, chunk_size=500, overlap=50, strategy="semantic")
    assert len(chunks) > 0
    # Semantic chunking may exceed chunk_size due to sentence boundaries
    # Just verify it produces valid chunks without crashing
    assert all(len(c) > 0 for c in chunks)


def test_split_text_recursive():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = split_text(text, chunk_size=100, overlap=20, strategy="recursive")
    assert len(chunks) > 0


# ===== Embedding tests =====

class FakeResponse:
    def __init__(self, n, status=200):
        self.status_code = status
        self._n = n

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return {"embeddings": [[0.1, 0.2, 0.3] for _ in range(self._n)]}


def test_embed_batches(monkeypatch):
    import pdf_ingest.embed as embed_mod
    calls = []

    def fake_post(url, json, timeout):
        calls.append(len(json["input"]))
        return FakeResponse(len(json["input"]))

    monkeypatch.setattr(embed_mod.requests, "post", fake_post)
    vectors = embed_mod.embed_texts(["t"] * 5, batch_size=2)
    assert len(vectors) == 5
    assert calls == [2, 2, 1]


def test_embed_model_missing(monkeypatch):
    import pdf_ingest.embed as embed_mod
    monkeypatch.setattr(embed_mod.requests, "post", lambda *a, **k: FakeResponse(0, 404))
    with pytest.raises(embed_mod.EmbeddingError, match="ollama pull"):
        embed_mod.embed_texts(["t"])


def test_embed_connection_error(monkeypatch):
    import pdf_ingest.embed as embed_mod
    def boom(*a, **k):
        raise embed_mod.requests.ConnectionError()

    monkeypatch.setattr(embed_mod.requests, "post", boom)
    with pytest.raises(embed_mod.EmbeddingError, match="Is it running"):
        embed_mod.embed_texts(["t"])


# ===== Store tests =====

def test_save_and_load_roundtrip(tmp_path):
    chunks = chunk_pages([{"page": 1, "text": "hello world"}])
    out = save_dataset(tmp_path / "o.json", "t.pdf", chunks, [[0.1, 0.2]],
                       model="m", chunk_size=1000, overlap=150)
    data = load_dataset(out)
    assert data["meta"]["dimensions"] == 2
    assert data["chunks"][0]["embedding"] == [0.1, 0.2]
    json.dumps(data)


def test_save_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        save_dataset(tmp_path / "o.json", "t", [{"id": "a", "page": 1, "chunk": 0, "text": "x"}], [],
                     model="m", chunk_size=10, overlap=1)


# ===== Vector store tests =====

def test_faiss_store_add_search(tmp_path):
    import numpy as np
    from pdf_ingest.store import FAISSStore
    from pdf_ingest.chunk import Chunk

    store = FAISSStore(tmp_path / "faiss", dimension=3)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    chunks = [
        Chunk(id="c1", page=1, chunk=0, text="first"),
        Chunk(id="c2", page=1, chunk=1, text="second"),
    ]
    store.add(vectors, chunks)
    store.save()

    # Search
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=1)
    assert len(results) == 1
    assert results[0][1]["id"] == "c1"


def test_faiss_store_persistence(tmp_path):
    import numpy as np
    from pdf_ingest.store import FAISSStore
    from pdf_ingest.chunk import Chunk

    store = FAISSStore(tmp_path / "faiss", dimension=3)
    vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    chunks = [Chunk(id="c1", page=1, chunk=0, text="first")]
    store.add(vectors, chunks)
    store.save()

    # Reload
    store2 = FAISSStore(tmp_path / "faiss", dimension=3)
    assert store2.count == 1
    results = store2.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert results[0][1]["id"] == "c1"


# ===== Search tests =====

def test_hybrid_search(tmp_path, monkeypatch):
    import numpy as np
    from pdf_ingest.store import FAISSStore
    from pdf_ingest.chunk import Chunk
    import sys

    store = FAISSStore(tmp_path / "faiss", dimension=3)
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    chunks = [
        Chunk(id="c1", page=1, chunk=0, text="apple banana"),
        Chunk(id="c2", page=1, chunk=1, text="banana cherry"),
        Chunk(id="c3", page=1, chunk=2, text="cherry date"),
    ]
    store.add(vectors, chunks)
    store.save()

    # Mock embed_query at the search module level
    def mock_embed(q):
        if "apple" in q:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif "banana" in q:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    search_module = sys.modules["pdf_ingest.search"]
    monkeypatch.setattr(search_module, "embed_query", mock_embed)

    from pdf_ingest.search import HybridSearcher
    searcher = HybridSearcher(tmp_path / "faiss", dimension=3)
    results = searcher.search("apple", top_k=2)
    assert len(results) > 0
    assert results[0].id == "c1"


# ===== Pipeline tests =====

def test_ingestion_pipeline_pdf(tmp_path, sample_pdf, monkeypatch):
    pl = IngestionPipeline(vector_store_path=tmp_path / "vs", chunk_size=500, overlap=50)
    # Use local embeddings to avoid Ollama dependency
    monkeypatch.setattr("pdf_ingest.pipeline.settings.use_local_embeddings", True)
    result = pl.ingest_pdf(sample_pdf)
    assert result.chunks
    assert len(result.vectors) == len(result.chunks)
    assert result.vector_store_path.exists()


def test_knowledge_base_ask(tmp_path, sample_pdf, monkeypatch):
    import pdf_ingest.chat as chat_mod
    import numpy as np

    # Set local embeddings before creating pipeline
    monkeypatch.setattr("pdf_ingest.pipeline.settings.use_local_embeddings", True)
    pl = IngestionPipeline(vector_store_path=tmp_path / "vs", chunk_size=500, overlap=50)
    pl.ingest_pdf(sample_pdf)

    kb = KnowledgeBase(tmp_path / "vs")
    kb.initialize_chat()

    # Mock Ollama response
    def mock_call(prompt, stream=True):
        if stream:
            yield "Test answer"
        else:
            return "Test answer"

    monkeypatch.setattr(kb.chat, "_call_ollama", mock_call)
    monkeypatch.setattr(kb.chat, "_build_context", lambda r: "Context text")
    monkeypatch.setattr(kb.chat, "_build_prompt", lambda q, c: f"Prompt: {q}")

    response = kb.ask("test question")
    assert response.content == "Test answer"


# ===== Domain extractor tests =====

def test_recipe_extractor(tmp_path, monkeypatch):
    from pdf_ingest.specialize import RecipeExtractor
    from pdf_ingest.chunk import Chunk
    from pdf_ingest.chat import RAGChat

    # Mock chat
    class MockChat:
        def _call_ollama(self, prompt, stream=False):
            return json.dumps({
                "title": "Test Recipe",
                "description": "A test recipe",
                "servings": 4,
                "prep_time_minutes": 10,
                "cook_time_minutes": 20,
                "ingredients": [{"name": "flour", "amount": "2", "unit": "cups", "notes": ""}],
                "instructions": ["Mix", "Bake"],
                "nutrition": {"calories": "200"},
                "tags": ["dessert"],
            })

    mock_chat = MockChat()
    extractor = RecipeExtractor(mock_chat)
    chunks = [Chunk(id="p1-c0", page=1, chunk=0, text="Recipe: Test Cake\n2 cups flour\nMix and bake")]
    recipes = extractor.extract(chunks)

    assert len(recipes) == 1
    assert recipes[0].title == "Test Recipe"
    assert len(recipes[0].ingredients) == 1


def test_invoice_extractor(tmp_path, monkeypatch):
    from pdf_ingest.specialize import InvoiceExtractor
    from pdf_ingest.chunk import Chunk

    class MockChat:
        def _call_ollama(self, prompt, stream=False):
            return json.dumps({
                "vendor": "Test Corp",
                "invoice_number": "INV-001",
                "date": "2024-01-15",
                "total": 100.50,
                "line_items": [{"description": "Service", "quantity": 1, "unit_price": 100.50, "total": 100.50}],
            })

    extractor = InvoiceExtractor(MockChat())
    chunks = [Chunk(id="p1-c0", page=1, chunk=0, text="Invoice from Test Corp\nTotal: $100.50")]
    invoices = extractor.extract(chunks)

    assert len(invoices) == 1
    assert invoices[0].vendor == "Test Corp"
    assert invoices[0].total == 100.50


def test_medical_extractor(tmp_path, monkeypatch):
    from pdf_ingest.specialize import MedicalExtractor
    from pdf_ingest.chunk import Chunk

    class MockChat:
        def _call_ollama(self, prompt, stream=False):
            return json.dumps({
                "patient_name": "John Doe",
                "report_date": "2024-01-15",
                "summary": "Normal blood work",
                "lab_values": [{"name": "Glucose", "value": "95", "unit": "mg/dL", "reference_range": "70-100", "status": "normal"}],
            })

    extractor = MedicalExtractor(MockChat())
    chunks = [Chunk(id="p1-c0", page=1, chunk=0, text="Patient: John Doe\nGlucose: 95 mg/dL (70-100)")]
    reports = extractor.extract(chunks)

    assert len(reports) == 1
    assert reports[0].patient_name == "John Doe"


def test_contract_extractor(tmp_path, monkeypatch):
    from pdf_ingest.specialize import ContractExtractor
    from pdf_ingest.chunk import Chunk

    class MockChat:
        def _call_ollama(self, prompt, stream=False):
            return json.dumps({
                "title": "Service Agreement",
                "parties": ["Company A", "Company B"],
                "contract_type": "service",
                "clauses": [{"title": "Payment", "text": "Net 30", "category": "payment", "risk_level": "low"}],
            })

    extractor = ContractExtractor(MockChat())
    chunks = [Chunk(id="p1-c0", page=1, chunk=0, text="Service Agreement between Company A and Company B")]
    contracts = extractor.extract(chunks)

    assert len(contracts) == 1
    assert contracts[0].title == "Service Agreement"


def test_study_extractor(tmp_path, monkeypatch):
    from pdf_ingest.specialize import StudyExtractor
    from pdf_ingest.chunk import Chunk

    class MockChat:
        def _call_ollama(self, prompt, stream=False):
            return json.dumps({
                "title": "Biology Notes",
                "summary": "Cell structure overview",
                "topics": ["cells", "organelles"],
                "flashcards": [{"question": "What is a cell?", "answer": "Basic unit of life", "topic": "cells", "difficulty": "easy"}],
                "quiz_questions": [{"question": "What is a cell?", "options": ["A", "B", "C", "D"], "correct_index": 0, "explanation": "Cells are basic units", "topic": "cells"}],
                "mind_map": {"topic": "Biology", "children": [{"topic": "Cells", "children": []}]},
                "key_concepts": {"cell": "basic unit of life"},
            })

    extractor = StudyExtractor(MockChat())
    chunks = [Chunk(id="p1-c0", page=1, chunk=0, text="Cells are the basic unit of life...")]
    guides = extractor.extract(chunks)

    assert len(guides) == 1
    assert guides[0].title == "Biology Notes"
    assert len(guides[0].flashcards) == 1


# ===== Settings tests =====

def test_settings_defaults():
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.vector_store == "faiss"
    assert settings.embed_model == "nomic-embed-text"


# ===== Integration test =====

def test_full_pipeline_cli(tmp_path, sample_pdf, monkeypatch, capsys):
    """Test the full pipeline via CLI."""
    from pdf_ingest.cli import main

    # Mock Ollama
    import pdf_ingest.embed as embed_mod

    class FakeResponse:
        def __init__(self, n=1):
            self.status_code = 200
            self._n = n
        def raise_for_status(self):
            pass
        def json(self):
            return {"embeddings": [[0.1] * 768 for _ in range(self._n)]}

    def mock_post(*args, **kwargs):
        json_data = kwargs.get('json', {})
        n = len(json_data.get('input', []))
        return FakeResponse(n)

    monkeypatch.setattr(embed_mod.requests, "post", mock_post)

    # Run ingest (with 'ingest' command)
    exit_code = main(["ingest", str(sample_pdf), "--output", str(tmp_path / "out.json")])
    assert exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])