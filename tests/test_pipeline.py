import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pdf_ingest import embed as embed_mod  # noqa: E402
from pdf_ingest.chunk import chunk_pages, split_text  # noqa: E402
from pdf_ingest.extract import extract_pages  # noqa: E402
from pdf_ingest.store import load_dataset, save_dataset  # noqa: E402


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
    chunks = split_text(text, chunk_size=1000, overlap=150)
    assert all(len(c) <= 1000 for c in chunks)
    assert len(chunks) == 3  # windows start at 0, 850, 1700; last reaches the end


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
    calls = []

    def fake_post(url, json, timeout):
        calls.append(len(json["input"]))
        return FakeResponse(len(json["input"]))

    monkeypatch.setattr(embed_mod.requests, "post", fake_post)
    vectors = embed_mod.embed_texts(["t"] * 5, batch_size=2)
    assert len(vectors) == 5
    assert calls == [2, 2, 1]


def test_embed_model_missing(monkeypatch):
    monkeypatch.setattr(embed_mod.requests, "post", lambda *a, **k: FakeResponse(0, 404))
    with pytest.raises(embed_mod.EmbeddingError, match="ollama pull"):
        embed_mod.embed_texts(["t"])


def test_embed_connection_error(monkeypatch):
    def boom(*a, **k):
        raise embed_mod.requests.ConnectionError()

    monkeypatch.setattr(embed_mod.requests, "post", boom)
    with pytest.raises(embed_mod.EmbeddingError, match="Is it running"):
        embed_mod.embed_texts(["t"])


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
