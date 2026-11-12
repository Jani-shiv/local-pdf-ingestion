<div align="center">

<img src="docs/images/00-cover.png" alt="Local PDF ingestion pipeline cover" width="100%">

# 📄 pdf-ingest: Local-First RAG Pipeline

### Build a private, local-first knowledge base from PDFs, audio, web pages, and documents — then chat with it, search it, summarize it, and extract structured data. 100% offline, open source, free.

<p>
  <a href="https://github.com/Jani-shiv/local-pdf-ingestion/stargazers"><img src="https://img.shields.io/github/stars/Jani-shiv/local-pdf-ingestion?style=for-the-badge&logo=github&color=yellow" alt="GitHub stars"></a>
  <a href="https://github.com/Jani-shiv/local-pdf-ingestion/network/members"><img src="https://img.shields.io/github/forks/Jani-shiv/local-pdf-ingestion?style=for-the-badge&logo=github&color=blue" alt="GitHub forks"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready">
</p>

</div>

> **NotebookLM alternative** that runs entirely on your machine. Ingest PDFs, audio recordings, web articles, Word/Excel/PowerPoint files → search with hybrid vector+keyword search → chat with citations → generate summaries, study guides, flashcards, and extract structured data (recipes, invoices, medical reports, contracts).

---

## 🎯 Why pdf-ingest?

| Feature | NotebookLM | pdf-ingest |
|---------|-----------|------------|
| **Privacy** | Google Cloud | 🔒 100% Local |
| **Cost** | Free tier limits | 💰 Free forever |
| **File Types** | PDF, Drive docs | 📄 PDF, Audio, Web, DOCX, XLSX, PPTX, Images |
| **Offline** | ❌ No | ✅ Yes |
| **Custom Extractors** | ❌ No | ✅ Recipes, Invoices, Medical, Contracts, Study |
| **Export** | Limited | ✅ JSON, Markdown, Anki CSV, CSV |
| **API** | ❌ No | ✅ FastAPI + WebSocket |
| **UI** | Web only | ✅ Streamlit + CLI + API |

---

## ✨ Features

### 🔄 Multi-Source Ingestion
- **PDFs** — Text extraction (PyMuPDF) + OCR (Tesseract/PaddleOCR) for scanned docs
- **Audio** — Local transcription with faster-whisper (no API keys)
- **Web** — Clean article extraction via trafilatura
- **Office** — DOCX, XLSX, PPTX text extraction
- **Images** — OCR on PNG/JPG/TIFF

### 🧠 Smart Processing
- **Semantic Chunking** — Sentence-aware with overlap (not naive character splitting)
- **Recursive Chunking** — Paragraph → sentence → word fallback
- **Local Embeddings** — Ollama (nomic-embed-text) + SentenceTransformers fallback
- **Vector Stores** — FAISS (lightweight) or ChromaDB (persistent)

### 🔍 Hybrid Search
- **Vector Similarity** — Cosine similarity via FAISS/Chroma
- **Keyword (BM25)** — Rank-BM25 for exact term matching
- **Hybrid Fusion** — Best of both worlds

### 💬 RAG Chat with Citations
- **Streaming Responses** — Token-by-token via Ollama
- **Source Citations** — Every answer cites `[chunk_id]` with page numbers
- **Conversation History** — Multi-turn context
- **Configurable** — Model, temperature, top-k, threshold

### 📝 Summarization & Study Tools
- **Map-Reduce Summaries** — Handles arbitrarily long documents
- **Key Points Extraction** — Bullet-point takeaways
- **Study Guides** — Flashcards (Anki export), Mind Maps, Quizzes
- **Multiple Formats** — JSON, Markdown, CSV

### 🎯 Domain-Specific Extractors
| Extractor | Output | Use Case |
|-----------|--------|----------|
| **Recipes** | Structured ingredients, steps, nutrition | Cookbooks, food blogs |
| **Invoices** | Vendor, line items, totals, categories | Expense tracking, bookkeeping |
| **Medical** | Lab values with flags, summaries, medications | Patient reports, lab results |
| **Contracts** | Clauses, risk flags, obligation calendar | Freelance, legal review |
| **Study** | Flashcards, mind maps, quizzes | Students, certifications |

### 🖥️ Multiple Interfaces
- **CLI** — `pdf-ingest ingest/chat/search/summarize/study/extract`
- **Streamlit Web UI** — Drag-drop, chat, search, visualize
- **FastAPI** — REST + WebSocket for integrations
- **Python Library** — `from pdf_ingest import ...`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Embedding model: `ollama pull nomic-embed-text`
- LLM model: `ollama pull llama3.1:8b` (or `phi3:mini`, `mistral:7b`)

### Installation

```bash
# Clone and setup
git clone https://github.com/Jani-shiv/local-pdf-ingestion.git
cd local-pdf-ingestion

# Using uv (recommended, fast)
uv sync

# Or traditional venv
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### CLI Usage

```bash
# Ingest a PDF
pdf-ingest ingest document.pdf

# Ingest multiple files + directory
pdf-ingest ingest file1.pdf file2.docx ./documents --recursive

# Ingest from URLs
pdf-ingest ingest --urls "https://example.com/article https://blog.post"

# Chat with your knowledge base
pdf-ingest chat "Summarize the key findings"

# Search
pdf-ingest search "machine learning"

# Summarize everything
pdf-ingest summarize --style detailed

# Generate study materials
pdf-ingest study --format flashcards --anki  # Exports Anki CSV

# Extract structured data
pdf-ingest extract recipes --format markdown
pdf-ingest extract invoices --format csv
pdf-ingest extract medical --format json
```

### Streamlit Web UI

```bash
streamlit run web/app.py
```
Then open http://localhost:8501

### Docker (Recommended for Production)

```bash
# Start everything (Ollama + pdf-ingest + optional API)
docker-compose up -d

# Access:
# - Streamlit UI: http://localhost:8501
# - FastAPI: http://localhost:8000/docs
# - Ollama: http://localhost:11434
```

---

## 📖 Detailed Usage

### Configuration

Create a `.env` file or set environment variables:

```bash
# Ollama
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=llama3.1:8b

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
CHUNK_STRATEGY=semantic  # semantic, recursive, character

# Vector Store
VECTOR_STORE=faiss  # or chroma
VECTOR_STORE_PATH=./data/vector_store

# Search
TOP_K=5
SIMILARITY_THRESHOLD=0.3
USE_HYBRID_SEARCH=true

# LLM
TEMPERATURE=0.1
MAX_TOKENS=2048
```

### Python API

```python
from pdf_ingest import (
    IngestionPipeline, KnowledgeBase,
    RecipeExtractor, InvoiceExtractor,
)

# Build pipeline
pipeline = IngestionPipeline(vector_store_path="./data/my_kb")

# Ingest various sources
pipeline.ingest_pdf("document.pdf", use_ocr=True)
pipeline.ingest_audio("meeting.wav")
pipeline.ingest_web(["https://example.com/article"])
pipeline.ingest_office("report.docx")
pipeline.ingest_directory("./documents", recursive=True)

# Create knowledge base for querying
kb = KnowledgeBase("./data/my_kb")
kb.initialize_chat(model="llama3.1:8b")

# Chat with citations
response = kb.ask("What are the main conclusions?")
print(response.content)
for source in response.sources:
    print(f"  [{source.id}] Page {source.page}")

# Stream responses
for chunk in kb.ask_stream("Explain the methodology"):
    print(chunk, end="", flush=True)

# Summarize
summary = kb.summarize_all(max_length=500, style="concise")
print(summary.text)
print(summary.key_points)

# Generate study guide (Anki-ready)
guide = kb.generate_study_guide("flashcards")
print(guide.to_anki_csv())

# Extract structured data
recipes = kb.extract_recipes()
invoices = kb.extract_invoices()
medical = kb.extract_medical()
contracts = kb.extract_contracts()

for recipe in recipes:
    print(recipe.to_markdown())
```

### FastAPI Endpoints

```bash
# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Interactive docs at http://localhost:8000/docs
```

| Endpoint | Description |
|----------|-------------|
| `POST /ingest/file` | Upload files (multipart) |
| `POST /ingest/urls` | Ingest from URLs |
| `POST /ingest/directory` | Ingest directory |
| `POST /chat` | Chat with KB (with citations) |
| `POST /chat/stream` | Streaming chat (SSE) |
| `POST /search` | Hybrid search |
| `POST /summarize` | Generate summary |
| `POST /study` | Generate study materials |
| `POST /extract` | Extract structured data |
| `GET /kb/stats` | Knowledge base stats |
| `DELETE /kb` | Delete knowledge base |

---

## 🏗️ Architecture

```
pdf-ingest/
├── src/pdf_ingest/           # Core library (pip installable)
│   ├── config.py             # Pydantic settings management
│   ├── extract.py            # PDF text extraction (PyMuPDF)
│   ├── extract_audio.py      # Audio transcription (faster-whisper)
│   ├── extract_web.py        # Web article extraction (trafilatura)
│   ├── extract_ocr.py        # OCR for scanned PDFs/images
│   ├── extract_office.py     # DOCX, XLSX, PPTX extraction
│   ├── chunk.py              # Semantic/recursive/character chunking
│   ├── embed.py              # Ollama + SentenceTransformers embeddings
│   ├── store.py              # JSON + FAISS/Chroma vector stores
│   ├── search.py             # Hybrid vector + BM25 search
│   ├── chat.py               # RAG chat with streaming + citations
│   ├── summarize.py          # Map-reduce summaries + study guides
│   ├── pipeline.py           # Unified ingestion + KB management
│   ├── specialize/           # Domain extractors
│   │   ├── recipe.py         # Recipe extraction
│   │   ├── invoice.py        # Invoice/receipt extraction
│   │   ├── medical.py        # Medical report extraction
│   │   ├── contract.py       # Contract analysis
│   │   └── study.py          # Study guide generation
│   └── cli.py                # Rich CLI with subcommands
├── web/app.py                # Streamlit web UI
├── api/main.py               # FastAPI REST + WebSocket API
├── docker-compose.yml        # Multi-container deployment
├── Dockerfile                # Streamlit container
├── Dockerfile.api            # API container
├── pyproject.toml            # Modern packaging
└── tests/                    # Pytest suite
```

---

## 🔧 Advanced Configuration

### Chunking Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| `semantic` | General docs | Sentence-aware using embeddings |
| `recursive` | Structured docs | Paragraph → sentence → word fallback |
| `character` | Simple/legacy | Fixed-size with overlap |

### Embedding Options

```python
# Use Ollama (default)
embed_texts(texts, use_local=False)

# Use local SentenceTransformers (no Ollama needed)
embed_texts(texts, use_local=True)  # Uses all-MiniLM-L6-v2
```

### Vector Store Selection

```bash
# FAISS (default, lightweight, file-based)
VECTOR_STORE=faiss

# ChromaDB (persistent, more features)
VECTOR_STORE=chroma
```

### Custom Domain Extractors

```python
from pdf_ingest import RAGChat, KnowledgeBase
from pdf_ingest.specialize import RecipeExtractor

kb = KnowledgeBase("./data/kb")
kb.initialize_chat()

# Use built-in
recipes = kb.extract_recipes()

# Or create custom
chat = RAGChat("./data/kb", dimension=768)
extractor = RecipeExtractor(chat)
recipes = extractor.extract(chunks)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=pdf_ingest

# Specific test file
pytest tests/test_pipeline.py -v
```

---

## 📦 Project Structure

```
local-pdf-ingestion/
├── src/pdf_ingest/           # Core library
├── web/app.py                # Streamlit UI
├── api/main.py               # FastAPI backend
├── documents/                # Sample inputs
├── output/                   # Generated outputs (gitignored)
├── data/                     # Vector stores (gitignored)
├── docs/images/              # Documentation images
├── tests/                    # Test suite
├── scripts/                  # Utility scripts
├── pyproject.toml            # Packaging config
├── requirements.txt          # Dependencies
├── requirements-dev.txt      # Dev dependencies
├── docker-compose.yml        # Docker orchestration
├── Dockerfile                # Streamlit image
├── Dockerfile.api            # API image
├── .env.example              # Config template
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install dev dependencies: `uv sync --extra dev`
4. Make changes with tests
5. Run linting: `ruff check . && mypy src/`
6. Run tests: `pytest`
7. Submit a Pull Request

### Code Style
- **Formatter**: Ruff (line-length 100)
- **Type Checking**: MyPy (strict)
- **Tests**: Pytest with mocks (no external services needed)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) — Local LLM inference
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF processing
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
- [SentenceTransformers](https://www.sbert.net) — Local embeddings
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Audio transcription
- [trafilatura](https://trafilatura.readthedocs.io) — Web extraction
- [Streamlit](https://streamlit.io) — Web UI
- [FastAPI](https://fastapi.tiangolo.com) — API framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Jani-shiv/local-pdf-ingestion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Jani-shiv/local-pdf-ingestion/discussions)
- **Author**: [Shiv Jani](https://github.com/Jani-shiv) — DevOps Engineer @ Elastic

---

<div align="center">

**Star ⭐ this repo if you find it useful!**

Made with ❤️ for the open-source community

</div>