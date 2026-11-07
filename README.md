<div align="center">

<img src="docs/images/00-cover.png" alt="Local PDF ingestion pipeline cover" width="100%">

# 📄 Local PDF Ingestion

### Build a private, local-first knowledge dataset from your PDFs
<p>
  <a href="https://github.com/Jani-shiv/local-pdf-ingestion/stargazers"><img src="https://img.shields.io/github/stars/Jani-shiv/local-pdf-ingestion?style=for-the-badge&logo=github&color=yellow" alt="GitHub stars"></a>
  <a href="https://github.com/Jani-shiv/local-pdf-ingestion/network/members"><img src="https://img.shields.io/github/forks/Jani-shiv/local-pdf-ingestion?style=for-the-badge&logo=github&color=blue" alt="GitHub forks"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License">
</p>

</div>

> Turn a PDF into a searchable knowledge dataset using **Python**, **PyMuPDF**,
> and a local **Ollama** embedding model. Nothing leaves your machine.

## 🧭 Project overview

This is step 1 of a local RAG (Retrieval-Augmented Generation) series: the
ingestion layer. It does **not** include similarity search or a chatbot yet.

| Input | Processing | Output |
|---|---|---|
| PDF with selectable text | Extract → chunk → embed locally | JSON knowledge dataset |

<div align="center">

<img src="docs/images/01-pipeline.png" alt="PDF extraction, chunking, embedding, and JSON storage pipeline" width="100%">

<br>
<em>PDF → extract text → chunk → embed with Ollama → save JSON</em>

</div>

## 👋 About the author

<table>
<tr>
<td width="96" align="center">
<img src="https://github.com/Jani-shiv.png" width="80" alt="Shiv Jani">
</td>
<td>

### Shiv Jani · `Jani-shiv`

**DevOps Engineer & Cloud Enthusiast** passionate about automation, scalable
systems, and modern deployment workflows.

📍 **Ahmedabad, Gujarat** &nbsp;·&nbsp; 🏢 **Elastic** &nbsp;·&nbsp; **he/him**

<a href="https://github.com/Jani-shiv">GitHub</a> ·
<a href="https://jani-shiv.github.io/Shiv_Jani/">Portfolio</a> ·
<a href="https://www.linkedin.com/in/shiv-jani">LinkedIn</a> ·
<a href="https://www.instagram.com/jani._.712/">Instagram</a>

</td>
</tr>
</table>

<p>
  <a href="https://github.com/Jani-shiv?tab=followers"><img src="https://img.shields.io/badge/Followers-35-8B5CF6?style=flat-square&logo=github" alt="35 followers"></a>
  <a href="https://github.com/Jani-shiv?tab=following"><img src="https://img.shields.io/badge/Following-19-EC4899?style=flat-square&logo=github" alt="19 following"></a>
</p>

## ✨ What it does

The pipeline:

1. Extracts selectable text from each PDF page with PyMuPDF.
2. Splits the text into overlapping character chunks.
3. Generates embeddings locally through Ollama.
4. Writes chunk text, metadata, and vectors to a JSON dataset.

The project is intentionally small and local-first. It does not upload documents
or embeddings to a hosted service, and it does not yet provide vector search or
a chatbot.

## 🖼️ Visual guide

<table>
<tr>
<td width="50%" align="center">
<img src="docs/images/02-chunking-overlap.png" alt="Chunking and overlap illustration" width="100%">
<br><strong>Chunking with overlap</strong>
</td>
<td width="50%" align="center">
<img src="docs/images/03-embeddings-concept.png" alt="Embedding concept illustration" width="100%">
<br><strong>Local embeddings</strong>
</td>
</tr>
</table>

## 🧰 Requirements

- Python 3.10 or later
- [Ollama](https://ollama.com) installed and running
- A PDF with **selectable text** (scanned PDFs need OCR first)

## 🚀 Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Jani-shiv/local-pdf-ingestion.git
cd local-pdf-ingestion

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, allow scripts only for the current
session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
```

## ▶️ Run the pipeline

Make sure Ollama is installed and running, then download the default embedding
model:

```bash
ollama pull nomic-embed-text
```

Generate the included sample PDF if needed:

```bash
python scripts/make_sample_pdf.py
```

Run ingestion on Linux/macOS:

```bash
PYTHONPATH=src python -m pdf_ingest documents/sample.pdf
```

Run ingestion in Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pdf_ingest documents\sample.pdf
```

The generated dataset is saved to `output/processed_data.json`. The `output/`
directory is ignored by Git because it is generated from the input PDF.

## 🧪 Validate extraction without Ollama

Use `--dry-run` to extract and chunk a PDF without downloading a model or
calling the Ollama API:

```bash
PYTHONPATH=src python -m pdf_ingest documents/sample.pdf --dry-run
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pdf_ingest documents\sample.pdf --dry-run
```

## ⚙️ Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `-o, --output` | `output/processed_data.json` | Where to save the dataset |
| `--model` | `nomic-embed-text` | Ollama embedding model |
| `--ollama-url` | `http://localhost:11434` | Ollama server |
| `--chunk-size` | `1000` | Characters per chunk |
| `--overlap` | `150` | Characters shared between neighbouring chunks |
| `--batch-size` | `16` | Chunks sent to Ollama per request |
| `--dry-run` | off | Skip embeddings |

Example with custom settings:

```bash
PYTHONPATH=src python -m pdf_ingest input.pdf \
  --output output/custom.json \
  --model nomic-embed-text \
  --chunk-size 1200 \
  --overlap 200
```

## 🗂️ Project layout

```
src/pdf_ingest/
  extract.py   PyMuPDF text extraction (page by page)
  chunk.py     character chunking with overlap
  embed.py     batched calls to Ollama /api/embed
  store.py     JSON dataset with metadata
  cli.py       command line interface
tests/         pytest suite (Ollama is mocked)
scripts/       sample PDF + diagram generators
docs/images/   article diagrams
```

## 📦 Output format

```json
{
  "meta": {"source": "sample.pdf", "model": "nomic-embed-text", "dimensions": 768,
           "chunk_size": 1000, "overlap": 150, "chunk_count": 6, "created_at": "..."},
  "chunks": [{"id": "p1-c0", "page": 1, "chunk": 0, "text": "...", "embedding": [0.01, "..."]}]
}
```

The model name and chunk settings are stored with the vectors on purpose. Vectors from different embedding models cannot be compared with each other.

## 🛠️ Development

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
pytest
```

The Ollama calls are mocked in tests, so a running Ollama server is not needed
to run the test suite.

## ⚠️ Known limits

- Character chunking ignores paragraphs, headings and tables.
- Scanned PDFs return no text. Add an OCR step first.
- JSON is fine for learning. For large collections use a vector database.
- The default output is one JSON file, which is not suitable for very large
  document collections.
- Ollama calls are covered by mocked tests, not by a live server run in CI.

## 🩹 Troubleshooting

### `source` is not recognized

`source venv/bin/activate` is for Bash. In Windows PowerShell, use:

```powershell
.\venv\Scripts\Activate.ps1
```

### `PYTHONPATH=src` is not recognized

That inline environment-variable syntax is for Bash. In Windows PowerShell,
use:

```powershell
$env:PYTHONPATH = "src"
python -m pdf_ingest documents\sample.pdf
```

### `ollama` is not recognized

Install Ollama from [ollama.com](https://ollama.com), restart the terminal,
ensure the Ollama application is running, and verify:

```powershell
ollama --version
ollama pull nomic-embed-text
```

## 🧭 Next

Vector search: finding the most relevant chunks for a question.

## 📜 License

MIT
