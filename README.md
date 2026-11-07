# local-pdf-ingestion

Turn a PDF into a searchable knowledge dataset using **Python**, **PyMuPDF** and a local **Ollama** embedding model. Nothing leaves your machine.

![Pipeline](docs/images/01-pipeline.png)

This is step 1 of a local RAG (Retrieval-Augmented Generation) series: the ingestion layer. It does **not** include similarity search or a chatbot yet.

```
PDF -> extract text -> chunk -> embed (Ollama) -> save JSON
```

## Requirements

- Python 3.10 or later
- [Ollama](https://ollama.com) installed and running
- A PDF with **selectable text** (scanned PDFs need OCR first)

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/local-pdf-ingestion.git
cd local-pdf-ingestion

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

ollama pull nomic-embed-text

python scripts/make_sample_pdf.py            # optional: regenerate documents/sample.pdf
PYTHONPATH=src python -m pdf_ingest documents/sample.pdf
```

On Windows PowerShell, set the path with `$env:PYTHONPATH="src"` before running the last command.

Output is written to `output/processed_data.json`.

### Try it without Ollama

```bash
PYTHONPATH=src python -m pdf_ingest documents/sample.pdf --dry-run
```

This extracts and chunks only, so you can check your PDF before spending time on embeddings.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-o, --output` | `output/processed_data.json` | Where to save the dataset |
| `--model` | `nomic-embed-text` | Ollama embedding model |
| `--ollama-url` | `http://localhost:11434` | Ollama server |
| `--chunk-size` | `1000` | Characters per chunk |
| `--overlap` | `150` | Characters shared between neighbouring chunks |
| `--batch-size` | `16` | Chunks sent to Ollama per request |
| `--dry-run` | off | Skip embeddings |

## Project layout

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

## Output format

```json
{
  "meta": {"source": "sample.pdf", "model": "nomic-embed-text", "dimensions": 768,
           "chunk_size": 1000, "overlap": 150, "chunk_count": 6, "created_at": "..."},
  "chunks": [{"id": "p1-c0", "page": 1, "chunk": 0, "text": "...", "embedding": [0.01, "..."]}]
}
```

The model name and chunk settings are stored with the vectors on purpose. Vectors from different embedding models cannot be compared with each other.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Known limits

- Character chunking ignores paragraphs, headings and tables.
- Scanned PDFs return no text. Add an OCR step first.
- JSON is fine for learning. For large collections use a vector database.
- Tested on Linux with Python 3.12. The Ollama calls are covered by mocked tests, not by a live server run in this repo's CI.

## Next

Vector search: finding the most relevant chunks for a question.

## License

MIT
