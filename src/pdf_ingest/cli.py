"""Command line entry point: python -m pdf_ingest documents/sample.pdf"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chunk import chunk_pages
from .embed import DEFAULT_MODEL, DEFAULT_URL, EmbeddingError, embed_texts
from .extract import extract_pages
from .store import save_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_ingest",
        description="Extract, chunk and embed a PDF with a local Ollama model.",
    )
    p.add_argument("pdf", help="path to a PDF with selectable text")
    p.add_argument("-o", "--output", default="output/processed_data.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--ollama-url", default=DEFAULT_URL)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="extract and chunk only, skip embeddings")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    pages, total_pages = extract_pages(args.pdf)
    print(f"Pages with text: {len(pages)} of {total_pages}")
    if not pages:
        print("No selectable text found. This PDF may be scanned and need OCR first.", file=sys.stderr)
        return 1
    if len(pages) < total_pages:
        print(f"Warning: {total_pages - len(pages)} page(s) had no extractable text.", file=sys.stderr)

    chunks = chunk_pages(pages, args.chunk_size, args.overlap)
    print(f"Chunks: {len(chunks)} (size={args.chunk_size}, overlap={args.overlap})")

    if args.dry_run:
        for c in chunks[:3]:
            print(f"\n[{c['id']}] {c['text'][:200]!r}")
        return 0

    try:
        vectors = embed_texts(
            [c["text"] for c in chunks],
            model=args.model,
            base_url=args.ollama_url,
            batch_size=args.batch_size,
        )
    except EmbeddingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    out = save_dataset(
        args.output, Path(args.pdf).name, chunks, vectors,
        model=args.model, chunk_size=args.chunk_size, overlap=args.overlap,
    )
    print(f"Embeddings: {len(vectors)} (dimensions: {len(vectors[0])})")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
