"""Enhanced command line entry point with all features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import settings
from .pipeline import IngestionPipeline, KnowledgeBase, create_pipeline, create_knowledge_base
from .specialize import Recipe, Invoice, MedicalReport, Contract, StudyGuide
from .validation import (
    InputValidator,
    ConfigurationError,
    validate_output_path,
    validate_store_path,
)
from .progress import progress_bar, track


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf-ingest",
        description="Local-first PDF ingestion + RAG: extract, chunk, embed, search, chat — all offline",
    )
    subparsers = p.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest = subparsers.add_parser("ingest", help="Ingest documents into knowledge base")
    ingest.add_argument("paths", nargs="+", help="File(s) or directory to ingest")
    ingest.add_argument("-o", "--output", default=str(settings.vector_store_path), help="Vector store path")
    ingest.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    ingest.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    ingest.add_argument("--strategy", choices=["character", "semantic", "recursive"], default=settings.chunk_strategy)
    ingest.add_argument("--ocr", action="store_true", help="Use OCR for PDFs")
    ingest.add_argument("--recursive", action="store_true", help="Recursively ingest directory")

    # Chat command
    chat = subparsers.add_parser("chat", help="Chat with your knowledge base")
    chat.add_argument("query", nargs="?", help="Question to ask (interactive if omitted)")
    chat.add_argument("-s", "--store", default=str(settings.vector_store_path), help="Vector store path")
    chat.add_argument("--model", default=settings.llm_model)
    chat.add_argument("--top-k", type=int, default=settings.top_k)
    chat.add_argument("--threshold", type=float, default=settings.similarity_threshold)
    chat.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search")
    chat.add_argument("--stream", action="store_true", help="Stream response")

    # Search command
    search = subparsers.add_parser("search", help="Search knowledge base")
    search.add_argument("query", help="Search query")
    search.add_argument("-s", "--store", default=str(settings.vector_store_path))
    search.add_argument("--top-k", type=int, default=settings.top_k)
    search.add_argument("--threshold", type=float, default=settings.similarity_threshold)
    search.add_argument("--json", action="store_true", help="Output as JSON")

    # Summarize command
    summarize = subparsers.add_parser("summarize", help="Summarize knowledge base")
    summarize.add_argument("-s", "--store", default=str(settings.vector_store_path))
    summarize.add_argument("--max-length", type=int, default=500)
    summarize.add_argument("--style", choices=["concise", "detailed", "bullet"], default="concise")
    summarize.add_argument("--output", help="Output file (JSON)")

    # Study guide command
    study = subparsers.add_parser("study", help="Generate study materials")
    study.add_argument("-s", "--store", default=str(settings.vector_store_path))
    study.add_argument("--format", choices=["flashcards", "mindmap", "quiz", "all"], default="flashcards")
    study.add_argument("--output", help="Output file")
    study.add_argument("--anki", action="store_true", help="Export as Anki CSV")

    # Extract command (domain-specific)
    extract = subparsers.add_parser("extract", help="Extract structured data")
    extract.add_argument("type", choices=["recipes", "invoices", "medical", "contracts", "study"])
    extract.add_argument("-s", "--store", default=str(settings.vector_store_path))
    extract.add_argument("--output", help="Output file (JSON)")
    extract.add_argument("--format", choices=["json", "markdown", "csv"], default="json")

    # List sources
    list_cmd = subparsers.add_parser("list", help="List ingested sources")
    list_cmd.add_argument("-s", "--store", default=str(settings.vector_store_path))

    # Config command
    config = subparsers.add_parser("config", help="Show current configuration")
    config.add_argument("--edit", action="store_true", help="Open config in editor")

    return p


def cmd_ingest(args) -> int:
    # Validate inputs
    try:
        InputValidator.validate_ingest_args(args)
    except ConfigurationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    pipeline = create_pipeline(
        vector_store_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy,
    )

    # Collect all files to process
    files_to_process: list[Path] = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_dir():
            pattern = "**/*" if args.recursive else "*"
            for file_path in path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in {
                    ".pdf", ".mp3", ".wav", ".m4a", ".flac", ".ogg",
                    ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
                }:
                    files_to_process.append(file_path)
        else:
            files_to_process.append(path)

    all_results = []

    # Process files with progress bar
    with progress_bar(total=len(files_to_process), desc="Ingesting", unit="file") as pbar:
        for file_path in files_to_process:
            pbar.set_description(f"Ingesting {file_path.name}")
            suffix = file_path.suffix.lower()

            try:
                if suffix == ".pdf":
                    result = pipeline.ingest_pdf(file_path, use_ocr=args.ocr)
                    all_results.append(result)
                elif suffix in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
                    result = pipeline.ingest_audio(file_path)
                    all_results.append(result)
                elif suffix in (".docx", ".xlsx", ".pptx"):
                    result = pipeline.ingest_office(file_path)
                    all_results.append(result)
                elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                    # Could add image OCR here
                    print(f"Skipping image (OCR not yet supported): {file_path}", file=sys.stderr)
                else:
                    print(f"Skipping unsupported file: {file_path}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to ingest {file_path}: {e}", file=sys.stderr)

            pbar.update(1)

    print(f"\nIngested {len(all_results)} source(s):")
    for r in all_results:
        print(f"  - {r.source}: {r.metadata.get('chunks', 0)} chunks")

    return 0


def cmd_chat(args) -> int:
    # Validate inputs
    try:
        InputValidator.validate_chat_args(args)
    except ConfigurationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    kb = create_knowledge_base(args.store)
    kb.initialize_chat(model=args.model)

    if args.query:
        if args.stream:
            print("Answer: ", end="", flush=True)
            sources = None
            for chunk in kb.ask_stream(args.query, top_k=args.top_k):
                print(chunk, end="", flush=True)
            print()
        else:
            response = kb.ask(args.query, top_k=args.top_k)
            print(f"Answer: {response.content}")
            if response.sources:
                print("\nSources:")
                for s in response.sources:
                    print(f"  [{s.id}] page {s.page} (score: {s.score:.3f})")
        return 0

    # Interactive mode
    print("Interactive chat (type 'exit' or 'quit' to leave)")
    print(f"Model: {args.model} | Store: {args.store}")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        print("Assistant: ", end="", flush=True)
        for chunk in kb.ask_stream(query, top_k=args.top_k):
            print(chunk, end="", flush=True)
        print()
    return 0


def cmd_search(args) -> int:
    # Validate inputs
    try:
        InputValidator.validate_search_args(args)
    except ConfigurationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    kb = create_knowledge_base(args.store)
    from .search import search
    results = search(args.query, args.store, kb.store.dimension, args.top_k, args.threshold)

    if args.json:
        print(json.dumps([{
            "id": r.id, "text": r.text[:200], "page": r.page,
            "chunk": r.chunk, "score": r.score, "source": r.source
        } for r in results], indent=2))
    else:
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  [{r.id}] page {r.page} (score: {r.score:.3f})")
            print(f"    {r.text[:200]}...")
    return 0


def cmd_summarize(args) -> int:
    # Validate inputs
    try:
        InputValidator.validate_summarize_args(args)
    except ConfigurationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    kb = create_knowledge_base(args.store)
    summary = kb.summarize_all(max_length=args.max_length, style=args.style)

    output = {
        "summary": summary.text,
        "key_points": summary.key_points,
        "chunks_used": summary.chunks_used,
    }

    if args.output:
        output_path = validate_output_path(args.output)
        output_path.write_text(json.dumps(output, indent=2))
        print(f"Saved to {output_path}")
    else:
        print(f"Summary:\n{summary.text}")
        if summary.key_points:
            print("\nKey Points:")
            for kp in summary.key_points:
                print(f"  - {kp}")
    return 0


def cmd_study(args) -> int:
    # Validate inputs
    try:
        InputValidator.validate_study_args(args)
    except ConfigurationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    kb = create_knowledge_base(args.store)
    result = kb.generate_study_guide(args.format)

    if args.format == "flashcards" and args.anki:
        from .specialize import StudyGuide
        guide = StudyGuide(**result)
        output = guide.to_anki_csv()
    else:
        output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        output_path = validate_output_path(args.output)
        output_path.write_text(output)
        print(f"Saved to {output_path}")
    else:
        print(output)
    return 0


def cmd_extract(args) -> int:
    # Validate inputs
    try:
        InputValidator.validate_extract_args(args)
    except ConfigurationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    kb = create_knowledge_base(args.store)

    extractors = {
        "recipes": (kb.extract_recipes, Recipe),
        "invoices": (kb.extract_invoices, Invoice),
        "medical": (kb.extract_medical, MedicalReport),
        "contracts": (kb.extract_contracts, Contract),
        "study": (kb.extract_study_guides, StudyGuide),
    }

    extract_fn, cls = extractors[args.type]
    items = extract_fn()

    if args.format == "json":
        output = json.dumps([item.to_dict() for item in items], indent=2, ensure_ascii=False)
    elif args.format == "markdown":
        output = "\n\n---\n\n".join(item.to_markdown() for item in items)
    elif args.format == "csv" and args.type == "invoices":
        import csv
        from io import StringIO
        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Vendor", "Date", "Total", "Category", "Items"])
        for item in items:
            writer.writerow([item.vendor, item.date or "", item.total or "", item.category or "", len(item.line_items)])
        output = buf.getvalue()
    else:
        output = json.dumps([item.to_dict() for item in items], indent=2, ensure_ascii=False)

    if args.output:
        output_path = validate_output_path(args.output)
        output_path.write_text(output)
        print(f"Saved {len(items)} {args.type} to {output_path}")
    else:
        print(output)
    return 0


def cmd_list(args) -> int:
    # Validate inputs
    try:
        InputValidator.validate_list_args(args)
    except ConfigurationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    kb = create_knowledge_base(args.store)
    print(f"Vector store: {args.store}")
    print(f"Total chunks: {kb.store.count}")
    print(f"Dimension: {kb.store.dimension}")
    return 0


def cmd_config(args) -> int:
    print("Current Configuration:")
    for key, value in settings.model_dump().items():
        print(f"  {key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    commands = {
        "ingest": cmd_ingest,
        "chat": cmd_chat,
        "search": cmd_search,
        "summarize": cmd_summarize,
        "study": cmd_study,
        "extract": cmd_extract,
        "list": cmd_list,
        "config": cmd_config,
    }

    if args.command not in commands:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())