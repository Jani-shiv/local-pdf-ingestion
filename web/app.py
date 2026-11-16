"""Streamlit web UI for pdf-ingest."""
import asyncio
import json
import shutil
import tempfile
import traceback
from pathlib import Path

import httpx
import numpy as np
import streamlit as st

from pdf_ingest import (
    IngestionPipeline,
    KnowledgeBase,
    create_knowledge_base,
    create_pipeline,
    settings,
)
from pdf_ingest import (
    search as search_fn,
)
from pdf_ingest.specialize import StudyGuide
from pdf_ingest.store import load_vector_store
from pdf_ingest.validation import ConfigurationError

st.set_page_config(
    page_title="pdf-ingest: Local RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
.stApp { max-width: 100%; }
.source-card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 8px 0; }
.source-card:hover { border-color: #007bff; }
.chat-message { padding: 12px; border-radius: 8px; margin: 8px 0; }
.user-message { background: #e3f2fd; margin-left: 20%; }
.assistant-message { background: #f5f5f5; margin-right: 20%; }
.source-citation { font-size: 0.85em; color: #666; font-style: italic; }
.error-box {
    background: #ffebee;
    border: 1px solid #ef5350;
    border-radius: 4px;
    padding: 12px;
    margin: 8px 0;
}
.success-box {
    background: #e8f5e9;
    border: 1px solid #66bb6a;
    border-radius: 4px;
    padding: 12px;
    margin: 8px 0;
}
.info-box {
    background: #e3f2fd;
    border: 1px solid #42a5f5;
    border-radius: 4px;
    padding: 12px;
    margin: 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading pipeline...")
def get_pipeline(vector_store_path: str) -> "IngestionPipeline":
    return create_pipeline(vector_store_path=vector_store_path)


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_kb(vector_store_path: str) -> "KnowledgeBase":
    return create_knowledge_base(vector_store_path)


def handle_error(error: Exception, context: str = "") -> None:
    """Display error with context."""
    if isinstance(error, ConfigurationError):
        st.error(f"⚠️ Configuration Error: {error}")
    elif isinstance(error, FileNotFoundError):
        st.error(f"📁 File Not Found: {error}")
    elif isinstance(error, ConnectionError):
        st.error("🔌 Connection Error: Could not connect to Ollama. Is it running?")
    else:
        st.error(f"❌ Error{f' in {context}' if context else ''}: {error}")
        if st.session_state.get("show_debug", False):
            with st.expander("🔍 Debug Info"):
                st.code(traceback.format_exc())


def sidebar() -> str:
    with st.sidebar:
        st.title("📄 pdf-ingest")
        st.caption("Local-first RAG - NotebookLM alternative")

        vector_store = st.text_input(
            "Vector Store Path",
            value=str(settings.vector_store_path),
            help="Where to store/load the knowledge base",
        )

        st.divider()
        st.subheader("⚙️ Settings")
        settings.chunk_size = st.number_input("Chunk Size", 100, 5000, settings.chunk_size)
        settings.chunk_overlap = st.number_input("Overlap", 0, 1000, settings.chunk_overlap)
        settings.chunk_strategy = st.selectbox(
            "Chunk Strategy", ["semantic", "recursive", "character"], index=0,
        )
        settings.top_k = st.number_input("Top K Results", 1, 20, settings.top_k)
        settings.similarity_threshold = st.slider(
            "Similarity Threshold", 0.0, 1.0, settings.similarity_threshold, 0.05,
        )
        settings.use_hybrid_search = st.checkbox(
            "Hybrid Search (Vector + Keyword)", value=settings.use_hybrid_search,
        )

        st.divider()
        st.subheader("🤖 LLM Settings")
        settings.llm_model = st.text_input("Ollama Model", value=settings.llm_model)
        settings.ollama_url = st.text_input("Ollama URL", value=settings.ollama_url)
        settings.temperature = st.slider("Temperature", 0.0, 1.0, settings.temperature, 0.1)
        settings.max_tokens = st.number_input("Max Tokens", 100, 8192, settings.max_tokens)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Cache & Reload"):
                st.cache_resource.clear()
                st.rerun()
        with col2:
            st.session_state.show_debug = st.checkbox(
                "Debug Mode", value=st.session_state.get("show_debug", False),
            )

        # Health check
        st.divider()
        st.subheader("🏥 System Status")

        async def check_ollama() -> bool:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{settings.ollama_url}/api/tags")
                return resp.status_code == httpx.codes.OK

        try:
            ollama_ok = asyncio.run(check_ollama())
            st.markdown(
                f"{'✅' if ollama_ok else '❌'} Ollama: "
                f"{'Connected' if ollama_ok else 'Disconnected'}",
            )
        except Exception:
            st.markdown("❌ Ollama: Connection failed")

        return vector_store


def page_ingest(vector_store: str) -> None:
    st.header("📥 Ingest Documents")
    st.markdown("Upload PDFs, audio, Office docs, or provide URLs to build your knowledge base.")

    pipeline = get_pipeline(vector_store)

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Files", "🔗 URLs", "🎵 Audio", "📁 Directory"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "xlsx", "pptx", "png", "jpg", "jpeg", "mp3", "wav", "m4a"],
            accept_multiple_files=True,
        )
        use_ocr = st.checkbox("Use OCR for PDFs (scanned documents)", value=False)
        if st.button("Ingest Files", type="primary") and uploaded:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            errors = []

            for i, file in enumerate(uploaded):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / len(uploaded))

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{file.name.split('.')[-1]}",
                ) as tmp:
                    tmp.write(file.read())
                    tmp_path = Path(tmp.name)

                try:
                    suffix = tmp_path.suffix.lower()
                    if suffix == ".pdf":
                        results.append(pipeline.ingest_pdf(tmp_path, use_ocr=use_ocr))
                    elif suffix in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
                        results.append(pipeline.ingest_audio(tmp_path))
                    elif suffix in (".docx", ".xlsx", ".pptx"):
                        results.append(pipeline.ingest_office(tmp_path))
                    else:
                        errors.append(f"Unsupported: {file.name}")
                except Exception as e:
                    errors.append(f"{file.name}: {e}")
                finally:
                    tmp_path.unlink(missing_ok=True)

            status_text.text("Done!")
            progress_bar.empty()

            if results:
                st.success(f"✅ Ingested {len(results)} file(s)!")
                for r in results:
                    st.write(f"✅ **{r.source}**: {r.metadata.get('chunks', 0)} chunks")

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")

    with tab2:
        urls = st.text_area(
            "URLs (one per line)", height=150, placeholder="https://example.com/article\nhttps://blog.example.com/post",
        )
        if st.button("Ingest URLs", type="primary") and urls.strip():
            url_list = [u.strip() for u in urls.split("\n") if u.strip()]
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Fetching and ingesting...")
                results = pipeline.ingest_web(url_list)
                progress_bar.progress(1.0)
                st.success(f"✅ Ingested {len(results)} URL(s)!")
                for r in results:
                    st.write(
                        f"✅ **{r.metadata.get('title', r.source)}**: "
                        f"{r.metadata.get('chunks', 0)} chunks",
                    )
            except Exception as e:
                handle_error(e, "ingesting URLs")
            finally:
                progress_bar.empty()
                status_text.empty()

    with tab3:
        audio_files = st.file_uploader(
            "Upload audio files",
            type=["mp3", "wav", "m4a", "flac", "ogg"],
            accept_multiple_files=True,
        )
        if st.button("Transcribe & Ingest", type="primary") and audio_files:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(audio_files):
                status_text.text(f"Transcribing {file.name}...")
                progress_bar.progress((i + 1) / len(audio_files))

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{file.name.split('.')[-1]}",
                ) as tmp:
                    tmp.write(file.read())
                    tmp_path = Path(tmp.name)
                try:
                    result = pipeline.ingest_audio(tmp_path)
                    st.success(
                        f"✅ **{result.source}**: "
                        f"{result.metadata.get('chunks', 0)} chunks",
                    )
                except Exception as e:
                    handle_error(e, f"transcribing {file.name}")
                finally:
                    tmp_path.unlink(missing_ok=True)

            progress_bar.empty()
            status_text.empty()

    with tab4:
        dir_path = st.text_input("Directory path", placeholder="/path/to/documents")
        recursive = st.checkbox("Recursive", value=True)
        if st.button("Ingest Directory", type="primary") and dir_path:
            try:
                with st.spinner("Scanning and ingesting..."):
                    results = pipeline.ingest_directory(dir_path, recursive=recursive)
                st.success(f"✅ Ingested {len(results)} file(s)!")
                for r in results:
                    st.write(
                        f"✅ **{r.source}**: {r.metadata.get('chunks', 0)} chunks",
                    )
            except Exception as e:
                handle_error(e, "ingesting directory")


def page_chat(vector_store: str) -> None:
    st.header("💬 Chat with Your Knowledge Base")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Model selector in chat
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        model_options = ["llama3.1:8b", "phi3:mini", "mistral:7b", "gemma2:9b"]
        idx = (
            model_options.index(settings.llm_model)
            if settings.llm_model in model_options
            else 0
        )
        model = st.selectbox("Model", model_options, index=idx)
        settings.llm_model = model
    with col2:
        if st.button("🔄 New Chat"):
            st.session_state.messages = []
            st.rerun()
    with col3:
        streaming = st.checkbox("Stream", value=True)

    kb = get_kb(vector_store)
    kb.initialize_chat(model=settings.llm_model)

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"**[{s['id']}]** Page {s['page']} • Score: {s['score']:.3f}")
                        st.text(s["text"][:300] + "...")

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                if streaming:
                    # Streaming response
                    response_placeholder = st.empty()
                    full_content = ""
                    for chunk in kb.ask_stream(prompt, top_k=settings.top_k):
                        full_content += chunk
                        response_placeholder.markdown(full_content + "▌")

                    # Get sources after streaming
                    response = kb.ask(prompt, top_k=settings.top_k)
                    response_placeholder.markdown(full_content)

                    if response.sources:
                        with st.expander(f"📚 Sources ({len(response.sources)})"):
                            for s in response.sources:
                                st.markdown(f"**[{s.id}]** Page {s.page} • Score: {s.score:.3f}")
                                st.text(s.text[:300] + "...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_content,
                        "sources": [
                            {"id": s.id, "text": s.text, "page": s.page, "score": s.score}
                            for s in response.sources
                        ],
                    })
                else:
                    # Non-streaming response
                    with st.spinner("Thinking..."):
                        response = kb.ask(prompt, top_k=settings.top_k)

                    st.markdown(response.content)

                    if response.sources:
                        with st.expander(f"📚 Sources ({len(response.sources)})"):
                            for s in response.sources:
                                st.markdown(f"**[{s.id}]** Page {s.page} • Score: {s.score:.3f}")
                                st.text(s.text[:300] + "...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "sources": [
                            {"id": s.id, "text": s.text, "page": s.page, "score": s.score}
                            for s in response.sources
                        ],
                    })
            except Exception as e:
                handle_error(e, "chat")


def page_search(vector_store: str) -> None:
    st.header("🔍 Search Knowledge Base")

    kb = get_kb(vector_store)

    query = st.text_input("Search query", placeholder="Enter your search...")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        top_k = st.slider("Results", 1, 20, settings.top_k)
    with col2:
        threshold = st.slider("Min Score", 0.0, 1.0, settings.similarity_threshold, 0.05)
    with col3:
        st.checkbox("Hybrid", value=settings.use_hybrid_search)

    if st.button("Search", type="primary") and query:
        try:
            with st.spinner("Searching..."):
                results = search_fn(query, vector_store, kb.store.dimension, top_k, threshold)

            st.write(f"Found **{len(results)}** results")
            for r in results:
                with st.container():
                    st.markdown(
                        f"**[{r.id}]** Page {r.page} • "
                        f"Score: `{r.score:.3f}` • Source: `{r.source}`",
                    )
                    PREVIEW_LENGTH = 500
                    st.text(
                        r.text[:PREVIEW_LENGTH]
                        + ("..." if len(r.text) > PREVIEW_LENGTH else ""),
                    )
                    st.divider()
        except Exception as e:
            handle_error(e, "search")


def page_summarize(vector_store: str) -> None:
    st.header("📝 Summarize & Study")

    kb = get_kb(vector_store)

    tab1, tab2, tab3 = st.tabs(["📋 Summary", "🎓 Study Guide", "📊 Extract"])

    with tab1:
        st.subheader("Generate Summary")
        col1, col2 = st.columns(2)
        with col1:
            max_len = st.number_input("Max Length (words)", 100, 2000, 500)
        with col2:
            style = st.selectbox("Style", ["concise", "detailed", "bullet"])

        if st.button("Generate Summary", type="primary"):
            try:
                with st.spinner("Summarizing..."):
                    summary = kb.summarize_all(max_length=max_len, style=style)
                st.markdown("### Summary")
                st.write(summary.text)
                if summary.key_points:
                    st.markdown("### Key Points")
                    for kp in summary.key_points:
                        st.markdown(f"- {kp}")
            except Exception as e:
                handle_error(e, "summarization")

    with tab2:
        st.subheader("Generate Study Materials")
        fmt = st.selectbox("Format", ["flashcards", "mindmap", "quiz", "all"])
        anki = st.checkbox("Export as Anki CSV", value=False)

        if st.button("Generate", type="primary"):
            try:
                with st.spinner("Creating study guide..."):
                    result = kb.generate_study_guide(fmt)
                if fmt == "flashcards" and anki:
                    guide = StudyGuide(**result)
                    csv_data = guide.to_anki_csv()
                    st.download_button(
                        "Download Anki CSV", csv_data, "study_cards.csv", "text/csv",
                    )
                else:
                    st.json(result)
            except Exception as e:
                handle_error(e, "study guide generation")

    with tab3:
        st.subheader("Extract Structured Data")
        extract_type = st.selectbox(
            "Type", ["recipes", "invoices", "medical", "contracts", "study"],
        )
        out_fmt = st.selectbox("Output Format", ["json", "markdown"])

        if st.button("Extract", type="primary"):
            try:
                with st.spinner(f"Extracting {extract_type}..."):
                    items = getattr(kb, f"extract_{extract_type}")()

                if out_fmt == "json":
                    st.json([item.to_dict() for item in items])
                else:
                    for item in items:
                        st.markdown(item.to_markdown())
                        st.divider()
            except Exception as e:
                handle_error(e, "extraction")


def page_manage(vector_store: str) -> None:
    st.header("📚 Knowledge Base Management")

    kb = get_kb(vector_store)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", kb.store.count)
    with col2:
        st.metric("Vector Dimension", kb.store.dimension)
    with col3:
        st.metric("Store Path", str(vector_store))

    st.divider()

    st.subheader("Export Data")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export as JSON"):
            try:
                store = load_vector_store(vector_store, kb.store.dimension)
                all_results = store.search(
                    np.zeros(kb.store.dimension), store.count,
                )
                data = {
                    "meta": {
                        "chunks": len(all_results),
                        "dimension": kb.store.dimension,
                    },
                    "chunks": [
                        {
                            "id": m["id"],
                            "text": m["text"],
                            "page": m["page"],
                            "chunk": m["chunk"],
                        }
                        for _, m in all_results
                    ],
                }
                st.download_button(
                    "Download JSON",
                    json.dumps(data, indent=2),
                    "knowledge_base.json",
                    "application/json",
                )
            except Exception as e:
                handle_error(e, "JSON export")

    with col2:
        if st.button("Export as Markdown"):
            try:
                store = load_vector_store(vector_store, kb.store.dimension)
                all_results = store.search(
                    np.zeros(kb.store.dimension), store.count,
                )
                md = "# Knowledge Base Export\n\n"
                for _, m in all_results:
                    md += f"## [{m['id']}] Page {m['page']}\n{m['text']}\n\n---\n\n"
                st.download_button(
                    "Download Markdown", md, "knowledge_base.md", "text/markdown",
                )
            except Exception as e:
                handle_error(e, "Markdown export")

    st.divider()
    st.subheader("Danger Zone")
    if st.button("🗑️ Delete Knowledge Base", type="secondary"):
        if st.checkbox("I understand this cannot be undone"):
            try:
                shutil.rmtree(vector_store, ignore_errors=True)
                st.success("Knowledge base deleted. Refresh to recreate.")
                st.cache_resource.clear()
            except Exception as e:
                handle_error(e, "deleting knowledge base")


def main() -> None:
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False

    vector_store = sidebar()

    pages = {
        "📥 Ingest": page_ingest,
        "💬 Chat": page_chat,
        "🔍 Search": page_search,
        "📝 Summarize": page_summarize,
        "📚 Manage": page_manage,
    }

    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[selection](vector_store)


if __name__ == "__main__":
    main()