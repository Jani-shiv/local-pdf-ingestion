"""RAG chat pipeline: retrieve -> prompt -> stream answer with citations."""
from __future__ import annotations

import json
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from .config import settings
from .search import SearchResult, search
from .embed import embed_query
from .store import load_vector_store


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    content: str
    sources: list[SearchResult]
    model: str


class RAGChat:
    def __init__(
        self,
        store_path: str | Path,
        dimension: int,
        model: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.store_path = Path(store_path)
        self.dimension = dimension
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.ollama_url
        self.system_prompt = system_prompt or settings.system_prompt
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        self.history: list[ChatMessage] = []

    def _build_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            parts.append(f"[Source {i+1}] (page {r.page}, chunk {r.chunk}):\n{r.text}\n")
        return "\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    def _call_ollama(self, prompt: str, stream: bool = True) -> Generator[str, None, None] | str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                timeout=300,
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}. Is it running?") from exc
        except requests.HTTPError as exc:
            if response.status_code == 404:
                raise RuntimeError(f"Model '{self.model}' not found. Run: ollama pull {self.model}") from exc
            raise

        if stream:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        else:
            return response.json().get("response", "")

    def chat(self, query: str, top_k: int | None = None) -> ChatResponse:
        results = search(query, self.store_path, self.dimension, top_k=top_k)
        context = self._build_context(results)
        prompt = self._build_prompt(query, context)

        content_parts = []
        for chunk in self._call_ollama(prompt, stream=True):
            content_parts.append(chunk)

        content = "".join(content_parts)
        self.history.append(ChatMessage(role="user", content=query))
        self.history.append(ChatMessage(role="assistant", content=content))

        return ChatResponse(
            content=content,
            sources=results,
            model=self.model,
        )

    def chat_stream(self, query: str, top_k: int | None = None) -> Generator[str, None, list[SearchResult]]:
        results = search(query, self.store_path, self.dimension, top_k=top_k)
        context = self._build_context(results)
        prompt = self._build_prompt(query, context)

        for chunk in self._call_ollama(prompt, stream=True):
            yield chunk

        self.history.append(ChatMessage(role="user", content=query))
        self.history.append(ChatMessage(role="assistant", content="".join([])))
        return results

    def clear_history(self) -> None:
        self.history.clear()


def create_chat(
    store_path: str | Path,
    dimension: int,
    **kwargs,
) -> RAGChat:
    return RAGChat(store_path, dimension, **kwargs)