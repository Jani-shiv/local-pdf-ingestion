"""Web content extraction using trafilatura."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
import trafilatura


@dataclass
class WebContent:
    url: str
    title: str
    text: str
    author: str | None = None
    date: str | None = None
    domain: str = ""


class WebExtractor:
    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; pdf-ingest/0.2; +https://github.com/Jani-shiv/local-pdf-ingestion)",
        })

    def extract(self, url: str) -> WebContent:
        downloaded = trafilatura.fetch_url(url, timeout=self.timeout)
        if not downloaded:
            raise ValueError(f"Failed to download: {url}")

        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_images=False,
            output_format="json",
            with_metadata=True,
        )
        if not extracted:
            raise ValueError(f"Failed to extract content from: {url}")

        import json
        data = json.loads(extracted)

        domain = urlparse(url).netloc
        return WebContent(
            url=url,
            title=data.get("title", ""),
            text=data.get("text", ""),
            author=data.get("author"),
            date=data.get("date"),
            domain=domain,
        )

    def extract_multiple(self, urls: list[str]) -> list[WebContent]:
        results = []
        for url in urls:
            with contextlib.suppress(Exception):
                results.append(self.extract(url))
        return results


def extract_web_content(url: str) -> WebContent:
    extractor = WebExtractor()
    return extractor.extract(url)


def extract_web_multiple(urls: list[str]) -> list[WebContent]:
    extractor = WebExtractor()
    return extractor.extract_multiple(urls)