"""Web content extraction using trafilatura."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import trafilatura
import requests


@dataclass
class WebContent:
    url: str
    title: str
    text: str
    author: Optional[str] = None
    date: Optional[str] = None
    domain: str = ""


class WebExtractor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; pdf-ingest/0.2; +https://github.com/Jani-shiv/local-pdf-ingestion)"
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
            try:
                results.append(self.extract(url))
            except Exception as e:
                print(f"Warning: Failed to extract {url}: {e}")
        return results


def extract_web_content(url: str) -> WebContent:
    extractor = WebExtractor()
    return extractor.extract(url)


def extract_web_multiple(urls: list[str]) -> list[WebContent]:
    extractor = WebExtractor()
    return extractor.extract_multiple(urls)