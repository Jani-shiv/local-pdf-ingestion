"""Invoice/Receipt extractor: structured data for expense tracking.

This module provides:
- Invoice, LineItem dataclasses with serialization
- InvoiceExtractor: LLM-based extraction from text chunks

Example:
    extractor = InvoiceExtractor(chat)
    invoices = extractor.extract(chunks)
    for inv in invoices:
        print(inv.to_markdown())
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional

from ..chat import RAGChat
from ..chunk import Chunk


@dataclass
class LineItem:
    """Invoice line item.

    Attributes:
        description: Item description
        quantity: Quantity ordered
        unit_price: Price per unit
        total: Line total (quantity * unit_price)
    """
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


@dataclass
class Invoice:
    """Structured invoice/receipt data.

    Attributes:
        vendor: Vendor/business name
        invoice_number: Invoice/reference number
        date: Invoice date (YYYY-MM-DD)
        due_date: Payment due date (YYYY-MM-DD)
        subtotal: Subtotal before tax
        tax: Tax amount
        total: Total amount
        currency: Currency code (default: USD)
        line_items: List of LineItem objects
        category: Expense category
        payment_method: Payment method used
        source_page: Source document page number
    """
    vendor: str
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: str = "USD"
    line_items: list["LineItem"] = field(default_factory=list)
    category: Optional[str] = None
    payment_method: Optional[str] = None
    source_page: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        """Convert to Markdown for display."""
        lines = [f"# Invoice: {self.vendor}"]
        if self.invoice_number:
            lines.append(f"**Invoice #:** {self.invoice_number}")
        if self.date:
            lines.append(f"**Date:** {self.date}")
        if self.due_date:
            lines.append(f"**Due Date:** {self.due_date}")
        if self.category:
            lines.append(f"**Category:** {self.category}")
        if self.payment_method:
            lines.append(f"**Payment:** {self.payment_method}")

        if self.line_items:
            lines.append("\n## Line Items")
            lines.append("| Description | Qty | Unit Price | Total |")
            lines.append("|---|---:|---:|---:|")
            for item in self.line_items:
                lines.append(f"| {item.description} | {item.quantity or ''} | {item.unit_price or ''} | {item.total or ''} |")

        lines.append(f"\n**Subtotal:** {self.currency} {self.subtotal or 'N/A'}")
        lines.append(f"**Tax:** {self.currency} {self.tax or 'N/A'}")
        lines.append(f"## **Total:** {self.currency} {self.total or 'N/A'}")

        return "\n".join(lines)

    def to_csv_row(self) -> list[str]:
        """Convert to CSV row for spreadsheet export."""
        return [
            self.vendor,
            self.date or "",
            str(self.total or ""),
            self.category or "",
            str(len(self.line_items)),
        ]

    @property
    def csv_headers(self) -> list[str]:
        """CSV headers for invoice export."""
        return ["Vendor", "Date", "Total", "Category", "Items"]


class InvoiceExtractor:
    """Extract structured invoices from text chunks using LLM.

    Uses RAGChat to extract invoice information with fallback to regex parsing.

    Example:
        extractor = InvoiceExtractor(chat)
        invoices = extractor.extract(chunks)
    """

    def __init__(self, chat: RAGChat):
        """Initialize extractor.

        Args:
            chat: RAGChat instance for LLM calls
        """
        self.chat = chat

    def extract(self, chunks: list[Chunk]) -> list[Invoice]:
        """Extract invoices from list of chunks.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of extracted Invoice objects
        """
        invoices = []
        for chunk in chunks:
            invoice = self._extract_from_chunk(chunk)
            if invoice:
                invoices.append(invoice)
        return invoices

    def _extract_from_chunk(self, chunk: Chunk) -> Optional[Invoice]:
        """Extract invoice from single chunk using LLM.

        Args:
            chunk: Text chunk to process

        Returns:
            Invoice object or None if extraction fails
        """
        prompt = f"""Extract invoice/receipt information from this text. Return ONLY valid JSON matching this schema:
{{
  "vendor": "string",
  "invoice_number": "string or null",
  "date": "string (YYYY-MM-DD) or null",
  "due_date": "string (YYYY-MM-DD) or null",
  "subtotal": "number or null",
  "tax": "number or null",
  "total": "number or null",
  "currency": "string (default USD)",
  "line_items": [{{"description": "string", "quantity": "number or null", "unit_price": "number or null", "total": "number or null"}}],
  "category": "string or null (e.g., meals, transport, software, office, travel)",
  "payment_method": "string or null"
}}

Text (page {chunk['page']}, chunk {chunk['chunk']}):
{chunk['text'][:3000]}"""

        response = self.chat._call_ollama(prompt, stream=False)
        try:
            data = json.loads(response)
            data["source_page"] = chunk["page"]
            return Invoice(**data)
        except (json.JSONDecodeError, TypeError):
            return self._fallback_extract(chunk)

    def _fallback_extract(self, chunk: Chunk) -> Optional[Invoice]:
        """Fallback regex-based extraction when LLM fails.

        Args:
            chunk: Text chunk to process

        Returns:
            Invoice object with best-effort extraction
        """
        text = chunk["text"]
        vendor = "Unknown Vendor"
        total = None

        # Try to find vendor (often at top)
        lines = text.split("\n")
        for line in lines[:5]:
            if line.strip() and len(line) > 3:
                vendor = line.strip()
                break

        # Try to find total
        total_matches = re.findall(r"(?i)(?:total|amount\s+due|balance\s+due)[:\s]*[$€£]?\s*(\d+[.,]\d{2})", text)
        if total_matches:
            total = float(total_matches[-1].replace(",", ""))

        return Invoice(
            vendor=vendor,
            total=total,
            source_page=chunk["page"],
        )