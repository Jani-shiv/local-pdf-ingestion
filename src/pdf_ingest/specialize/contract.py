"""Contract analyzer: clause extraction, risk flags, obligation calendar.

This module provides:
- Contract, Clause, Obligation dataclasses with serialization
- ContractExtractor: LLM-based extraction from text chunks

Example:
    extractor = ContractExtractor(chat)
    contracts = extractor.extract(chunks)
    for c in contracts:
        print(c.to_markdown())
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Optional

from ..chat import RAGChat
from ..chunk import Chunk


@dataclass
class Clause:
    """Contract clause with risk assessment.

    Attributes:
        title: Clause title/heading
        text: Full clause text
        category: Clause category (payment, termination, liability, ip, confidentiality, etc.)
        risk_level: "low", "medium", "high", or "critical"
        page: Source page number
    """
    title: str
    text: str
    category: str  # payment, termination, liability, ip, confidentiality, etc.
    risk_level: str  # low, medium, high, critical
    page: int


@dataclass
class Obligation:
    """Contractual obligation with due date.

    Attributes:
        description: Obligation description
        party: Responsible party - "me", "them", or "both"
        due_date: Due date (YYYY-MM-DD)
        frequency: "one-time", "monthly", "quarterly", "annually"
        clause_ref: Reference to originating clause
    """
    description: str
    party: str  # me, them, both
    due_date: Optional[str] = None
    frequency: Optional[str] = None  # one-time, monthly, quarterly, annually
    clause_ref: Optional[str] = None


@dataclass
class Contract:
    """Structured contract analysis with clauses, obligations, and risk flags.

    Attributes:
        title: Contract title
        parties: List of party names
        effective_date: Contract start date (YYYY-MM-DD)
        expiration_date: Contract end date (YYYY-MM-DD)
        contract_type: Type (service, employment, nda, lease, etc.)
        clauses: List of Clause objects
        obligations: List of Obligation objects
        risk_flags: List of risk warning strings
        key_terms: Dictionary of key terms
        source_page: Source document page number
    """
    title: str
    parties: list[str] = field(default_factory=list)
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    contract_type: Optional[str] = None  # service, employment, nda, lease, etc.
    clauses: list[Clause] = field(default_factory=list)
    obligations: list[Obligation] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    key_terms: dict = field(default_factory=dict)
    source_page: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        """Convert to Markdown for display."""
        lines = [f"# Contract Analysis: {self.title}"]
        if self.parties:
            lines.append(f"**Parties:** {', '.join(self.parties)}")
        if self.effective_date:
            lines.append(f"**Effective:** {self.effective_date}")
        if self.expiration_date:
            lines.append(f"**Expires:** {self.expiration_date}")
        if self.contract_type:
            lines.append(f"**Type:** {self.contract_type}")

        if self.risk_flags:
            lines.append("\n## ⚠️ Risk Flags")
            for flag in self.risk_flags:
                lines.append(f"- {flag}")

        if self.clauses:
            lines.append("\n## Key Clauses")
            for clause in self.clauses:
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(clause.risk_level, "")
                lines.append(f"### {clause.title} {risk_emoji} ({clause.category})")
                lines.append(f"*Page {clause.page}*")
                lines.append(f"{clause.text[:500]}...")
                lines.append("")

        if self.obligations:
            lines.append("\n## Obligations Calendar")
            lines.append("| Due | Frequency | Party | Description |")
            lines.append("|---|---|---|---|")
            for obl in sorted(self.obligations, key=lambda x: x.due_date or "9999-12-31"):
                lines.append(f"| {obl.due_date or 'TBD'} | {obl.frequency or 'One-time'} | {obl.party} | {obl.description} |")

        return "\n".join(lines)


class ContractExtractor:
    """Extract structured contract data from text chunks using LLM.

    Uses RAGChat to extract clauses, obligations, and risk flags with
    automatic risk flagging.

    Example:
        extractor = ContractExtractor(chat)
        contracts = extractor.extract(chunks)
    """

    def __init__(self, chat: RAGChat):
        """Initialize extractor.

        Args:
            chat: RAGChat instance for LLM calls
        """
        self.chat = chat

    def extract(self, chunks: list[Chunk]) -> list[Contract]:
        """Extract contracts from list of chunks.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of extracted Contract objects
        """
        contracts = []
        for chunk in chunks:
            contract = self._extract_from_chunk(chunk)
            if contract:
                contracts.append(contract)
        return contracts

    def _extract_from_chunk(self, chunk: Chunk) -> Optional[Contract]:
        """Extract contract from single chunk using LLM.

        Args:
            chunk: Text chunk to process

        Returns:
            Contract object or None if extraction fails
        """
        prompt = f"""Analyze this contract text. Return ONLY valid JSON matching this schema:
{{
  "title": "string",
  "parties": ["string"],
  "effective_date": "string (YYYY-MM-DD) or null",
  "expiration_date": "string (YYYY-MM-DD) or null",
  "contract_type": "string or null",
  "clauses": [
    {{"title": "string", "text": "string", "category": "payment|termination|liability|ip|confidentiality|indemnification|force_majeure|governing_law|other", "risk_level": "low|medium|high|critical"}}
  ],
  "obligations": [
    {{"description": "string", "party": "me|them|both", "due_date": "string (YYYY-MM-DD) or null", "frequency": "one-time|monthly|quarterly|annually or null", "clause_ref": "string or null"}}
  ],
  "risk_flags": ["string"],
  "key_terms": {{}}
}}

Text (page {chunk['page']}, chunk {chunk['chunk']}):
{chunk['text'][:4000]}"""

        response = self.chat._call_ollama(prompt, stream=False)
        try:
            data = json.loads(response)
            data["source_page"] = chunk["page"]
            contract = Contract(**data)
            self._auto_flag_risks(contract)
            return contract
        except (json.JSONDecodeError, TypeError):
            return self._fallback_extract(chunk)

    def _auto_flag_risks(self, contract: Contract) -> None:
        """Automatically flag common contract risks.

        Args:
            contract: Contract to analyze
        """
        risk_keywords = {
            "auto-renewal": "Automatic renewal without notice",
            "unlimited liability": "Unlimited liability exposure",
            "indemnification": "Broad indemnification clause",
            "non-compete": "Restrictive non-compete",
            "ip assignment": "IP assignment without compensation",
            "termination for convenience": "Can be terminated without cause",
            "data ownership": "Vendor owns your data",
            "price increase": "Unilateral price increase right",
        }

        full_text = " ".join(
                c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                for c in contract.clauses
            ).lower()
        for keyword, description in risk_keywords.items():
            if keyword in full_text:
                contract.risk_flags.append(description)

    def _fallback_extract(self, chunk: Chunk) -> Optional[Contract]:
        """Fallback regex-based extraction when LLM fails.

        Args:
            chunk: Text chunk to process

        Returns:
            Contract object with minimal extraction
        """
        text = chunk["text"]
        title = "Contract"
        title_match = re.search(r"(?i)(?:agreement|contract|mou|nda|lease|service\s+order)[:\s]*([A-Z][A-Za-z\s]+)", text)
        if title_match:
            title = title_match.group(1).strip()

        return Contract(
            title=title,
            source_page=chunk["page"],
        )