"""Contract analyzer: clause extraction, risk flags, obligation calendar."""
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
    title: str
    text: str
    category: str  # payment, termination, liability, ip, confidentiality, etc.
    risk_level: str  # low, medium, high, critical
    page: int


@dataclass
class Obligation:
    description: str
    party: str  # me, them, both
    due_date: Optional[str] = None
    frequency: Optional[str] = None  # one-time, monthly, quarterly, annually
    clause_ref: Optional[str] = None


@dataclass
class Contract:
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
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
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
    def __init__(self, chat: RAGChat):
        self.chat = chat

    def extract(self, chunks: list[Chunk]) -> list[Contract]:
        contracts = []
        for chunk in chunks:
            contract = self._extract_from_chunk(chunk)
            if contract:
                contracts.append(contract)
        return contracts

    def _extract_from_chunk(self, chunk: Chunk) -> Optional[Contract]:
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
        text = chunk["text"]
        title = "Contract"
        title_match = re.search(r"(?i)(?:agreement|contract|mou|nda|lease|service\s+order)[:\s]*([A-Z][A-Za-z\s]+)", text)
        if title_match:
            title = title_match.group(1).strip()

        return Contract(
            title=title,
            source_page=chunk["page"],
        )