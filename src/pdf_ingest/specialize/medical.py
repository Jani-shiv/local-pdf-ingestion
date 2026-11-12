"""Medical report extractor: patient-friendly summaries with flagged values."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Optional

from ..chat import RAGChat
from ..chunk import Chunk


@dataclass
class LabValue:
    name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: Optional[str] = None  # normal, high, low, critical
    flag: bool = False


@dataclass
class MedicalReport:
    patient_name: Optional[str] = None
    report_date: Optional[str] = None
    provider: Optional[str] = None
    report_type: Optional[str] = None  # blood work, imaging, pathology, etc.
    summary: Optional[str] = None
    lab_values: list[LabValue] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    flagged_items: list[str] = field(default_factory=list)
    source_page: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        lines = ["# Medical Report Summary"]
        if self.patient_name:
            lines.append(f"**Patient:** {self.patient_name}")
        if self.report_date:
            lines.append(f"**Date:** {self.report_date}")
        if self.provider:
            lines.append(f"**Provider:** {self.provider}")
        if self.report_type:
            lines.append(f"**Type:** {self.report_type}")

        if self.summary:
            lines.append(f"\n## Summary\n{self.summary}")

        if self.lab_values:
            lines.append("\n## Lab Results")
            lines.append("| Test | Value | Unit | Reference Range | Status |")
            lines.append("|---|---:|---|---|---:|")
            for lv in self.lab_values:
                flag = " ⚠️" if lv.flag else ""
                lines.append(f"| {lv.name} | {lv.value} | {lv.unit or ''} | {lv.reference_range or ''} | {lv.status or ''}{flag} |")

        if self.diagnoses:
            lines.append("\n## Diagnoses")
            for d in self.diagnoses:
                lines.append(f"- {d}")

        if self.medications:
            lines.append("\n## Medications")
            for m in self.medications:
                lines.append(f"- {m}")

        if self.recommendations:
            lines.append("\n## Recommendations")
            for r in self.recommendations:
                lines.append(f"- {r}")

        if self.flagged_items:
            lines.append("\n## ⚠️ Items Requiring Attention")
            for f in self.flagged_items:
                lines.append(f"- {f}")

        lines.append("\n---\n*Disclaimer: This is an AI-generated summary. Always consult your healthcare provider for medical advice.*")
        return "\n".join(lines)


class MedicalExtractor:
    def __init__(self, chat: RAGChat):
        self.chat = chat

    def extract(self, chunks: list[Chunk]) -> list[MedicalReport]:
        reports = []
        for chunk in chunks:
            report = self._extract_from_chunk(chunk)
            if report:
                reports.append(report)
        return reports

    def _extract_from_chunk(self, chunk: Chunk) -> Optional[MedicalReport]:
        prompt = f"""Extract medical report information from this text. Return ONLY valid JSON matching this schema:
{{
  "patient_name": "string or null",
  "report_date": "string (YYYY-MM-DD) or null",
  "provider": "string or null",
  "report_type": "string or null",
  "summary": "string (patient-friendly, 3-5 sentences)",
  "lab_values": [
    {{"name": "string", "value": "string", "unit": "string or null", "reference_range": "string or null", "status": "normal|high|low|critical or null"}}
  ],
  "diagnoses": ["string"],
  "medications": ["string"],
  "recommendations": ["string"],
  "flagged_items": ["string (values outside normal range or needing follow-up)"]
}}

Text (page {chunk['page']}, chunk {chunk['chunk']}):
{chunk['text'][:4000]}"""

        response = self.chat._call_ollama(prompt, stream=False)
        try:
            data = json.loads(response)
            data["source_page"] = chunk["page"]
            report = MedicalReport(**data)
            # Auto-flag based on status
            for lv in report.lab_values:
                    if isinstance(lv, dict):
                        status = lv.get("status")
                    else:
                        status = getattr(lv, "status", None)
                    if status in ("high", "low", "critical"):
                        if isinstance(lv, dict):
                            lv["flag"] = True
                            name = lv.get("name", "")
                            value = lv.get("value", "")
                            unit = lv.get("unit", "")
                        else:
                            lv.flag = True
                            name = lv.name
                            value = lv.value
                            unit = lv.unit or ""
                        report.flagged_items.append(f"{name}: {value} {unit} ({status})")
            return report
        except (json.JSONDecodeError, TypeError):
            return self._fallback_extract(chunk)

    def _fallback_extract(self, chunk: Chunk) -> Optional[MedicalReport]:
        text = chunk["text"]
        lab_values = []

        # Simple regex for lab values
        pattern = r"([A-Za-z\s]+)\s+(\d+\.?\d*)\s*(\w+)?\s*\(?([\d\.\-\s]+)\)?"
        for match in re.finditer(pattern, text):
            name, value, unit, ref = match.groups()
            if any(kw in name.lower() for kw in ["glucose", "cholesterol", "hemoglobin", "wbc", "rbc", "platelet", "creatinine", "bun", "alt", "ast", "tsh", "vitamin"]):
                lab_values.append(LabValue(
                    name=name.strip(),
                    value=value,
                    unit=unit,
                    reference_range=ref.strip("()") if ref else None,
                ))

        return MedicalReport(
            lab_values=lab_values[:20],
            source_page=chunk["page"],
        )