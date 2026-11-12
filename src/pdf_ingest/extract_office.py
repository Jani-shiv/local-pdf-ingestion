"""Office document extraction: DOCX, XLSX, PPTX."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None


@dataclass
class OfficeContent:
    source: str
    page: int
    text: str
    metadata: dict


def extract_docx(path: str | Path) -> Generator[OfficeContent, None, None]:
    if Document is None:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")
    doc = Document(path)
    full_text = []
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            full_text.append(para.text)
    yield OfficeContent(
        source=str(path),
        page=1,
        text="\n\n".join(full_text),
        metadata={"paragraphs": len(doc.paragraphs)},
    )


def extract_xlsx(path: str | Path) -> Generator[OfficeContent, None, None]:
    if openpyxl is None:
        raise RuntimeError("openpyxl not installed. Run: pip install openpyxl")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for sheet_idx, sheet_name in enumerate(wb.sheetnames, start=1):
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            row_text = " | ".join(str(c) for c in row if c is not None)
            if row_text.strip():
                rows.append(row_text)
        if rows:
            yield OfficeContent(
                source=str(path),
                page=sheet_idx,
                text="\n".join(rows),
                metadata={"sheet": sheet_name, "rows": len(rows)},
            )


def extract_pptx(path: str | Path) -> Generator[OfficeContent, None, None]:
    if Presentation is None:
        raise RuntimeError("python-pptx not installed. Run: pip install python-pptx")
    prs = Presentation(path)
    for slide_idx, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        texts.append(para.text)
        if texts:
            yield OfficeContent(
                source=str(path),
                page=slide_idx,
                text="\n".join(texts),
                metadata={"slide": slide_idx, "shapes": len(slide.shapes)},
            )


def extract_office(path: str | Path) -> list[OfficeContent]:
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        return list(extract_docx(path))
    elif suffix == ".xlsx":
        return list(extract_xlsx(path))
    elif suffix == ".pptx":
        return list(extract_pptx(path))
    else:
        raise ValueError(f"Unsupported format: {suffix}")