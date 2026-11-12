"""OCR extraction for scanned PDFs and images using Tesseract or PaddleOCR."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


@dataclass
class OCRResult:
    page: int
    text: str
    confidence: float
    language: str


class OCRExtractor:
    def __init__(
        self,
        engine: str = "tesseract",
        languages: str = "eng",
        paddle_lang: str = "en",
    ):
        self.engine = engine
        self.languages = languages

        if engine == "paddle":
            if PaddleOCR is None:
                raise RuntimeError("paddleocr not installed. Run: pip install paddleocr")
            self.paddle = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
        elif engine == "tesseract":
            if pytesseract is None:
                raise RuntimeError("pytesseract not installed. Run: pip install pytesseract")
            # Verify tesseract is available
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                raise RuntimeError("Tesseract not found. Install: https://github.com/tesseract-ocr/tesseract")

    def extract_pdf(self, pdf_path: str | Path) -> Generator[OCRResult, None, None]:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield from self._extract_image(img, page_num)

    def extract_image(self, image_path: str | Path) -> Generator[OCRResult, None, None]:
        img = Image.open(image_path)
        yield from self._extract_image(img, 1)

    def _extract_image(self, img, page: int) -> Generator[OCRResult, None, None]:
        if self.engine == "tesseract":
            data = pytesseract.image_to_data(
                img,
                lang=self.languages,
                output_type=pytesseract.Output.DICT,
            )
            texts = [t for t in data["text"] if t.strip()]
            confidences = [c for c, t in zip(data["conf"], data["text"]) if t.strip()]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            text = " ".join(texts)
            yield OCRResult(page=page, text=text, confidence=avg_conf / 100, language=self.languages)

        elif self.engine == "paddle":
            result = self.paddle.ocr(img, cls=True)
            texts = []
            confidences = []
            for line in result:
                for word in line:
                    texts.append(word[1][0])
                    confidences.append(word[1][1])
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            text = " ".join(texts)
            yield OCRResult(page=page, text=text, confidence=avg_conf, language=self.paddle.lang)


def extract_pdf_ocr(
    pdf_path: str | Path,
    engine: str = "tesseract",
    languages: str = "eng",
) -> list[OCRResult]:
    extractor = OCRExtractor(engine=engine, languages=languages)
    return list(extractor.extract_pdf(pdf_path))


def extract_image_ocr(
    image_path: str | Path,
    engine: str = "tesseract",
    languages: str = "eng",
) -> list[OCRResult]:
    extractor = OCRExtractor(engine=engine, languages=languages)
    return list(extractor.extract_image(image_path))