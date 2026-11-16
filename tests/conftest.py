import pymupdf
import pytest


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "t.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_textbox(
        pymupdf.Rect(72, 72, 523, 700), "alpha beta gamma. " * 120, fontsize=11
    )
    doc.new_page()  # blank page: should be skipped
    doc.save(path)
    return path
