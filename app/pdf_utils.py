# PDF parsing logic using PyMuPDF
import fitz  # PyMuPDF
from typing import List

def extract_text_from_pdf(file_path: str) -> List[str]:
    """
    Extracts text from each page of the PDF.
    Returns a list of page texts.
    """
    doc = fitz.open(file_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages
