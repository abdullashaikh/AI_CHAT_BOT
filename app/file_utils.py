# app/file_utils.py
import pytesseract
from PIL import Image
import pdfplumber
from docx import Document
from pathlib import Path

def extract_text_from_file(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_text_from_pdf(filepath: str) -> str:
    with pdfplumber.open(filepath) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def extract_text_from_image(filepath: str) -> str:
    image = Image.open(filepath)
    return pytesseract.image_to_string(image)

def extract_text_from_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
