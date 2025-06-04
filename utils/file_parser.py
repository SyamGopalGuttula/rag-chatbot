import fitz  # PyMuPDF
from docx import Document

def extract_text_from_file(filepath: str) -> str:
    if filepath.lower().endswith(".pdf"):
        return extract_pdf_text(filepath)
    elif filepath.lower().endswith(".txt"):
        return extract_txt_text(filepath)
    elif filepath.lower().endswith(".docx"):
        return extract_docx_text(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

def extract_pdf_text(filepath: str) -> str:
    doc = fitz.open(filepath)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        text += f"\n\n--- Page {page_num} ---\n"
        text += page.get_text()
    doc.close()
    return text

def extract_txt_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def extract_docx_text(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
