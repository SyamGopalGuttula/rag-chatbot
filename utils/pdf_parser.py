import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text += f"\n\n--- Page {page_num} ---\n{text}"

    doc.close()
    return full_text
