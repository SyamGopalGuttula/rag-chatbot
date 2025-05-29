from pathlib import Path
import fitz #PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(data_path: str):
    docs = []
    data_dir = Path(data_path)

    for file_path in data_dir.glob("*"):
        if file_path.suffix == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_path.suffix == ".txt":
            text = file_path.read_texg(encoding="utf-8")
        else:
            continue

        docs.append({"content": text, "source": str(file_path.name)})
    return docs

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

if __name__ == "__main__":
    all_docs = load_documents("data/")
    print(f"Loaded {len(all_docs)} documents.")
    print("Previes of first document:")
    print(all_docs[0]["content"][:1000]) #print first 1000 characters