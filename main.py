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

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(splits):
            chunks.append({
                "content": chunk,
                "source": doc["source"],
                "chunk_id": i
            })
    return chunks


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

if __name__ == "__main__":
    all_docs = load_documents("data/")
    print(f"Loaded {len(all_docs)} documents.")

    all_chunks = chunk_documents(all_docs)
    print(f"Created {len(all_chunks)} chunks.")
    print("Preview of first chunk:")
    print(all_chunks[0]["content"])
