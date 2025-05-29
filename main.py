from pathlib import Path
import fitz #PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb  # store embedding in each chunk
    return chunks

import chromadb

def store_in_chromadb(chunks, persist_directory="db"):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="policy_chunks")

    documents = [chunk["content"] for chunk in chunks]
    ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"Stored {len(documents)} chunks in ChromaDB.")

def query_chromadb(query, model_name="all-MiniLM-L6-v2", persist_directory="db", k=3):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query)

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="policy_chunks")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results

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
    import sys

    # If you want to rebuild everything, uncomment this:
    # all_docs = load_documents("data/")
    # all_chunks = chunk_documents(all_docs)
    # embedded_chunks = embed_chunks(all_chunks)
    # store_in_chromadb(embedded_chunks)

    print("Ask a question about the documents (type 'exit' to quit):")
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            sys.exit()

        results = query_chromadb(user_input)
        print("\n Top Results:")
        for i, doc in enumerate(results["documents"][0]):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.strip())

