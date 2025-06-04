import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag-policy-docs")

client = PersistentClient(path="vectorstore/chroma")
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")  # CPU-friendly

def embed_and_store(chunks: list[str], source_id: str):
    embeddings = model.encode(chunks).tolist()

    ids = [f"{source_id}-{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": source_id}] * len(chunks)
    )
    print(f"Stored {len(chunks)} chunks into ChromaDB")
