import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# === Config ===
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE")
MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# === Initialize ===
client = PersistentClient(path="vectorstore/chroma")
collection = client.get_or_create_collection(name=COLLECTION_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

llm_client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_API_BASE
)

# === Functions ===

def retrieve_context(query, k=5):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return "\n\n".join(results["documents"][0]) if results["documents"] else ""

def ask_llm(question, context):
    prompt = f"""You are a helpful assistant for internal company policy. Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

# === CLI Loop ===
if __name__ == "__main__":
    print("Ask your company policy questions (type 'exit' to quit)")

    while True:
        user_q = input("\nQuestion: ")
        if user_q.lower() in {"exit", "quit"}:
            break

        context = retrieve_context(user_q)
        if not context:
            print("No relevant context found.")
            continue

        answer = ask_llm(user_q, context)
        print(f"\nAnswer:\n{answer}")
