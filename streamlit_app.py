import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# === Load secrets ===
load_dotenv()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag-policy-docs")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# === Init LangChain RAG pipeline ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory="vectorstore/chroma"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    temperature=0.3,
    model=LLM_MODEL_NAME,
    openai_api_base=LLM_API_BASE,
    openai_api_key=LLM_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False  # Can toggle True later to show metadata
)

# === Streamlit UI ===
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 Internal Policy RAG Chatbot")
st.markdown("Ask a question about FCRA, privacy, cybersecurity, or internal policy.")

query = st.text_input("🔍 Your question:", placeholder="e.g. What does FCRA say about disputed credit info?")

if query:
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke(query)
            st.markdown("### ✅ Answer:")
            st.write(response["result"] if isinstance(response, dict) else response)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
