# Local RAG Chatbot for Policy Documents

This is a **fully local Retrieval-Augmented Generation (RAG)** chatbot that answers questions about internal policy documents like:

- Credit Reporting Policy
- Privacy Policy
- FCRA Compliance Guide
- Cybersecurity Guidelines
- Code of Conduct

It runs 100% offline using open-source tools and supports PDF or `.txt` files stored in the `data/` folder.

---

## Features

- Load & chunk documents (PDF or text)
- Generate embeddings using `sentence-transformers`
- Vector search via **ChromaDB** (local)
- Answer generation via **Mistral** running on **Ollama**
- Completely local and private (no cloud APIs)

---

## Tech Stack

| Component       | Tool / Library           |
|----------------|--------------------------|
| Document Parser | `PyMuPDF`                |
| Chunking        | `LangChain`'s `RecursiveCharacterTextSplitter` |
| Embedding Model | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB       | `ChromaDB` (local)       |
| LLM             | `Mistral` via `Ollama`   |
| Language        | Python 3.13              |

---

## Folder Structure

```
rag-chatbot/
├── data/              # Input PDFs or text files
├── db/                # ChromaDB storage (auto-generated)
├── venv/              # Python virtual environment
├── main.py            # Main chatbot script
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Setup Instructions

1. **Clone or download this repo**

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Python packages**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and run a model**  
   Download from [https://ollama.com](https://ollama.com)  
   Start Mistral:
   ```bash
   ollama run mistral
   ```

5. **Add your policy documents to `/data`**

6. **Run the chatbot**  
   ```bash
   python main.py
   ```

7. **Ask questions like**:
   ```
   What is the purpose of a credit report?
   ```

---

## Privacy Notice

This project is designed for **educational use**. All data and LLMs are run **locally** with no cloud connections.

---

## Credits

Built using:
- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma](https://github.com/chroma-core/chroma)
- [Ollama](https://github.com/jmorganca/ollama)
- [SentenceTransformers](https://www.sbert.net/)
- [Mistral](https://ollama.com/library/mistral)

---

## Coming Soon

- Streamlit UI
- Citations in responses
- Multi-file summarization