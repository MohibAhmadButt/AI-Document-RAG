# 📚 Document RAG Engine

An enterprise-ready, completely free-to-host Retrieval-Augmented Generation (RAG) application. This tool allows users to upload unstructured PDF documents, instantly processes and chunks the text, computes semantic embeddings, and builds a local vector store to provide accurate, context-aware answers to user queries using high-speed LLMs.

By offloading embeddings and language processing to the Groq API, this application remains incredibly lightweight and fast, running seamlessly without requiring massive machine learning frameworks (like PyTorch) locally.

## 🚀 Features
* **PDF Ingestion & Text Extraction:** Dynamically reads uploaded PDFs and extracts text elements natively.
* **Semantic Vector Search:** Tokenizes and splits text into optimized chunks using a recursive character text splitter, then generates mathematical vector embeddings.
* **In-Memory Vector Database:** Uses FAISS (Facebook AI Similarity Search) to perform ultra-fast similarity lookups on local vector spaces.
* **No Local ML Overhead:** Fully relies on Groq cloud architecture for both embeddings and reasoning, keeping local deployment footprints minimal.
* **Context Auditing:** Transparency feature allowing developers to audit the exact text chunks extracted from the document that informed the AI's final answer.

## 🛠️ Tech Stack
* **Frontend Interface:** [Streamlit](https://streamlit.io/)
* **Orchestration Framework:** [LangChain](https://www.langchain.com/)
* **Vector Database:** [FAISS (CPU)](https://github.com/facebookresearch/faiss)
* **LLM Provider:** [Groq](https://groq.com/) (Model: `llama-3.1-8b-instant`)
* **Embedding Model:** [Groq / Nomic](https://groq.com/) (Model: `nomic-embed-text-v1.5`)
* **Document Parsing:** `PyPDF`
