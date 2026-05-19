# 📚 Document RAG Engine (Modern LCEL Architecture)

An enterprise-ready, production-grade Retrieval-Augmented Generation (RAG) application. This tool allows users to upload unstructured PDF documents, dynamically processes and chunks the raw text, generates semantic vector embeddings, and constructs an in-memory vector space to answer highly complex queries with exact context-driven precision.

This repository leverages **LangChain Expression Language (LCEL)**, bypassing legacy monolithic chain structures to ensure a future-proof, highly optimized execution pipeline.

## 🚀 Features
* **Nativized Document Ingestion:** Processes unstructured multi-page PDF records on the fly using `PyPDF`.
* **Recursive Token Chunking:** Splits text elements using a `RecursiveCharacterTextSplitter` configured for an optimal `1000` token window with a `200` token semantic overlap.
* **In-Memory Vector Search Index:** Builds a high-speed similarity mapping cluster using FAISS (Facebook AI Similarity Search).
* **Modern LCEL Pipeline:** Uses LangChain's decoupled, declarative expression syntax (`inputs | prompt | llm | parser`) for deterministic execution.
* **Context Auditing Engine:** A dedicated UI panel allowing developers and users to view the exact text chunks and page numbers pulled by the retriever to verify factual consistency.

## 🛠️ Tech Stack
* **Frontend Dashboard:** [Streamlit](https://streamlit.io/)
* **Orchestration Architecture:** [LangChain (LCEL Core)](https://www.langchain.com/)
* **Vector Storage Subsystem:** [FAISS (CPU variant)](https://github.com/facebookresearch/faiss)
* **Inference Engine:** [Groq Cloud Ecosystem](https://groq.com/) (Model: `llama-3.1-8b-instant`)
* **Embedding Cluster:** [Hugging Face Transformers](https://huggingface.co/) (Model: `all-MiniLM-L6-v2`)
