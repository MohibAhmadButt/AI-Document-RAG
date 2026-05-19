import streamlit as st
from langchain_groq import ChatGroq
# CHANGED: Import the correct community embeddings module
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

st.set_page_config(page_title="Document RAG Engine", page_icon="📚")
st.title("📚 PDF Retrieval-Augmented Generation (RAG) Engine")

# 1. Load API Key securely
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Missing GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# 2. Initialize Groq LLM & Local Embeddings
llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")

# CHANGED: Using a free, highly efficient open-source embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# 3. File Uploader UI
uploaded_file = st.file_uploader("Upload a PDF document to query", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily to disk so PyPDFLoader can read it
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Use session state to cache the vector store so it doesn't re-index on every click
    if "vector_store" not in st.session_state:
        with st.spinner("Processing document... Extracting text and generating embeddings..."):
            # Load PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            # Split text into small chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            
            # Create FAISS Vector Store
            st.session_state.vector_store = FAISS.from_documents(final_documents, embeddings)
            st.success("Document successfully processed and indexed!")
            
    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    # 4. Query Interface
    user_query = st.text_input("Ask a question about the document:")
    
    if user_query:
        if "vector_store" in st.session_state:
            with st.spinner("Searching document for answers..."):
                # Define a prompt template
                system_prompt = (
                    "You are an expert document analysis assistant. Answer the user's question "
                    "using only the provided context below. If you do not know the answer based "
                    "on the context, say that you cannot find the answer in the document.\n\n"
                    "Context:\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                # Build the RAG chain
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Execute the chain
                response = rag_chain.invoke({"input": user_query})
                
                # Display Results
                st.subheader("Answer:")
                st.write(response["answer"])
                
                # Show the text chunks that were retrieved
                with st.expander("See Source Context Chunks"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.caption(doc.page_content)
        else:
            st.error("Please wait until the document is fully indexed.")
