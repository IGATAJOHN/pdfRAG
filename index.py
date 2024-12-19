import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from ollama import Client

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Path to knowledge base and FAISS index
KB_DIR = 'knowledge_base'
INDEX_FILE = 'knowledge_base.index'

# Ensure knowledge_base directory exists
os.makedirs(KB_DIR, exist_ok=True)

# Initialize Ollama client
client = Client()

# Function to load documents and create embeddings
def create_index():
    documents = []
    file_paths = []

    # Read PDF files
    for file_name in os.listdir(KB_DIR):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(KB_DIR, file_name)
            try:
                reader = PdfReader(file_path)
                content = ''
                for page in reader.pages:
                    content += page.extract_text() or ''
                if content.strip():
                    documents.append(content)
                    file_paths.append(file_path)
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")

    # Check if there are documents
    if not documents:
        st.warning("No valid documents found in the knowledge base.")
        return

    # Create embeddings for the documents
    doc_embeddings = embedder.encode(documents, convert_to_tensor=True).cpu().numpy()

    # Create and save a FAISS index
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    faiss.write_index(index, INDEX_FILE)

    st.success("FAISS index created and saved successfully!")

# Function to perform RAG and generate response
def generate_response(query, top_k=3):
    if not os.path.exists(INDEX_FILE):
        st.error("FAISS index not found. Please create the index first.")
        return

    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)

    # Load documents
    documents = []
    for file_name in os.listdir(KB_DIR):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(KB_DIR, file_name)
            try:
                reader = PdfReader(file_path)
                content = ''
                for page in reader.pages:
                    content += page.extract_text() or ''
                if content.strip():
                    documents.append(content)
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")

    # Embed the query
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().numpy()

    # Search for similar documents
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve top-k relevant documents
    retrieved_docs = [documents[i] for i in indices[0] if i < len(documents)]
    context = "\n\n".join(retrieved_docs)

    # Generate response using Ollama
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.generate(model='llama2-custom', prompt=prompt)

    return response['response']

# Streamlit UI
st.title("PDF Knowledge Base with RAG using LLaMA")

# Sidebar for PDF upload
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)

if st.sidebar.button("Upload and Save PDFs"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(KB_DIR, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
        st.sidebar.success("PDFs uploaded successfully!")
    else:
        st.sidebar.warning("Please upload at least one PDF file.")

# Button to create index
if st.sidebar.button("Create FAISS Index"):
    create_index()

# Query input and RAG operation
st.header("Ask a Question")
query = st.text_input("Enter your question:")

if st.button("Get Response"):
    if query:
        with st.spinner("Generating response..."):
            response = generate_response(query)
            if response:
                st.write("### Response:")
                st.write(response)
    else:
        st.warning("Please enter a question to get a response.")
