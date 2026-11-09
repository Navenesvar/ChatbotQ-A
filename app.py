"""
PDF Q&A Chatbot using RAG (Retrieval-Augmented Generation)
- Embeddings: sentence-transformers (local, free)
- Vector Store: FAISS (local, free)
- LLM: Groq API (free tier, no credit card needed)
- UI: Streamlit
"""

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq
import os

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="PDF Q&A Chatbot (RAG)", page_icon="📄", layout="wide")

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks
TOP_K = 3              # number of chunks to retrieve

# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def load_embedding_model():
    """Load the local embedding model (downloaded once, cached)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


# ---------------------------
# Helper functions
# ---------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks for better context retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_faiss_index(chunks, embed_model):
    """Embed chunks and build a FAISS index for similarity search."""
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def retrieve_relevant_chunks(query, chunks, index, embed_model, top_k=TOP_K):
    """Find the most relevant chunks for a given query."""
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def generate_answer(client, query, context_chunks):
    """Send the query + retrieved context to the LLM (Groq) for an answer."""
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context from a PDF document.
If the answer is not in the context, say "I couldn't find that information in the document."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📄 PDF Q&A Chatbot (RAG)")
st.caption("Upload a PDF, ask questions, and get answers grounded in the document — "
           "powered by local embeddings (sentence-transformers + FAISS) and Groq's free LLM API.")

# Sidebar: API key + setup
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Groq API Key", type="password",
                             help="Get a free key at https://console.groq.com/keys")
    st.markdown("---")
    st.markdown("""
    **How it works (RAG):**
    1. PDF text is split into chunks
    2. Each chunk is converted into an embedding (vector)
    3. Your question is embedded too
    4. FAISS finds the most similar chunks
    5. Those chunks + your question are sent to the LLM
    6. LLM answers based on that context
    """)

# Session state init
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("🔍 Process PDF"):
        with st.spinner("Extracting text and building search index..."):
            text = extract_text_from_pdf(uploaded_file)

            if not text.strip():
                st.error("Couldn't extract text from this PDF. It might be a scanned/image-based PDF.")
            else:
                chunks = chunk_text(text)
                embed_model = load_embedding_model()
                index, _ = build_faiss_index(chunks, embed_model)

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.chat_history = []

                st.success(f"✅ Processed! Document split into {len(chunks)} chunks. You can now ask questions.")

# Q&A section
if st.session_state.chunks is not None:
    st.markdown("---")
    st.subheader("💬 Ask a question about the document")

    query = st.text_input("Your question:")

    if st.button("Get Answer") and query:
        if not api_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                embed_model = load_embedding_model()
                relevant_chunks = retrieve_relevant_chunks(
                    query, st.session_state.chunks, st.session_state.index, embed_model
                )

                try:
                    client = get_groq_client(api_key)
                    answer = generate_answer(client, query, relevant_chunks)
                except Exception as e:
                    answer = f"Error calling Groq API: {e}"

                st.session_state.chat_history.append((query, answer, relevant_chunks))

    # Display chat history (most recent first)
    for q, a, retrieved in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Answer:** {a}")
        with st.expander("📌 View retrieved context chunks"):
            for i, chunk in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {i}:**")
                st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
        st.markdown("---")
else:
    st.info("👆 Upload a PDF and click 'Process PDF' to get started.")
