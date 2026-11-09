# PDF Q&A Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about any PDF you upload. Built entirely with free tools — great for a resume project.
Link : https://chatbotq-a-tottph7optf2waoopsargv.streamlit.app/
## Tech Stack
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) — runs locally, free
- **Vector Search**: FAISS — local similarity search
- **LLM**: Groq API (free tier, Llama 3.1 8B) — no credit card required
- **PDF Parsing**: pypdf
- **UI**: Streamlit

## How RAG Works Here
1. PDF text is extracted and split into overlapping chunks (~500 chars).
2. Each chunk is converted into a vector embedding using a sentence-transformer model.
3. The embeddings are stored in a FAISS index for fast similarity search.
4. When you ask a question, it's also embedded, and FAISS finds the top-3 most similar chunks.
5. Those chunks are inserted into a prompt as "context" and sent to the LLM (Groq).
6. The LLM answers based only on that retrieved context, reducing hallucination.

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
- Go to https://console.groq.com/keys
- Sign up (free, no credit card needed)
- Create an API key

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use it
- Paste your Groq API key in the sidebar
- Upload a PDF
- Click "Process PDF"
- Ask questions!

## Resume Line (suggested)
> Built a PDF Q&A chatbot using Retrieval-Augmented Generation (RAG) — implemented text chunking, semantic embeddings (sentence-transformers), FAISS vector search, and LLM-based answer generation via Groq API, with a Streamlit interface.

## Possible Extensions (talk about these in interviews)
- Support multiple PDFs / persistent vector store (e.g., ChromaDB)
- Add source citations (which chunk/page the answer came from)
- Conversation memory for follow-up questions
- Swap embedding model or LLM provider easily (modular design)
- Add evaluation: precision@k for retrieval quality
