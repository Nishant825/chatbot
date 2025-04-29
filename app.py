import streamlit as st
import fitz  # PyMuPDF
import redis
import uuid
import numpy as np
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---------- CONFIGURATION ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyC23ApPsVXGoVkX9KwsDWk_PPXv6T6K1aA")
REDIS_HOST = "redis-16132.crce182.ap-south-1-1.ec2.redns.redis-cloud.com"
REDIS_PORT = 16132
VECTOR_DIM = 768  # Gemini embedding size

# Initialize Redis connection
redis_conn = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password="UMuw76rrG4fo9bgUQD1wn8v8B6GbZB6t",
    db=0,
)

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyC23ApPsVXGoVkX9KwsDWk_PPXv6T6K1aA")

# ---------- HELPER FUNCTIONS ----------
def extract_text_from_pdf(pdf_file):
    """Extract full text from a PDF."""
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_text(text):
    """Generate embedding for text using Gemini."""
    return embeddings.embed_query(text)

def store_embeddings(chunks):
    """Store text chunks and their embeddings in Redis."""
    for chunk in chunks:
        embedding = np.array(embed_text(chunk), dtype=np.float32)
        doc_id = str(uuid.uuid4())
        redis_conn.hset(f"doc:{doc_id}", mapping={
            "text": chunk,
            "embedding": embedding.tobytes()
        })
        redis_conn.sadd("doc_ids", f"doc:{doc_id}")

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_similar_chunks(query, top_k=5):
    """Find top-K similar text chunks from Redis."""
    query_vec = np.array(embed_text(query), dtype=np.float32)
    results = []
    for doc_id in redis_conn.smembers("doc_ids"):
        doc = redis_conn.hgetall(doc_id)
        text = doc[b'text'].decode()
        embedding = np.frombuffer(doc[b'embedding'], dtype=np.float32)
        sim = cosine_similarity(query_vec, embedding)
        results.append((sim, text))
    results.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in results[:top_k]]

def summarize_with_gemini(chunks, query):
    """Generate a summary or answer using Gemini-Pro model."""
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Use the following document excerpts to answer the question:

Context:
{context}

Question: {query}

Give a helpful, concise answer."""
    
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Chat with Your PDF (Gemini + Redis)", layout="wide")
st.title("📄 Chat with Your Resume / PDF using Gemini (Free)")

# Check if PDF has been uploaded and indexed
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a PDF", type=["pdf"])

if uploaded_file and not st.session_state.indexed:
    with st.spinner("🔍 Extracting and indexing document..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        store_embeddings(chunks)
        st.session_state.indexed = True
    st.success("✅ PDF uploaded and indexed!")

# Ask a question
query = st.text_input("💬 Ask a question about the uploaded document:")

if query:
    with st.spinner("🤖 Thinking..."):
        # st.text_input = ""
        context = search_similar_chunks(query)
        answer = summarize_with_gemini(context, query)

        st.subheader("📘 Answer:")
        st.write(answer)
