import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis as RedisVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Access secrets securely using Streamlit secrets
google_api_key = st.secrets["google"]["api_key"]
redis_url = st.secrets["redis"]["url"]
index_name = st.secrets["redis"]["index_name"]

# Set the API Key for Gemini (from Streamlit secrets)
os.environ["GOOGLE_API_KEY"] = google_api_key

# Redis connection and model settings (using Streamlit secrets for security)
REDIS_URL = redis_url
INDEX_NAME = index_name

# Initialize the Gemini LLM and Embedding model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Set up Streamlit page
st.set_page_config(page_title="ChatPDF with Gemini", layout="centered")
st.title("🤖 AI-Powered PDF Assistant")

# Initialize QA Chain (only once)
if "qa_chain" not in st.session_state:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split PDF into chunks
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Initialize Redis Vector DB with the documents
        vectorstore = RedisVectorStore.from_documents(
            documents=chunks,
            embedding=embedding,
            redis_url=REDIS_URL,
            index_name=INDEX_NAME
        )

        # Setup memory for the conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Explicitly set the output key for memory
        )

        # Initialize the ConversationalRetrievalChain
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )

        st.session_state.messages = []
        st.success("✅ PDF processed. You can now start chatting!")

# Show chat interface if PDF is uploaded and processed
if "qa_chain" in st.session_state:
    # Display the conversation history (chat messages)
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Get user input for a question
    prompt = st.chat_input("Ask a question about your PDF...")
    if prompt:
        # Append the user's question
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Get the response from the ConversationalRetrievalChain
        response = st.session_state.qa_chain({"question": prompt})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")

        # Append the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
