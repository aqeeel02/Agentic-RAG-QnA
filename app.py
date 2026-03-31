import streamlit as st
import os
import pyttsx3
import shutil
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Agentic RAG Chatbot", layout="centered")
st.title("🤖 Agentic RAG Chatbot")

# -------------------- RESET --------------------
if st.button("🔄 Reset"):
    if os.path.exists("data"):
        shutil.rmtree("data")
    st.session_state.clear()
    st.success("Chat and uploaded files have been reset.")

# -------------------- FILE UPLOAD --------------------
st.header("📤 Upload Your Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, PPTX, CSV, or TXT files",
    type=["pdf", "docx", "pptx", "csv", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.success("✅ Files uploaded successfully!")

# -------------------- ASK A QUESTION --------------------
st.header("💬 Ask a Question")
query = st.text_input("Type your question here:")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("🧠 Processing..."):
            # Step 1: Ingest
            ingestion = IngestionAgent()
            ingestion_msg = ingestion.ingest()

            # Step 2: Retrieve
            retrieval = RetrievalAgent()
            retrieval.process_documents(ingestion_msg)
            retrieved_msg = retrieval.retrieve(query)

            # Step 3: Generate Response
            llm = LLMResponseAgent()
            response = llm.generate_response(retrieved_msg)

            print("DEBUG RESPONSE:", response)

            # -------------------- OUTPUT --------------------
            st.subheader("💡 Answer:")
            st.success(response["payload"]["answer"])

            st.subheader("📚 Sources")

            sources = response["payload"].get("sources", [])

            def extract_relevant_sentence(text, query):
                sentences = text.split(".")
                for s in sentences:
                    if any(word.lower() in s.lower() for word in query.split()):
                        return s.strip()
                return text[:200]

            for i, src in enumerate(sources, 1):
                text = src.get("text", "").replace("\n", " ").strip()
                source_name = src.get("source", "unknown")

                relevant_text = extract_relevant_sentence(text, query)

                st.markdown(f"**📌 Source {i}: {source_name}**")
                st.info(relevant_text[:300] + "...")
                st.write("---")
