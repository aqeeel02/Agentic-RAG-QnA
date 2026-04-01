import streamlit as st
import os
import shutil

from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from agents.planner_agent import PlannerAgent  # ✅ NEW
llm = LLMResponseAgent()

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Agentic RAG Chatbot", layout="centered")
st.title("🤖 Agentic RAG Chatbot")

# -------------------- RESET --------------------
if st.button("🔄 Reset"):
    if os.path.exists("data"):
        shutil.rmtree("data")

    # also reset vector DB files
    if os.path.exists("faiss_index.index"):
        os.remove("faiss_index.index")
    if os.path.exists("metadata.pkl"):
        os.remove("metadata.pkl")

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

            # -------------------- STEP 1: INGEST --------------------
            ingestion = IngestionAgent()
            ingestion_msg = ingestion.ingest()

            # -------------------- STEP 2: RETRIEVE --------------------
            retrieval = RetrievalAgent()
            retrieval.process_documents(ingestion_msg)

            # -------------------- STEP 3: PLANNER --------------------
            planner = PlannerAgent()
            steps = planner.plan(query)
            print("🧠 PLAN:", steps)

            # -------------------- STEP 4: RETRIEVE CONTEXT --------------------
            retrieved_msg = retrieval.retrieve(query)

            # -------------------- STEP 5: DRAFT ANSWER --------------------
            llm = LLMResponseAgent()                
            draft_response = llm.generate_response(retrieved_msg)
            draft_answer = draft_response["payload"]["answer"]

            # -------------------- STEP 6: REFINE ANSWER --------------------
            refine_query = f"Improve and refine this answer:\n{draft_answer}"

            refined_msg = {
                "payload": {
                    "top_chunks": retrieved_msg["payload"]["top_chunks"],
                    "query": refine_query
                }
            }

            refined_response = llm.generate_response(refined_msg)

            # -------------------- STEP 7: SELF-CHECK --------------------
            check_query = f"Check if this answer is correct based on context. If not, fix it:\n{refined_response['payload']['answer']}"

            check_msg = {
                "payload": {
                    "top_chunks": retrieved_msg["payload"]["top_chunks"],
                    "query": check_query
                }
            }

            final_response = llm.generate_response(check_msg)

            response = final_response

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

            if not sources:
                st.warning("⚠️ No sources found.")
            else:
                for i, src in enumerate(sources, 1):
                    text = src.get("text", "").replace("\n", " ").strip()
                    source_name = src.get("source", "unknown")

                    relevant_text = extract_relevant_sentence(text, query)

                    st.markdown(f"**📌 Source {i}: {source_name}**")
                    st.info(relevant_text[:300] + "...")
                    st.write("---")