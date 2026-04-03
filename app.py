import streamlit as st
import os
import shutil

from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from agents.planner_agent import PlannerAgent
from agents.tools import (
    retrieval_tool,
    llm_general_tool,
    rewrite_tool,
    clarify_tool
)

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Agentic RAG Chatbot", layout="centered")
st.title("🤖 Agentic RAG Chatbot")

# -------------------- INIT AGENTS --------------------
retrieval = RetrievalAgent()
llm_agent = LLMResponseAgent()
planner = PlannerAgent(llm_agent)

# -------------------- RESET --------------------
if st.button("🔄 Reset"):
    if os.path.exists("data"):
        shutil.rmtree("data")
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

if uploaded_files and not st.session_state.get("documents_loaded"):
    os.makedirs("data", exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join("data", file.name), "wb") as f:
            f.write(file.read())

    ingestion = IngestionAgent()
    ingestion_msg = ingestion.ingest()
    retrieval.process_documents(ingestion_msg)

    st.session_state["documents_loaded"] = True
    st.success("✅ Documents processed and indexed!")

# -------------------- ASK QUESTION --------------------
st.header("💬 Ask a Question")

query = st.text_input("Type your question here:")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("🧠 Thinking..."):

        current_query = query
        retrieval_score = None
        answered = False

        for attempt in range(3):
            decision = planner.select_tool(current_query, retrieval_score, attempt)
            tool = decision["tool"]

            st.caption(f"🧠 Tool selected: **{tool}** (attempt {attempt + 1})")

            # ── RETRIEVAL ──
            if tool == "retrieval":
                # Only retrieve if documents are loaded
                if not st.session_state.get("documents_loaded"):
                    # No docs — treat as general question immediately
                    answer = llm_general_tool(llm_agent, current_query)["answer"]
                    st.info("💡 No documents uploaded — answering from general knowledge")
                    st.subheader("💡 Answer:")
                    st.success(answer)
                    answered = True
                    break

                result = retrieval_tool(retrieval, current_query)

                if result["chunks"]:
                    response = llm_agent.generate_response(result["msg"])
                    answer = response["payload"]["answer"]

                    # If LLM says not found in doc → escalate to general
                    not_found_phrases = [
                        "not found in the document",
                        "not mentioned in the document",
                        "not in the document",
                        "no information",
                        "cannot find"
                    ]
                    if any(phrase in answer.lower() for phrase in not_found_phrases):
                        # Document had chunks but answer wasn't there
                        # Try general knowledge
                        retrieval_score = 0.2
                        continue

                    st.subheader("💡 Answer:")
                    st.success(answer)

                    sources = response["payload"].get("sources", [])
                    if sources:
                        st.subheader("📚 Sources")
                        for src in sources:
                            st.info(src["text"][:200] + "...")

                    answered = True
                    break

                else:
                    # No chunks retrieved at all
                    retrieval_score = result.get("score", 0.1)
                    continue

            # ── REWRITE ──
            elif tool == "rewrite":
                new_query = planner.rewrite_query(current_query)
                st.info(f"🔄 Query rewritten: *{new_query}*")
                current_query = new_query
                retrieval_score = None
                continue

            # ── GENERAL KNOWLEDGE ──
            elif tool == "llm_general":
                answer = llm_general_tool(llm_agent, current_query)["answer"]
                st.caption("🌐 Answered from general knowledge")
                st.subheader("💡 Answer:")
                st.success(answer)
                answered = True
                break

            # ── CLARIFY ──
            elif tool == "clarify":
                st.info(clarify_tool()["message"])
                answered = True
                break

        # ── FINAL FALLBACK (loop exhausted without answer) ──
        if not answered:
            answer = llm_general_tool(llm_agent, current_query)["answer"]
            st.caption("🌐 Fallback: answered from general knowledge")
            st.subheader("💡 Answer:")
            st.success(answer)