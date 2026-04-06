# 🤖 Agentic RAG Q&A Chatbot

> An intelligent, multi-format document Q&A system powered by a fully agentic **Retrieval-Augmented Generation (RAG)** pipeline — built with FAISS, SentenceTransformers, Groq (LLaMA 3.3-70B), and Streamlit.

---

## 📌 Overview

**Agentic RAG Q&A** is an AI-powered chatbot that lets you upload documents in multiple formats and ask natural language questions about them. Unlike a simple RAG pipeline, this system uses a **Planner Agent** that dynamically decides *how* to answer each query — whether to retrieve from your documents, rewrite the query, answer from general knowledge, or ask for clarification.

Agents communicate through a structured **MCP (Message Control Protocol)** — every handoff between agents is a typed, traceable message with a sender, receiver, payload, and unique trace ID.

All LLM calls go through **Groq's API** using `llama-3.3-70b-versatile` for fast, high-quality responses.

---

## ✨ Features

- 📁 **Multi-format document support** — PDF, DOCX, PPTX, CSV, TXT, and Markdown
- 🔍 **Semantic search** using FAISS + `all-MiniLM-L6-v2` SentenceTransformer embeddings
- 📐 **Smart chunking** — 400 token chunks with 80 token overlap, handled by the Retrieval Agent
- 🏆 **Semantic reranking** — top-8 FAISS candidates reranked by entity match + keyword overlap before returning top-3
- 🧠 **LLM-powered answers** via Groq API (LLaMA 3.3-70B) — fast and accurate
- 🗂️ **Planner Agent** — dynamically selects the best tool for every query
- 🔄 **Query rewriting** — automatically rewrites failed queries for better retrieval
- 🌐 **General knowledge fallback** — answers questions even without uploaded documents
- 📚 **Source attribution** — shows exactly which document chunks backed the answer
- 📨 **MCP message passing** — all agents communicate via structured, traceable messages
- 🔁 **Session reset** — clears all uploads, FAISS index, and conversation in one click
- 🖥️ **Clean Streamlit UI** — easy to use for anyone

---

## 🏗️ Architecture

The system uses a **four-agent + tools** architecture. The Planner Agent sits at the center, deciding which tool to invoke on each attempt (up to 3 retries). All inter-agent communication uses MCP messages.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│             Ingestion Agent                 │
│  file_parser.py → extracts raw text         │
└────────────────────┬────────────────────────┘
                     │ MCP Message (raw text + source)
                     ▼
┌──────────────────────────────────────────────────┐
│               Retrieval Agent                    │
│  chunk_text() → 400 token chunks, 80 overlap     │
│  embedding_utils.py (all-MiniLM-L6-v2)           │
│  → FAISS index → top-8 search → rerank → top-3  │
└────────────────────┬─────────────────────────────┘
                     │ MCP Message (chunks + score)
                     ▼
┌─────────────────────────────────────────────────────┐
│                  Planner Agent                      │
│                                                     │
│  attempt 0 → retrieval_tool                         │
│  attempt 1 → rewrite_tool | llm_general_tool        │
│  attempt 2 → llm_general_tool  (final fallback)     │
│                                                     │
│  Tools: retrieval | rewrite | llm_general | clarify │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           LLM Response Agent                │
│  Groq → LLaMA 3.3-70B → answer + sources   │
└────────────────────┬────────────────────────┘
                     │ MCP Message
                     ▼
              Answer + Source Chunks
```

---

## 🤖 Agents

| Agent | File | Responsibility |
|---|---|---|
| **Ingestion Agent** | `agents/ingestion_agent.py` | Uses `file_parser.py` to parse all documents in `data/` into text chunks with source metadata, returned as an MCP message |
| **Retrieval Agent** | `agents/retrieval_agent.py` | Chunks raw text (400 tokens, 80 overlap), embeds with `all-MiniLM-L6-v2`, indexes in FAISS, fetches top-8 candidates, then reranks by entity match + keyword overlap before returning top-3 |
| **Planner Agent** | `agents/planner_agent.py` | Selects the best tool per attempt using rule-based logic and LLaMA; also rewrites failed queries via a dedicated Groq prompt |
| **LLM Response Agent** | `agents/llm_response_agent.py` | Scores and deduplicates retrieved chunks, constructs a strict context-grounded prompt, calls Groq (LLaMA 3.3-70B), and returns a structured MCP answer |

---

## 🛠️ Tools

Defined in `agents/tools.py`, these are the callable actions the Planner Agent selects from:

| Tool | Function | What it does |
|---|---|---|
| `retrieval` | `retrieval_tool()` | Runs semantic search over the FAISS index and returns matching chunks with their score |
| `llm_general` | `llm_general_tool()` | Answers directly from LLaMA's general knowledge — no documents needed |
| `rewrite` | `rewrite_tool()` | Prompts LLaMA to rewrite a failed query into a more retrieval-friendly form |
| `clarify` | `clarify_tool()` | Returns a clarification message when the query is too vague to process |

---

## 📨 MCP Protocol

All agents communicate using structured **MCP (Message Control Protocol)** messages defined in `mcp/protocol.py`:

```python
{
    "sender":   "LLMResponseAgent",
    "receiver": "User",
    "type":     "FINAL_RESPONSE",
    "trace_id": "uuid-...",          # unique per request for traceability
    "payload":  { ... }              # agent-specific data
}
```

Every handoff — ingestion → retrieval → planner → LLM — passes through this format, making the pipeline easy to debug, extend, and trace.

---

## 🔁 How the Planner Decides

The Planner Agent runs up to **3 attempts** per query:

- **Attempt 0** — always tries `retrieval` first
- **Attempt 1** — if retrieval score is low (`< 0.3`), tries `rewrite`; otherwise asks LLaMA to pick the best tool based on the query and score
- **Attempt 2** — falls back to `llm_general` unconditionally

If the LLM answers "not found in the document" even after a successful retrieval, the planner automatically escalates to general knowledge. If the LLM confidence is below `0.35`, it selects `clarify` instead. This loop ensures the user always gets an answer.

---

## 📁 Project Structure

```
Agentic-RAG-QnA/
│
├── agents/
│   ├── ingestion_agent.py        # Document parsing agent
│   ├── retrieval_agent.py        # FAISS embedding & retrieval agent
│   ├── planner_agent.py          # Tool selection & query rewriting agent
│   ├── llm_response_agent.py     # Groq LLaMA answer generation agent
│   └── tools.py                  # retrieval, llm_general, rewrite, clarify tools
│
├── mcp/
│   └── protocol.py               # create_mcp_message() — structured agent messaging
│
├── utils/
│   ├── file_parser.py            # File type dispatcher (PDF, DOCX, PPTX, CSV, TXT, MD)
│   └── embedding_utils.py        # EmbeddingModel wrapper (all-MiniLM-L6-v2)
│
├── vector_store/                 # FAISS index + metadata persistence
│
├── app.py                        # Streamlit web application
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A [Groq API key](https://console.groq.com/) (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/aqeeel02/Agentic-RAG-QnA.git
cd Agentic-RAG-QnA
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web UI |
| [Groq API](https://groq.com/) | Fast LLM inference |
| [LLaMA 3.3-70B](https://huggingface.co/meta-llama) | Answer generation & planning |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search |
| [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Text embeddings |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF parsing |
| [python-docx](https://python-docx.readthedocs.io/) | DOCX parsing |
| [python-pptx](https://python-pptx.readthedocs.io/) | PPTX parsing |
| [Pandas](https://pandas.pydata.org/) | CSV parsing |

---

## 📦 Dependencies

```
streamlit
python-docx
python-pptx
PyMuPDF
pandas
markdown
faiss-cpu
sentence-transformers
transformers
groq
python-dotenv
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🗺️ Pipeline Deep Dive

### `utils/file_parser.py`
A file-type dispatcher that routes each uploaded file to the correct parser — `fitz` for PDFs, `python-docx` for Word files, `python-pptx` for PowerPoint, `pandas` for CSV, and plain file reading for TXT and Markdown. Returns raw extracted text.

### `utils/embedding_utils.py`
A thin wrapper around SentenceTransformers that loads `all-MiniLM-L6-v2` and exposes an `embed_texts()` method. Used by the Retrieval Agent to encode both document chunks and incoming queries into the same vector space.

### Ingestion Agent
Scans the `data/` directory, calls `file_parser.py` for each file, and wraps the resulting raw text in an MCP message with source metadata. It does **not** chunk — it hands off the full text to the Retrieval Agent.

### Retrieval Agent
The most complex agent in the pipeline. It first **chunks** the raw text into 400-token windows with 80-token overlap using `chunk_text()`. Each chunk is embedded using `EmbeddingModel.embed_texts()` and indexed in FAISS with metadata (`source`, `chunk_id`, `entities`). On a query, it fetches the **top-8 candidates** from FAISS, then **reranks** them using a scoring formula that boosts named entity matches (+2.5 per entity) and keyword overlap (+0.2 per word hit) on top of the base similarity score. The top-3 reranked chunks and their average score are returned via MCP message.

### Planner Agent
The brain of the system. On each attempt it evaluates the query and retrieval score to pick the most appropriate tool. Uses LLaMA itself to reason about tool selection when the decision isn't purely rule-based. Also rewrites failed queries by prompting LLaMA for a more retrieval-friendly version.

### LLM Response Agent
Receives retrieved chunks, scores and deduplicates them by relevance, builds a strict context-grounded prompt (max 2000 characters of context), and calls LLaMA 3.3-70B via Groq. Returns a final MCP message with the answer and the top 2 most relevant source snippets with their chunk IDs.

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute it.

---

> ⭐ If you found this project useful, consider giving it a star!
