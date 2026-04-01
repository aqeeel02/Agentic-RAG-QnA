# 🤖 Agentic RAG Q&A System

A **from-scratch implementation of an Agentic Retrieval-Augmented Generation (RAG) system** designed to perform context-aware question answering over unstructured documents using a modular, multi-agent architecture.

Unlike typical implementations that rely on orchestration frameworks such as LangChain, this system is built at a **lower abstraction level**, directly managing retrieval pipelines, agent coordination, and LLM interactions. This provides fine-grained control over data flow, context construction, and reasoning behavior.

---

## 🚀 Core Capabilities

* Context-aware semantic retrieval over document corpus
* Multi-step reasoning using an agent-driven pipeline
* Explicit control over prompt construction and context injection
* Modular design enabling independent agent orchestration
* Streamlit-based interface for real-time interaction

---

## 🧠 System Architecture

The system follows an **agent-oriented execution pipeline**, where each stage is explicitly handled without external orchestration frameworks:

User Query
→ Planner Agent
→ Retrieval Agent
→ LLM Response Agent
→ Final Output

### 🔹 Planner Agent

Responsible for interpreting the user query and determining the retrieval strategy. It acts as a lightweight reasoning layer that prepares structured input for downstream components.

### 🔹 Retrieval Agent

Handles document ingestion, chunk processing, and semantic retrieval. It constructs the context window by selecting the most relevant document segments for the query.

### 🔹 LLM Response Agent

Generates the final response by combining:

* user query
* retrieved context
* system-level prompting

This agent explicitly controls prompt formatting and avoids reliance on high-level abstractions.

---

## ⚙️ Design Philosophy

This project intentionally avoids frameworks like LangChain to demonstrate a **ground-up understanding of RAG systems**, including:

* Manual orchestration of agent workflows
* Direct handling of embedding-based retrieval
* Explicit prompt engineering and context assembly
* Fine control over latency, token usage, and response quality

The architecture emphasizes **transparency, debuggability, and extensibility**, making it suitable for experimentation with advanced retrieval and reasoning strategies.

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Vector-based retrieval (FAISS or equivalent)
* LLM API (Groq / OpenAI)

---

## 📂 Project Structure

```text
agentic-rag-chatbot/
│
├── agents/
│   ├── ingestion_agent.py
│   ├── retrieval_agent.py
│   ├── planner_agent.py
│   ├── llm_response_agent.py
│
├── app.py
├── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/aqeeel02/Agentic-RAG-QnA.git
cd Agentic-RAG-QnA
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Usage

* Query documents using natural language
* Retrieve contextually relevant information
* Generate grounded responses based on retrieved evidence

---

## 🧩 Implementation Highlights

* End-to-end RAG pipeline implemented without orchestration libraries
* Explicit separation of concerns via agent abstraction
* Deterministic control over retrieval and generation stages
* Designed for extensibility into hybrid search, memory, and advanced reasoning systems
