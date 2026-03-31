# main.py
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent  # ✅ NEW

if __name__ == "__main__":
    # Step 1: Ingest docs
    ingestion = IngestionAgent()
    ingestion_message = ingestion.ingest()

    # Step 2: Store embeddings
    retrieval = RetrievalAgent()
    store_response = retrieval.process_documents(ingestion_message)
    print("📄 Documents stored:", store_response["payload"]["stored_docs"])

    # Step 3: Query
    query = "What is artificial intelligence?"
    response = retrieval.retrieve(query)
    print("\n🧠 Retrieved Chunks for Query:")
    for chunk in response["payload"]["top_chunks"]:
        print("-", chunk[:100], "...\n")

    # ✅ Step 4: Generate Final Answer
    llm_agent = LLMResponseAgent()
    final_msg = llm_agent.generate_response(response)

    print("💬 Final Answer from Chatbot:")
    print(final_msg["payload"]["answer"])
