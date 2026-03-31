from utils.embedding_utils import EmbeddingModel
from vector_store.faiss_store import FAISSVectorStore
from mcp.protocol import create_mcp_message
import re


class RetrievalAgent:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = FAISSVectorStore()

    # -------------------- ENTITY EXTRACTION --------------------
    def extract_entities(self, text):
        return list(set(re.findall(r'\b[A-Z][a-z]+\b', text)))

    # -------------------- DOCUMENT PROCESSING --------------------
    def process_documents(self, mcp_message):
        docs = mcp_message["payload"].get("documents", {})

        if not docs:
            print("⚠️ No documents found to embed.")
            return create_mcp_message(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",
                msg_type="RETRIEVAL_FAILED",
                payload={"error": "No documents received for retrieval."}
            )

        self.vector_store.reset()

        texts = []
        metadatas = []

        for doc_name, content in docs.items():
            chunks = self.chunk_text(content)

            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "source": doc_name,
                    "chunk_id": i,  # ✅ NEW (helps debugging + tracing)
                    "entities": self.extract_entities(chunk)
                })

        embeddings = self.embedder.embed_texts(texts)

        self.vector_store.add_texts(texts, embeddings, metadatas)

        return create_mcp_message(
            sender="RetrievalAgent",
            receiver="LLMResponseAgent",
            msg_type="RETRIEVAL_READY",
            payload={"stored_docs": list(docs.keys())}
        )

    # -------------------- CHUNKING --------------------
    def chunk_text(self, text, chunk_size=400, overlap=80):
        words = text.split()
        chunks = []

        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    # -------------------- SEMANTIC RERANK --------------------
    def rerank(self, query, docs):
        query_lower = query.lower()
        query_entities = self.extract_entities(query)

        scored_docs = []

        for doc in docs:
            score = 0

            text = doc.get("text", "").lower()
            metadata = doc.get("metadata", {})

            # ✅ base similarity score
            score += doc.get("score", 0)

            # ✅ entity boost (important for names like Rustin)
            for ent in query_entities:
                if ent.lower() in text:
                    score += 2.5

            # ✅ keyword overlap (soft signal)
            overlap = sum(word in text for word in query_lower.split())
            score += overlap * 0.2

            # ✅ slight boost if same source repeats (stability)
            if metadata.get("source"):
                score += 0.1

            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]

    # -------------------- RETRIEVAL --------------------
    def retrieve(self, query):
        if not query or not query.strip():
            return create_mcp_message(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",
                msg_type="RETRIEVAL_FAILED",
                payload={"error": "Empty query provided."}
            )

        query_embedding = self.embedder.embed_texts([query])[0]

        # 🔥 Increase recall
        candidates = self.vector_store.search(query_embedding, k=8)

        if not candidates:
            return create_mcp_message(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",
                msg_type="RETRIEVAL_FAILED",
                payload={"error": "No relevant documents found."}
            )

        reranked = self.rerank(query, candidates)

        top_chunks = reranked[:3]

        # 🧪 DEBUG (very useful)
        print("\n🔍 Retrieved Chunks:\n")
        for chunk in top_chunks:
            print("SOURCE:", chunk.get("metadata", {}).get("source"))
            print(chunk.get("text", "")[:200])
            print("------")

        return create_mcp_message(
            sender="RetrievalAgent",
            receiver="LLMResponseAgent",
            msg_type="CONTEXT_RESPONSE",
            payload={
                "top_chunks": top_chunks,
                "query": query
            }
        )