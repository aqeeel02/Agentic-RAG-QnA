import os
from groq import Groq
from mcp.protocol import create_mcp_message
from dotenv import load_dotenv
load_dotenv()


class LLMResponseAgent:
    def __init__(self):
        self.name = "LLMResponseAgent"

        # ⚠️ Move to env variable later
        import os
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # -------------------- CONTEXT EXTRACTION --------------------
    def format_context(self, context_chunks):
        formatted_chunks = []

        for chunk in context_chunks[:3]:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk)

            if text:
                formatted_chunks.append(text)

        return "\n".join(formatted_chunks)

    # -------------------- SOURCE FORMATTING --------------------
    def format_sources(self, context_chunks):
        sources = []

        for chunk in context_chunks[:3]:
            if isinstance(chunk, dict):
                sources.append({
                    "text": chunk.get("text", "")[:300],
                    "source": chunk.get("metadata", {}).get("source", "unknown")
                })
            else:
                sources.append({
                    "text": str(chunk)[:300],
                    "source": "unknown"
                })

        return sources

    # -------------------- RESPONSE GENERATION --------------------
    def generate_response(self, mcp_message):
        payload = mcp_message.get("payload", {})
        context_chunks = payload.get("top_chunks", [])
        user_query = payload.get("query", "")

        if not context_chunks or not user_query:
            return create_mcp_message(
                sender=self.name,
                receiver="User",
                msg_type="FINAL_RESPONSE",
                payload={"answer": "⚠️ Missing context or query."}
            )

        # ✅ Extract context
        context = self.format_context(context_chunks)
        context = context[:2000]

        # ✅ Prompt
        prompt = f"""
You are a precise question-answering assistant.

Rules:
- Answer ONLY using the provided context
- If answer is not present, say: "Not found in the document"
- Be concise (2–3 sentences)

Context:
{context}

Question:
{user_query}

Answer:
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You answer strictly from context."},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content.strip()

        except Exception as e:
            return create_mcp_message(
                sender=self.name,
                receiver="User",
                msg_type="FINAL_RESPONSE",
                payload={"answer": f"⚠️ LLM Error: {str(e)}"}
            )

        # ✅ FIX: format sources properly
        formatted_sources = self.format_sources(context_chunks)

        return create_mcp_message(
            sender=self.name,
            receiver="User",
            msg_type="FINAL_RESPONSE",
            payload={
                "answer": answer,
                "sources": formatted_sources   # 🔥 THIS FIXES YOUR ISSUE
            }
        )