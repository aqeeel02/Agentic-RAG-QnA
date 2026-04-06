import os
import re
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
    def _extract_supporting_snippet(self, text, answer, query):
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return ""

        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        answer_words = {w.lower() for w in answer.split() if len(w) > 3}
        query_words = {w.lower() for w in query.split() if len(w) > 2}

        best_sentence = ""
        best_score = -1

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_lower = sentence.lower()
            score = sum(2 for word in query_words if word in sentence_lower)
            score += sum(1 for word in answer_words if word in sentence_lower)

            # punish heading/title-like fragments
            if len(sentence.split()) < 6:
                score -= 3
            if sentence.startswith("#"):
                score -= 5

            if score > best_score:
                best_score = score
                best_sentence = sentence

        if best_sentence and best_score > 0:
            return best_sentence

        # fallback to a compact chunk preview if no strong sentence exists
        preview = clean_text[:450]
        if len(clean_text) > 450:
            preview = preview.rsplit(" ", 1)[0]
        return preview

    def select_sources(self, context_chunks, query, answer=None):
        scored_sources = []

        for chunk in context_chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                source_name = metadata.get("source", "unknown")
                chunk_id = metadata.get("chunk_id")
            else:
                text = str(chunk)
                source_name = "unknown"
                chunk_id = None

            clean_text = text.replace("\n", " ").strip()
            if not clean_text:
                continue

            snippet = self._extract_supporting_snippet(clean_text, answer or query, query)
            snippet_lower = snippet.lower()
            query_score = sum(2 for word in query.lower().split() if len(word) > 2 and word in snippet_lower)
            answer_score = 0
            if answer:
                answer_score = sum(1 for word in answer.lower().split() if len(word) > 3 and word in snippet_lower)
            total_score = query_score + answer_score

            if len(snippet.split()) < 6:
                total_score -= 3

            scored_sources.append({
                "score": total_score,
                "text": snippet,
                "source": source_name,
                "chunk_id": chunk_id,
                "full_text": clean_text
            })

        scored_sources.sort(key=lambda x: x["score"], reverse=True)

        final_sources = []
        seen = set()

        for src in scored_sources:
            fingerprint = src["text"][:120].lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            final_sources.append(src)
            if len(final_sources) >= 2:
                break

        return final_sources

    def format_sources(self, selected_sources):
        return [
            {
                "text": src["text"],
                "source": src["source"],
                "chunk_id": src["chunk_id"]
            }
            for src in selected_sources
        ]

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

        # ✅ Extract context from top selected sources so answer + sources stay aligned
        selected_sources = self.select_sources(context_chunks, user_query)
        selected_context_chunks = [
            {
                "text": src["full_text"],
                "metadata": {
                    "source": src["source"],
                    "chunk_id": src["chunk_id"]
                }
            }
            for src in selected_sources
        ] or context_chunks[:2]

        context = self.format_context(selected_context_chunks)
        context = context[:2000]

        # ✅ Prompt
        prompt = f"""
You are a precise question-answering assistant.

Rules:
- Answer ONLY using the provided context
- If answer is not present, say: "Not found in the document"
- Be concise (2–3 sentences)
- Include only claims directly supported by the retrieved context
- Prefer the most central idea from the strongest matching chunk
- Do not combine unrelated points unless the context clearly connects them
- Do not infer extra details beyond what is explicitly stated

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

        # ✅ Use the same selected chunks for displayed sources
        selected_sources = self.select_sources(selected_context_chunks, user_query, answer)
        formatted_sources = self.format_sources(selected_sources)

        return create_mcp_message(
            sender=self.name,
            receiver="User",
            msg_type="FINAL_RESPONSE",
            payload={
                "answer": answer,
                "sources": formatted_sources   # 🔥 THIS FIXES YOUR ISSUE
            }
        )