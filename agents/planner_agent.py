# agents/planner_agent.py

import json
import re


class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    # ─────────────────────────────────────────────
    # PUBLIC — called from app.py
    # ─────────────────────────────────────────────
    def select_tool(self, query, retrieval_score=None, attempt=0):
        """
        Returns: { "tool": str }
        Tools: retrieval | rewrite | llm_general | clarify
        """
        tool = self._decide_tool(query, retrieval_score, attempt)
        return {"tool": tool}

    def rewrite_query(self, query):
        """
        Public rewrite — called from app.py when tool == rewrite.
        Returns the rewritten query string.
        """
        prompt = f"""The following search query failed to find useful results in a document.
Rewrite it to be more specific and retrieval-friendly.

Original query: "{query}"

Return ONLY the rewritten query as plain text. No explanation. No quotes."""

        try:
            response = self.llm.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query  # fallback to original

    # ─────────────────────────────────────────────
    # CORE DECISION LOGIC
    # ─────────────────────────────────────────────
    def _decide_tool(self, query, retrieval_score, attempt):
        """
        attempt 0 → always retrieval
        attempt 1 → rewrite if score low, else ask LLM
        attempt 2 → llm_general (final fallback)
        """

        if attempt == 0:
            return "retrieval"

        if attempt >= 2:
            return "llm_general"

        # If retrieval clearly failed → rewrite
        if retrieval_score is not None and retrieval_score < 0.3:
            return "rewrite"

        # Ask LLM to decide
        prompt = f"""You are an AI agent deciding how to answer a user query.

Query: "{query}"
Previous retrieval score: {retrieval_score if retrieval_score is not None else "unknown"}
Attempt: {attempt}

Available tools:
- retrieval    → answer is inside an uploaded document
- llm_general  → answer is general knowledge (history, science, common facts, etc.)
- rewrite      → retrieval failed, try a better search query
- clarify      → query is too vague

IMPORTANT: If the query is clearly general knowledge (e.g. sky color, capital cities,
basic science, common facts) and NOT something that would be in a document,
choose llm_general.

Respond ONLY with valid JSON. No markdown. No explanation.
Example: {{"tool": "llm_general", "confidence": 0.9}}"""

        try:
            response = self.llm.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            decision = json.loads(raw)

            tool = decision.get("tool", "llm_general")
            confidence = float(decision.get("confidence", 0.5))

            if confidence < 0.35:
                return "clarify"

            if tool not in ("retrieval", "llm_general", "rewrite", "clarify"):
                return "llm_general"

            return tool

        except Exception:
            return "llm_general"