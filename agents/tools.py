# agents/tools.py

def retrieval_tool(retrieval_agent, query):
    msg = retrieval_agent.retrieve(query)

    payload = msg.get("payload", {})
    chunks = payload.get("top_chunks", [])
    score = payload.get("score", 0.0)

    return {
        "chunks": chunks,
        "msg": msg,
        "score": score
    }


def llm_general_tool(llm_agent, query):
    return {
        "answer": llm_agent.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer carefully. Say if unsure."},
                {"role": "user", "content": query}
            ]
        ).choices[0].message.content
    }


def rewrite_tool(llm_agent, query):
    new_query = llm_agent.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": f"Rewrite for better search:\n{query}"}
        ]
    ).choices[0].message.content

    return {"query": new_query}


def clarify_tool():
    return {"message": "Can you clarify your question?"}
