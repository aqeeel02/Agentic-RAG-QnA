class PlannerAgent:
    def __init__(self):
        self.name = "PlannerAgent"

    def plan(self, query):
        q = query.lower()

        # Decide steps based on query type
        if "summarize" in q or "overview" in q:
            return ["retrieve", "summarize"]

        elif "analyze" in q or "compare" in q:
            return ["retrieve", "analyze", "refine"]

        else:
            return ["retrieve", "answer"]
