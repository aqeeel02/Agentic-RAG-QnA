# agents/ingestion_agent.py

import os
from utils.file_parser import parse_file
from mcp.protocol import create_mcp_message

class IngestionAgent:
    def __init__(self, folder="data"):
        self.folder = folder

    def ingest(self):
        docs = {}

        for filename in os.listdir(self.folder):
            file_path = os.path.join(self.folder, filename)

            if os.path.isfile(file_path):
                try:
                    content = parse_file(file_path)

                    if not isinstance(content, str):
                        continue

                    # 🔥 CLEAN TEXT
                    content = content.replace("\n", " ").strip()

                    # 🔥 FILTER SMALL DOCS
                    if len(content) < 50:
                        continue

                    docs[filename] = content

                except Exception as e:
                    print(f"❌ Failed to parse {filename}: {e}")

        if not docs:
            return create_mcp_message(
                sender="IngestionAgent",
                receiver="RetrievalAgent",
                msg_type="INGESTION_FAILED",
                payload={"error": "No valid documents"}
            )

        print(f"✅ Loaded {len(docs)} documents")

        return create_mcp_message(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            msg_type="INGESTION_COMPLETE",
            payload={"documents": docs}
        )