# agents/ingestion_agent.py

import os
from utils.file_parser import parse_file
from mcp.protocol import create_mcp_message

class IngestionAgent:
    def __init__(self, folder="data"):
        self.folder = folder

    def ingest(self):
        docs = {}

        # Check if data folder exists
        if not os.path.exists(self.folder):
            print(f"❌ Folder '{self.folder}' does not exist.")
            return create_mcp_message(
                sender="IngestionAgent",
                receiver="RetrievalAgent",
                msg_type="INGESTION_FAILED",
                payload={"error": "Folder not found"}
            )

        # Loop through files and parse them
        for filename in os.listdir(self.folder):
            file_path = os.path.join(self.folder, filename)
            if os.path.isfile(file_path):
                print(f"📂 Parsing file: {filename}")
                try:
                    content = parse_file(file_path)
                    docs[filename] = content.strip() if isinstance(content, str) else ""
                except Exception as e:
                    print(f"❌ Failed to parse {filename}: {e}")

        if not docs:
            print("⚠️ No documents successfully parsed.")
            return create_mcp_message(
                sender="IngestionAgent",
                receiver="RetrievalAgent",
                msg_type="INGESTION_FAILED",
                payload={"error": "No documents parsed."}
            )

        return create_mcp_message(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            msg_type="INGESTION_COMPLETE",
            payload={"documents": docs}
        )
