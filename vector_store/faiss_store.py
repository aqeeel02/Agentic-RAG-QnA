import faiss
import numpy as np
import os
import pickle


class FAISSVectorStore:
    def __init__(self, dim=384, index_file="faiss_index.index", meta_file="metadata.pkl"):
        self.dim = dim
        self.index_file = index_file
        self.meta_file = meta_file
        self.index = faiss.IndexFlatL2(dim)

        # 🔥 store structured metadata
        self.texts = []
        self.metadatas = []

        # Load if exists
        if os.path.exists(index_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                data = pickle.load(f)
                self.texts = data["texts"]
                self.metadatas = data["metadatas"]

    # -------------------- RESET --------------------
    def reset(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.texts = []
        self.metadatas = []

        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.meta_file):
            os.remove(self.meta_file)

    # -------------------- ADD TEXTS --------------------
    def add_texts(self, texts, embeddings, metadatas=None):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)

        self.texts.extend(texts)

        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in texts])

        # Save everything
        with open(self.meta_file, "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadatas": self.metadatas
            }, f)

        faiss.write_index(self.index, self.index_file)

    # -------------------- SEARCH --------------------
    def search(self, query_embedding, k=5):
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(-dist)  # convert L2 distance → similarity
                })

        return results