import json
import os
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger


class CivilComplaintRetriever:
    """
    Retriever for civil complaints using multilingual-e5-large and FAISS.
    Optimized for M3 Phase: Retrieval-Augmented Generation.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        index_path: Optional[str] = None,
        data_path: Optional[str] = None,
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

        # Priority: Load existing index -> Build from data
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        elif data_path and os.path.exists(data_path):
            self.build_index(data_path)

    def _parse_complaint(self, text: str) -> str:
        """Extract the actual complaint content from the chat template text."""
        try:
            # Look for the user part and extract "민원 내용: "
            if "[|user|]" in text:
                user_part = text.split("[|user|]")[1].split("[|endofturn|]")[0]
                if "민원 내용:" in user_part:
                    return user_part.split("민원 내용:")[1].strip()
                return user_part.strip()
            return text
        except Exception as e:
            logger.warning(f"Failed to parse complaint text: {e}")
            return text

    def build_index(self, data_path: str, save_path: Optional[str] = None):
        """Build FAISS index from JSONL data."""
        logger.info(f"Building index from {data_path}")

        complaints = []
        self.metadata = []

        if not os.path.exists(data_path):
            logger.error(f"Data path not found: {data_path}")
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    complaint_text = self._parse_complaint(item["text"])

                    # Extract assistant's answer for metadata (RAG target)
                    answer = ""
                    if "[|assistant|]" in item["text"]:
                        answer = (
                            item["text"].split("[|assistant|]")[1].split("[|endofturn|]")[0].strip()
                        )

                    complaints.append(f"passage: {complaint_text}")
                    self.metadata.append(
                        {
                            "id": item.get("id"),
                            "category": item.get("category"),
                            "complaint": complaint_text,
                            "answer": answer,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Skipping line due to parsing error: {e}. Data might be corrupted or in wrong format."
                    )
                    continue

        if not complaints:
            logger.warning("No complaints found to index.")
            return

        logger.info(f"Encoding {len(complaints)} complaints...")
        embeddings = self.model.encode(
            complaints, show_progress_bar=True, normalize_embeddings=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(
            dimension
        )  # Inner Product for cosine similarity with normalized embeddings
        self.index.add(embeddings.astype("float32"))

        if save_path:
            self.save_index(save_path)

        logger.info("Index build complete.")

    def save_index(self, path: str):
        """Save FAISS index and metadata."""
        if self.index is None:
            raise ValueError("Index not built yet.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

        meta_path = path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Index and metadata saved to {path}")

    def load_index(self, path: str):
        """Load FAISS index and metadata."""
        logger.info(f"Loading index from {path}")
        self.index = faiss.read_index(path)

        meta_path = path + ".meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(f"Index loaded with {len(self.metadata)} entries.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar complaints using E5 'query:' prefix."""
        if self.index is None:
            logger.error("Index is not initialized.")
            return []

        query_embedding = self.model.encode([f"query: {query}"], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata) and idx != -1:
                item = self.metadata[idx].copy()
                item["score"] = float(dist)
                results.append(item)

        return results
