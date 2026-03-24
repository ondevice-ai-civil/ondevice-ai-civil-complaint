"""
BM25 Indexer for Korean civil complaint search.

Provides sparse keyword-based retrieval using morpheme analysis (Okt/Mecab)
and BM25Okapi ranking. Complements the dense FAISS retriever for hybrid search.

Issue: #153
"""

import json
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi


class KoreanTokenizer:
    """
    Korean morpheme tokenizer with Mecab (preferred) and Okt (fallback).
    In closed-network environments where Mecab is not installed, Okt is used.
    """

    def __init__(self, tokenizer_type: str = "auto"):
        """
        Args:
            tokenizer_type: "mecab", "okt", or "auto" (tries Mecab first, falls back to Okt)
        """
        self.tokenizer_type = tokenizer_type
        self._tagger = None
        self._init_tokenizer(tokenizer_type)

    def _init_tokenizer(self, tokenizer_type: str) -> None:
        if tokenizer_type in ("mecab", "auto"):
            try:
                from konlpy.tag import Mecab
                self._tagger = Mecab()
                self.tokenizer_type = "mecab"
                logger.info("Tokenizer initialized: Mecab")
                return
            except Exception:
                if tokenizer_type == "mecab":
                    raise RuntimeError(
                        "Mecab is not installed. Install it or use tokenizer_type='okt'."
                    )
                logger.warning("Mecab unavailable, falling back to Okt.")

        # Okt path
        try:
            from konlpy.tag import Okt
            self._tagger = Okt()
            self.tokenizer_type = "okt"
            logger.info("Tokenizer initialized: Okt")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize any Korean tokenizer: {e}")

    def morphs(self, text: str) -> List[str]:
        """Tokenize text into morphemes, filtering stopwords and short tokens."""
        if not text or not text.strip():
            return []
        try:
            tokens = self._tagger.morphs(text)
            # Filter single characters and common stopwords
            return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]
        except Exception as e:
            logger.warning(f"Tokenization error: {e}. Falling back to whitespace split.")
            return [t for t in text.split() if len(t) > 1]


# Minimal Korean stopwords relevant to civil complaints
_STOPWORDS = {
    "이다", "있다", "하다", "되다", "없다", "않다", "이런", "저런", "그런",
    "합니다", "입니다", "습니다", "됩니다", "있습니다", "없습니다",
    "에서", "으로", "에게", "까지", "부터", "에서는", "으로는",
    "그리고", "하지만", "그러나", "따라서", "그래서",
}


class BM25Indexer:
    """
    BM25 keyword index for civil complaint documents.

    Builds a sparse BM25Okapi index over tokenized Korean text,
    enabling keyword-exact matching for terms like law article numbers,
    department names, and specific complaint keywords.

    Usage:
        indexer = BM25Indexer()
        indexer.build_index(documents)
        results = indexer.search("도로 포장 균열 신고", top_k=10)
        indexer.save("models/bm25_index/complaints.pkl")

        # Later:
        indexer2 = BM25Indexer()
        indexer2.load("models/bm25_index/complaints.pkl")
    """

    def __init__(self, tokenizer_type: str = "auto"):
        self.tokenizer = KoreanTokenizer(tokenizer_type)
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: Optional[List[List[str]]] = None
        self._doc_count: int = 0

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, documents: List[str]) -> None:
        """
        Build BM25 index from a list of document strings.

        Args:
            documents: Raw text documents (one per entry).
        """
        if not documents:
            raise ValueError("Document list is empty.")

        logger.info(f"Tokenizing {len(documents)} documents...")
        self._tokenized_corpus = [self.tokenizer.morphs(doc) for doc in documents]

        # Warn about empty tokenizations
        empty_count = sum(1 for t in self._tokenized_corpus if not t)
        if empty_count:
            logger.warning(f"{empty_count} documents produced empty token lists.")

        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        self._doc_count = len(documents)
        logger.info(f"BM25 index built: {self._doc_count} documents.")

    def build_index_from_jsonl(self, data_path: str, text_field: str = "text") -> None:
        """
        Build index by loading documents from a JSONL file.

        Each line must be a JSON object with a field matching `text_field`.
        For files using EXAONE chat template format, the complaint content
        is extracted from the [|user|] section automatically.

        Args:
            data_path: Path to JSONL file.
            text_field: JSON field containing the text ("text" or "complaint").
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        documents = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if text_field in item:
                        text = item[text_field]
                    elif "complaint" in item:
                        text = item["complaint"]
                    else:
                        # Try extracting from EXAONE chat template
                        text = self._extract_complaint_from_template(
                            item.get("text", "")
                        )
                    documents.append(text)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Line {line_no}: skipping due to error: {e}")

        logger.info(f"Loaded {len(documents)} documents from {data_path}")
        self.build_index(documents)

    @staticmethod
    def _extract_complaint_from_template(text: str) -> str:
        """Extract complaint content from EXAONE chat template format."""
        try:
            if "[|user|]" in text:
                user_part = text.split("[|user|]")[1].split("[|endofturn|]")[0]
                if "민원 내용:" in user_part:
                    return user_part.split("민원 내용:")[1].strip()
                return user_part.strip()
        except Exception:
            pass
        return text

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the BM25 index and return top-k (index, score) pairs.

        Scores are raw BM25 values. Zero-score documents are excluded.

        Args:
            query: Korean query string.
            top_k: Number of results to return.

        Returns:
            List of (document_index, bm25_score) tuples, sorted by score desc.
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not query or not query.strip():
            return []

        tokenized_query = self.tokenizer.morphs(query)
        if not tokenized_query:
            logger.warning("Query tokenized to empty list. Returning no results.")
            return []

        scores: np.ndarray = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0.0
        ]
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize and save the BM25 index to disk.

        Saves both the BM25 model and its tokenized corpus in a single pickle.

        Args:
            path: Destination file path (e.g., "models/bm25_index/complaints.pkl").
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "bm25": self.bm25,
            "tokenized_corpus": self._tokenized_corpus,
            "doc_count": self._doc_count,
            "tokenizer_type": self.tokenizer.tokenizer_type,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 index saved to {path} ({self._doc_count} documents).")

    def load(self, path: str) -> None:
        """
        Load a previously saved BM25 index from disk.

        Args:
            path: Path to the pickle file saved by `save()`.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.bm25 = payload["bm25"]
        self._tokenized_corpus = payload["tokenized_corpus"]
        self._doc_count = payload["doc_count"]
        logger.info(
            f"BM25 index loaded from {path} ({self._doc_count} documents, "
            f"tokenizer: {payload.get('tokenizer_type', 'unknown')})."
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def is_ready(self) -> bool:
        return self.bm25 is not None
