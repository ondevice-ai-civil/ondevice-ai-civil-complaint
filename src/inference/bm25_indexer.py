"""
BM25 Indexer for Korean civil complaint search.

Provides sparse keyword-based retrieval using morpheme analysis (Okt/Mecab)
and BM25Okapi ranking. Complements the dense FAISS retriever for hybrid search.

Issue: #153

Known limitation:
    BM25Okapi assigns negative IDF when a term appears in every document
    (df == N). search() returns only positive-scoring results, so a single-
    document corpus may return empty results for exact-match queries.
    In practice this does not occur at production scale (10k+ documents).

Security:
    Uses pickle for BM25Okapi serialization. Only load index files from
    trusted sources within the closed-network environment. When the
    BM25_INDEX_HMAC_KEY environment variable is set, save() signs the
    payload and load() verifies the HMAC before deserialization.
"""

import hashlib
import hmac
import json
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi


# Minimal Korean stopwords relevant to civil complaints
# Defined before KoreanTokenizer to avoid forward-reference maintenance hazard.
_STOPWORDS = frozenset({
    "이다", "있다", "하다", "되다", "없다", "않다", "이런", "저런", "그런",
    "합니다", "입니다", "습니다", "됩니다", "있습니다", "없습니다",
    "에서", "으로", "에게", "까지", "부터", "에서는", "으로는",
    "그리고", "하지만", "그러나", "따라서", "그래서",
})


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
            tokens = self._tagger.morphs(str(text))
            # Filter single characters and common stopwords
            return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]
        except Exception as e:
            logger.warning(
                f"Tokenization error (len={len(text)}): {type(e).__name__}. "
                "Falling back to whitespace split."
            )
            return [t for t in str(text).split() if len(t) > 1]


class BM25Indexer:
    """
    BM25 keyword index for civil complaint documents.

    Builds a sparse BM25Okapi index over tokenized Korean text,
    enabling keyword-exact matching for terms like law article numbers,
    department names, and specific complaint keywords.

    Return type note:
        search() returns List[Tuple[int, float]] — raw corpus indices and BM25
        scores. This is intentionally lower-level than CivilComplaintRetriever
        which returns List[Dict]. The HybridSearchEngine is responsible for
        mapping indices to metadata and fusing scores across both retrievers.

    Usage:
        indexer = BM25Indexer()
        indexer.build_index(documents)
        results = indexer.search("도로 포장 균열 신고", top_k=10)
        indexer.save("models/bm25_index/complaints.pkl")

        # Later:
        indexer2 = BM25Indexer()
        indexer2.load("models/bm25_index/complaints.pkl")
    """

    _PAYLOAD_VERSION = 1
    _HMAC_KEY_ENV = "BM25_INDEX_HMAC_KEY"

    def __init__(self, tokenizer_type: str = "auto"):
        self.tokenizer = KoreanTokenizer(tokenizer_type)
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: Optional[List[List[str]]] = None
        self._doc_count: int = 0

    def __repr__(self) -> str:
        return (
            f"BM25Indexer(docs={self._doc_count}, "
            f"tokenizer={self.tokenizer.tokenizer_type}, "
            f"ready={self.is_ready()})"
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, documents: List[str]) -> None:
        """
        Build BM25 index from a list of document strings.

        Args:
            documents: Raw text documents (one per entry).

        Raises:
            ValueError: If documents list is empty or all documents tokenize
                        to empty token lists (would cause ZeroDivisionError
                        inside BM25Okapi).
        """
        if not documents:
            raise ValueError("Document list is empty.")

        if self.bm25 is not None:
            logger.warning("Rebuilding BM25 index — existing index will be replaced.")

        logger.info(f"Tokenizing {len(documents)} documents...")
        tokenized = [self.tokenizer.morphs(doc) for doc in documents]

        empty_count = sum(1 for t in tokenized if not t)
        if empty_count:
            logger.warning(f"{empty_count} documents produced empty token lists.")

        # Guard against all-empty corpus which causes ZeroDivisionError in BM25Okapi
        if all(len(t) == 0 for t in tokenized):
            raise ValueError(
                "All documents produced empty token lists. "
                "Check that documents contain valid Korean text."
            )

        logger.info("Building BM25 index...")
        self._tokenized_corpus = tokenized
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
                        raw = item[text_field]
                        # Auto-extract complaint from EXAONE chat template
                        if isinstance(raw, str) and "[|user|]" in raw:
                            text = self._extract_complaint_from_template(raw)
                        else:
                            text = raw
                    elif "complaint" in item:
                        text = item["complaint"]
                    else:
                        text = self._extract_complaint_from_template(
                            item.get("text", "")
                        )
                    # Ensure text is always a string
                    if not isinstance(text, str):
                        text = str(text) if text is not None else ""
                    documents.append(text)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Line {line_no}: skipping due to error: {e}")

        logger.info(f"Loaded {len(documents)} documents from {data_path}")
        self.build_index(documents)

    @staticmethod
    def _extract_complaint_from_template(text: str) -> str:
        """Extract complaint content from EXAONE chat template format."""
        if not text:
            return text
        try:
            if "[|user|]" in text:
                user_part = text.split("[|user|]")[1].split("[|endofturn|]")[0]
                if "민원 내용:" in user_part:
                    return user_part.split("민원 내용:")[1].strip()
                return user_part.strip()
        except Exception as e:
            logger.debug(f"Template extraction fallback: {type(e).__name__}")
        return text

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the BM25 index and return top-k (index, score) pairs.

        Only positive-scoring documents are returned. Scores are raw BM25
        values and are not normalized — the HybridSearchEngine handles
        score fusion (e.g., RRF) across dense and sparse retrievers.

        Args:
            query: Korean query string.
            top_k: Number of results to return.

        Returns:
            List of (document_index, bm25_score) tuples, sorted by score desc.

        Raises:
            RuntimeError: If index has not been built or loaded.
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

        # Use argpartition O(N) instead of argsort O(N log N) for top-k selection
        actual_k = min(top_k, len(scores))
        if actual_k == 0:
            return []

        top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

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

        Security: Uses pickle for BM25Okapi serialization. When the
        ``BM25_INDEX_HMAC_KEY`` environment variable is set, the payload is
        signed with HMAC-SHA256 and a ``.sig`` sidecar file is written. Only
        load index files from trusted sources within the closed-network
        environment.

        Args:
            path: Destination file path (e.g., "models/bm25_index/complaints.pkl").
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Fix: use abspath to avoid makedirs("") crash on bare filenames
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

        payload = {
            "version": self._PAYLOAD_VERSION,
            "bm25": self.bm25,
            "tokenized_corpus": self._tokenized_corpus,
            "doc_count": self._doc_count,
            "tokenizer_type": self.tokenizer.tokenizer_type,
        }
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

        # HMAC signing (when key is configured)
        hmac_key = os.getenv(self._HMAC_KEY_ENV)
        if hmac_key:
            sig = hmac.new(hmac_key.encode(), data, hashlib.sha256).hexdigest()
            sig_path = path + ".sig"
            with open(sig_path, "w", encoding="utf-8") as sf:
                sf.write(sig)
            logger.info(f"HMAC signature written to {sig_path}")

        with open(path, "wb") as f:
            f.write(data)
        logger.info(f"BM25 index saved to {path} ({self._doc_count} documents).")

    def load(self, path: str) -> None:
        """
        Load a previously saved BM25 index from disk.

        Security: When the ``BM25_INDEX_HMAC_KEY`` environment variable is
        set, the HMAC-SHA256 signature is verified before deserialization.
        Pickle deserialization can execute arbitrary code — only load files
        from trusted sources within the closed-network environment.

        Args:
            path: Path to the pickle file saved by `save()`.

        Raises:
            FileNotFoundError: If the index file does not exist.
            ValueError: If the file is corrupt, has an incompatible schema,
                        or fails HMAC verification.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as f:
            data = f.read()

        # HMAC verification (when key is configured)
        hmac_key = os.getenv(self._HMAC_KEY_ENV)
        if hmac_key:
            sig_path = path + ".sig"
            if not os.path.exists(sig_path):
                raise ValueError(
                    f"HMAC signature file missing: {sig_path}. "
                    "Index file cannot be verified — rebuild the index."
                )
            with open(sig_path, "r", encoding="utf-8") as sf:
                expected_sig = sf.read().strip()
            actual_sig = hmac.new(
                hmac_key.encode(), data, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(actual_sig, expected_sig):
                raise ValueError(
                    "BM25 index HMAC verification failed — file may be tampered. "
                    "Rebuild the index with a trusted data source."
                )
            logger.info("HMAC signature verified.")

        try:
            payload = pickle.loads(data)
        except Exception as e:
            raise ValueError(f"Failed to load BM25 index (corrupt or incompatible): {e}") from e

        # Payload version check
        saved_version = payload.get("version")
        if saved_version != self._PAYLOAD_VERSION:
            raise ValueError(
                f"BM25 index version mismatch: file has v{saved_version}, "
                f"expected v{self._PAYLOAD_VERSION}. Rebuild the index."
            )

        try:
            self.bm25 = payload["bm25"]
            self._tokenized_corpus = payload["tokenized_corpus"]
            self._doc_count = payload["doc_count"]
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"BM25 index file has incompatible schema (missing key: {e}). "
                "Rebuild the index."
            ) from e

        saved_tokenizer = payload.get("tokenizer_type", "unknown")
        if saved_tokenizer != self.tokenizer.tokenizer_type:
            logger.warning(
                f"Tokenizer mismatch: index was built with '{saved_tokenizer}' "
                f"but current tokenizer is '{self.tokenizer.tokenizer_type}'. "
                "Search recall may be degraded. Rebuild the index to resolve."
            )

        logger.info(
            f"BM25 index loaded from {path} ({self._doc_count} documents, "
            f"tokenizer: {saved_tokenizer})."
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def is_ready(self) -> bool:
        return self.bm25 is not None
