"""
Unit tests for BM25Indexer (Issue #153).

Tests cover:
- Korean tokenization (Okt)
- Index build from list and JSONL
- Search with top-k results
- Save / load round-trip
- Edge cases (empty query, empty docs, uninitialized index)
"""

import json
import os
import tempfile
import time

import pytest

from src.inference.bm25_indexer import BM25Indexer, KoreanTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    "도로 포장 균열로 인해 자전거 사고가 발생했습니다. 즉시 보수 요청드립니다.",
    "아파트 단지 앞 불법 주정차 차량 때문에 보행자 통행이 불편합니다.",
    "공원 내 가로등이 고장나 야간 안전사고가 우려됩니다. 점검 바랍니다.",
    "음식물 쓰레기통이 항상 넘쳐 악취와 해충 문제가 심각합니다.",
    "주민센터 복지 서비스 신청 방법을 안내해 주시기 바랍니다.",
    "버스 정류장 쉼터가 파손되어 비가 올 때 불편합니다.",
    "불법 광고 현수막이 도로변에 다수 설치되어 있습니다.",
    "아파트 주차장 진입로가 좁아 대형 차량 통행이 어렵습니다.",
    "하수구 악취가 심하여 민원을 제기합니다. 청소 및 점검 요청합니다.",
    "공공 화장실 청결 상태가 불량합니다. 관리 강화를 요청합니다.",
]


@pytest.fixture
def indexer():
    idx = BM25Indexer(tokenizer_type="okt")
    idx.build_index(SAMPLE_DOCUMENTS)
    return idx


@pytest.fixture
def jsonl_file(tmp_path):
    """Write sample docs to a JSONL file in EXAONE chat template format."""
    path = tmp_path / "test_data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(SAMPLE_DOCUMENTS):
            record = {
                "id": f"doc_{i}",
                "text": f"[|system|]시스템[|endofturn|][|user|]민원 내용: {doc}[|endofturn|][|assistant|]답변입니다.[|endofturn|]",
                "category": "test",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# KoreanTokenizer tests
# ---------------------------------------------------------------------------

class TestKoreanTokenizer:
    def test_okt_initialization(self):
        tok = KoreanTokenizer("okt")
        assert tok.tokenizer_type == "okt"

    def test_morphs_returns_list(self):
        tok = KoreanTokenizer("okt")
        result = tok.morphs("도로 포장 균열 신고")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_morphs_filters_short_tokens(self):
        tok = KoreanTokenizer("okt")
        result = tok.morphs("가 나 도로 포장")
        # Single-char tokens should be filtered
        assert all(len(t) > 1 for t in result)

    def test_morphs_empty_string(self):
        tok = KoreanTokenizer("okt")
        assert tok.morphs("") == []

    def test_morphs_whitespace_only(self):
        tok = KoreanTokenizer("okt")
        assert tok.morphs("   ") == []

    def test_mecab_fallback_to_okt_on_auto(self):
        """'auto' should gracefully use Okt when Mecab is unavailable."""
        tok = KoreanTokenizer("auto")
        # Should succeed regardless of whether Mecab is installed
        result = tok.morphs("민원 처리 요청")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# BM25Indexer build tests
# ---------------------------------------------------------------------------

class TestBM25IndexerBuild:
    def test_build_from_list(self, indexer):
        assert indexer.is_ready()
        assert indexer.doc_count == len(SAMPLE_DOCUMENTS)

    def test_build_raises_on_empty(self):
        idx = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(ValueError):
            idx.build_index([])

    def test_build_from_jsonl(self, jsonl_file):
        idx = BM25Indexer(tokenizer_type="okt")
        idx.build_index_from_jsonl(jsonl_file)
        assert idx.is_ready()
        assert idx.doc_count == len(SAMPLE_DOCUMENTS)

    def test_build_from_jsonl_missing_file(self, tmp_path):
        idx = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(FileNotFoundError):
            idx.build_index_from_jsonl(str(tmp_path / "nonexistent.jsonl"))

    def test_build_time_under_threshold(self):
        """Build time for 1000 docs should be under 30 seconds."""
        # Repeat sample docs to reach 1000
        docs = SAMPLE_DOCUMENTS * 100  # 1000 docs
        idx = BM25Indexer(tokenizer_type="okt")
        start = time.time()
        idx.build_index(docs)
        elapsed = time.time() - start
        assert elapsed < 30.0, f"Build took {elapsed:.1f}s, expected < 30s"


# ---------------------------------------------------------------------------
# BM25Indexer search tests
# ---------------------------------------------------------------------------

class TestBM25IndexerSearch:
    def test_search_returns_list(self, indexer):
        results = indexer.search("도로 포장 균열", top_k=3)
        assert isinstance(results, list)

    def test_search_top_k_limit(self, indexer):
        results = indexer.search("도로 포장", top_k=3)
        assert len(results) <= 3

    def test_search_result_format(self, indexer):
        results = indexer.search("도로 포장 균열", top_k=5)
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)
            assert score > 0.0

    def test_search_relevance(self, indexer):
        """Top result for '도로 포장 균열' should be doc 0."""
        results = indexer.search("도로 포장 균열", top_k=3)
        assert len(results) > 0
        top_idx, top_score = results[0]
        assert top_idx == 0  # First document is most relevant

    def test_search_empty_query(self, indexer):
        results = indexer.search("", top_k=5)
        assert results == []

    def test_search_whitespace_query(self, indexer):
        results = indexer.search("   ", top_k=5)
        assert results == []

    def test_search_unrelated_query_returns_empty(self, indexer):
        """Query with no overlapping tokens returns empty list."""
        results = indexer.search("zzz", top_k=5)
        assert results == []

    def test_search_before_build_raises(self):
        idx = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(RuntimeError):
            idx.search("테스트")

    def test_search_scores_sorted_descending(self, indexer):
        results = indexer.search("민원 신고 요청", top_k=5)
        if len(results) > 1:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# BM25Indexer save / load tests
# ---------------------------------------------------------------------------

class TestBM25IndexerPersistence:
    def test_save_and_load(self, indexer, tmp_path):
        save_path = str(tmp_path / "bm25.pkl")
        indexer.save(save_path)
        assert os.path.exists(save_path)

        loaded = BM25Indexer(tokenizer_type="okt")
        loaded.load(save_path)
        assert loaded.is_ready()
        assert loaded.doc_count == indexer.doc_count

    def test_save_load_search_consistency(self, indexer, tmp_path):
        """Search results should be identical before and after save/load."""
        query = "도로 포장 균열"
        original_results = indexer.search(query, top_k=5)

        save_path = str(tmp_path / "bm25.pkl")
        indexer.save(save_path)

        loaded = BM25Indexer(tokenizer_type="okt")
        loaded.load(save_path)
        loaded_results = loaded.search(query, top_k=5)

        assert original_results == loaded_results

    def test_save_creates_parent_dirs(self, indexer, tmp_path):
        save_path = str(tmp_path / "nested" / "deep" / "bm25.pkl")
        indexer.save(save_path)
        assert os.path.exists(save_path)

    def test_load_missing_file_raises(self, tmp_path):
        idx = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(FileNotFoundError):
            idx.load(str(tmp_path / "nonexistent.pkl"))

    def test_save_before_build_raises(self, tmp_path):
        idx = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(RuntimeError):
            idx.save(str(tmp_path / "bm25.pkl"))
