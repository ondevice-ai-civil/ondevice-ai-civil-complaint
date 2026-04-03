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
import pickle
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

    def test_save_flat_filename_no_crash(self, indexer, tmp_path, monkeypatch):
        """save() with a flat filename (no dir component) must not crash."""
        monkeypatch.chdir(tmp_path)
        indexer.save("bm25_flat.pkl")
        assert os.path.exists(tmp_path / "bm25_flat.pkl")

    def test_load_corrupt_pickle_raises(self, tmp_path):
        """Loading a corrupt file raises ValueError, not a raw UnpicklingError."""
        corrupt = tmp_path / "corrupt.pkl"
        corrupt.write_bytes(b"not a valid pickle")
        idx = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(ValueError, match="corrupt or incompatible"):
            idx.load(str(corrupt))

    def test_load_incompatible_schema_raises(self, tmp_path):
        """A pickle missing required keys raises ValueError with a clear message."""
        bad_path = tmp_path / "bad.pkl"
        with open(bad_path, "wb") as f:
            # Include valid version but missing bm25/tokenized_corpus/doc_count
            pickle.dump({"version": 1, "some_key": "no bm25 here"}, f)
        idx = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(ValueError, match="incompatible schema"):
            idx.load(str(bad_path))

    def test_load_tokenizer_mismatch_warns(self, indexer, tmp_path):
        """Loading an index built with a different tokenizer emits a warning."""
        from unittest.mock import patch

        save_path = str(tmp_path / "bm25.pkl")
        indexer.save(save_path)

        # Patch saved tokenizer_type to simulate a mismatch
        with open(save_path, "rb") as f:
            payload = pickle.loads(f.read())
        payload["tokenizer_type"] = "mecab"
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

        loaded = BM25Indexer(tokenizer_type="okt")
        with patch("src.inference.bm25_indexer.logger") as mock_logger:
            loaded.load(save_path)
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("mismatch" in c.lower() for c in warning_calls)

    def test_load_version_mismatch_raises(self, indexer, tmp_path):
        """Loading an index with a different payload version raises ValueError."""
        save_path = str(tmp_path / "bm25.pkl")
        indexer.save(save_path)

        # Patch version to simulate mismatch
        with open(save_path, "rb") as f:
            payload = pickle.loads(f.read())
        payload["version"] = 999
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

        loaded = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(ValueError, match="version mismatch"):
            loaded.load(save_path)

    def test_hmac_sign_and_verify(self, indexer, tmp_path, monkeypatch):
        """When HMAC key is set, save() writes .sig and load() verifies it."""
        monkeypatch.setenv("BM25_INDEX_HMAC_KEY", "test-secret-key-1234")
        save_path = str(tmp_path / "bm25.pkl")
        indexer.save(save_path)

        # .sig file should exist
        sig_path = save_path + ".sig"
        assert os.path.exists(sig_path)

        # load should succeed with correct sig
        loaded = BM25Indexer(tokenizer_type="okt")
        loaded.load(save_path)
        assert loaded.is_ready()
        assert loaded.doc_count == indexer.doc_count

    def test_hmac_tampered_file_raises(self, indexer, tmp_path, monkeypatch):
        """Tampered index file fails HMAC verification."""
        monkeypatch.setenv("BM25_INDEX_HMAC_KEY", "test-secret-key-1234")
        save_path = str(tmp_path / "bm25.pkl")
        indexer.save(save_path)

        # Tamper with the index file
        with open(save_path, "ab") as f:
            f.write(b"tampered")

        loaded = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(ValueError, match="HMAC verification failed"):
            loaded.load(save_path)

    def test_hmac_missing_sig_raises(self, indexer, tmp_path, monkeypatch):
        """Missing .sig file when HMAC key is set raises ValueError."""
        # Save without HMAC key
        save_path = str(tmp_path / "bm25.pkl")
        indexer.save(save_path)

        # Now try loading with HMAC key — .sig doesn't exist
        monkeypatch.setenv("BM25_INDEX_HMAC_KEY", "test-secret-key-1234")
        loaded = BM25Indexer(tokenizer_type="okt")
        with pytest.raises(ValueError, match="signature file missing"):
            loaded.load(save_path)


# ---------------------------------------------------------------------------
# Bug regression tests (found by agents)
# ---------------------------------------------------------------------------


class TestBM25IndexerRegressions:
    def test_all_empty_token_docs_raises(self):
        """All-stopword corpus raises ValueError, not ZeroDivisionError."""
        idx = BM25Indexer(tokenizer_type="okt")
        all_stopword_docs = ["합니다 입니다 됩니다", "있습니다 없습니다 그래서"]
        with pytest.raises(ValueError, match="empty token lists"):
            idx.build_index(all_stopword_docs)

    def test_search_top_k_exceeds_doc_count(self, indexer):
        """top_k larger than corpus size does not crash."""
        results = indexer.search("도로 포장", top_k=10_000)
        assert len(results) <= indexer.doc_count

    def test_rebuild_replaces_index(self):
        """build_index() called twice replaces the previous index."""
        idx = BM25Indexer(tokenizer_type="okt")
        idx.build_index(["첫번째 문서"])
        idx.build_index(SAMPLE_DOCUMENTS)
        assert idx.doc_count == len(SAMPLE_DOCUMENTS)
        assert idx.is_ready()

    def test_search_stopwords_only_query(self, indexer):
        """Query of pure stopwords returns empty list."""
        results = indexer.search("합니다 입니다 됩니다", top_k=5)
        assert results == []

    def test_repr(self, indexer):
        r = repr(indexer)
        assert "BM25Indexer" in r
        assert "ready=True" in r

    def test_build_from_jsonl_malformed_lines(self, tmp_path):
        """Malformed JSONL lines are skipped; valid lines still build the index."""
        path = tmp_path / "partial.jsonl"
        path.write_text(
            '{"text": "정상 문서입니다 민원"}\nNOT_JSON\n{"text": "두번째 문서입니다 신고"}\n',
            encoding="utf-8",
        )
        idx = BM25Indexer(tokenizer_type="okt")
        idx.build_index_from_jsonl(str(path))
        assert idx.doc_count == 2

    def test_build_from_jsonl_complaint_field(self, tmp_path):
        """JSONL records using 'complaint' field (not 'text') are indexed."""
        path = tmp_path / "complaints.jsonl"
        records = [{"complaint": doc} for doc in SAMPLE_DOCUMENTS]
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
            encoding="utf-8",
        )
        idx = BM25Indexer(tokenizer_type="okt")
        idx.build_index_from_jsonl(str(path))
        assert idx.doc_count == len(SAMPLE_DOCUMENTS)

    def test_build_from_jsonl_auto_extracts_template(self, jsonl_file):
        """JSONL with EXAONE templates in 'text' field auto-extracts complaint."""
        idx = BM25Indexer(tokenizer_type="okt")
        idx.build_index_from_jsonl(jsonl_file)
        assert idx.is_ready()
        # Should find "도로 포장 균열" in extracted text, not in raw template
        results = idx.search("도로 포장 균열", top_k=3)
        assert len(results) > 0
        top_idx, _ = results[0]
        assert top_idx == 0


# ---------------------------------------------------------------------------
# _extract_complaint_from_template unit tests
# ---------------------------------------------------------------------------


class TestExtractComplaintFromTemplate:
    def test_extracts_minwon_naeyo(self):
        text = "[|system|]sys[|endofturn|][|user|]민원 내용: 도로 포장 균열[|endofturn|][|assistant|]답변[|endofturn|]"
        assert BM25Indexer._extract_complaint_from_template(text) == "도로 포장 균열"

    def test_extracts_user_without_minwon_prefix(self):
        text = "[|system|]sys[|endofturn|][|user|]그냥 내용입니다[|endofturn|]"
        assert BM25Indexer._extract_complaint_from_template(text) == "그냥 내용입니다"

    def test_returns_original_when_no_template(self):
        assert BM25Indexer._extract_complaint_from_template("일반 텍스트") == "일반 텍스트"

    def test_returns_empty_string_on_empty_input(self):
        assert BM25Indexer._extract_complaint_from_template("") == ""
