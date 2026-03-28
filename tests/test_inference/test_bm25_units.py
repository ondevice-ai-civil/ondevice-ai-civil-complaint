"""
BM25Indexer 단위 테스트.

KoreanTokenizer, BM25 인덱스 빌드/검색/저장/로드, HMAC 검증을 테스트한다.
konlpy 의존성을 Mock으로 대체하여 GPU 없이 실행 가능.
"""

import hashlib
import hmac
import json
import os
import pickle
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 무거운 의존성 mock 등록
# ---------------------------------------------------------------------------
# konlpy mock
_konlpy_mock = MagicMock()
sys.modules.setdefault("konlpy", _konlpy_mock)
sys.modules.setdefault("konlpy.tag", _konlpy_mock)

# rank_bm25 mock - BM25Okapi를 실제처럼 동작하도록 설정
_bm25_mock_module = MagicMock()
sys.modules.setdefault("rank_bm25", _bm25_mock_module)

# faiss mock
_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, MagicMock)
if not _faiss_is_real:
    _faiss_mock = MagicMock()
    _faiss_mock.IndexIVFFlat = type("IndexIVFFlat", (), {})
    _faiss_mock.IndexFlatIP = type("IndexFlatIP", (), {})
    sys.modules.setdefault("faiss", _faiss_mock)

from src.inference.bm25_indexer import _STOPWORDS, BM25Indexer, KoreanTokenizer

# ---------------------------------------------------------------------------
# KoreanTokenizer 테스트
# ---------------------------------------------------------------------------


class TestKoreanTokenizer:
    def test_morphs_filters_short_tokens(self):
        """한 글자 토큰을 필터링한다."""
        mock_tagger = MagicMock()
        mock_tagger.morphs.return_value = ["도로", "가", "파손", "됨"]

        with patch("src.inference.bm25_indexer.KoreanTokenizer._init_tokenizer"):
            tokenizer = KoreanTokenizer()
            tokenizer._tagger = mock_tagger
            tokenizer.tokenizer_type = "okt"

        result = tokenizer.morphs("도로가 파손됨")
        # '가'는 1글자이므로 필터링
        assert "가" not in result
        assert "도로" in result

    def test_morphs_filters_stopwords(self):
        """불용어를 필터링한다."""
        mock_tagger = MagicMock()
        mock_tagger.morphs.return_value = ["도로", "합니다", "파손", "있습니다"]

        with patch("src.inference.bm25_indexer.KoreanTokenizer._init_tokenizer"):
            tokenizer = KoreanTokenizer()
            tokenizer._tagger = mock_tagger
            tokenizer.tokenizer_type = "okt"

        result = tokenizer.morphs("도로 합니다 파손 있습니다")
        assert "합니다" not in result
        assert "있습니다" not in result
        assert "도로" in result

    def test_morphs_empty_string(self):
        """빈 문자열은 빈 리스트를 반환한다."""
        with patch("src.inference.bm25_indexer.KoreanTokenizer._init_tokenizer"):
            tokenizer = KoreanTokenizer()
            tokenizer._tagger = MagicMock()
            tokenizer.tokenizer_type = "okt"

        assert tokenizer.morphs("") == []
        assert tokenizer.morphs("   ") == []

    def test_morphs_fallback_on_error(self):
        """토크나이저 에러 시 공백 분리로 폴백한다."""
        mock_tagger = MagicMock()
        mock_tagger.morphs.side_effect = RuntimeError("tokenizer error")

        with patch("src.inference.bm25_indexer.KoreanTokenizer._init_tokenizer"):
            tokenizer = KoreanTokenizer()
            tokenizer._tagger = mock_tagger
            tokenizer.tokenizer_type = "okt"

        result = tokenizer.morphs("도로 파손 신고")
        assert "도로" in result
        assert "파손" in result

    def test_stopwords_defined(self):
        """불용어 집합이 frozenset으로 정의되어 있다."""
        assert isinstance(_STOPWORDS, frozenset)
        assert "합니다" in _STOPWORDS
        assert "입니다" in _STOPWORDS


# ---------------------------------------------------------------------------
# BM25Indexer 초기화 테스트
# ---------------------------------------------------------------------------


class TestBM25IndexerInit:
    def test_init_creates_tokenizer(self):
        """초기화 시 토크나이저를 생성한다."""
        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            indexer = BM25Indexer(tokenizer_type="okt")
            mock_cls.assert_called_once_with("okt")

    def test_not_ready_initially(self):
        """초기 상태에서 is_ready()는 False."""
        with patch("src.inference.bm25_indexer.KoreanTokenizer"):
            indexer = BM25Indexer()
        assert indexer.is_ready() is False

    def test_doc_count_zero(self):
        """초기 doc_count는 0."""
        with patch("src.inference.bm25_indexer.KoreanTokenizer"):
            indexer = BM25Indexer()
        assert indexer.doc_count == 0

    def test_repr(self):
        """__repr__이 올바른 형식을 반환한다."""
        with patch("src.inference.bm25_indexer.KoreanTokenizer"):
            indexer = BM25Indexer()
        repr_str = repr(indexer)
        assert "BM25Indexer" in repr_str
        assert "docs=0" in repr_str


# ---------------------------------------------------------------------------
# build_index 테스트
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def _make_indexer(self):
        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            tokenizer = MagicMock()
            tokenizer.morphs.side_effect = lambda text: text.split() if text else []
            tokenizer.tokenizer_type = "okt"
            mock_cls.return_value = tokenizer
            indexer = BM25Indexer()
        return indexer

    def test_build_index_basic(self):
        """기본 인덱스 빌드."""
        indexer = self._make_indexer()
        mock_bm25_cls = MagicMock()
        with patch("src.inference.bm25_indexer.BM25Okapi", mock_bm25_cls):
            indexer.build_index(["도로 파손 신고", "가로등 고장 신고"])

        assert indexer.doc_count == 2
        assert indexer.is_ready()
        mock_bm25_cls.assert_called_once()

    def test_build_index_empty_raises(self):
        """빈 문서 리스트는 ValueError를 발생시킨다."""
        indexer = self._make_indexer()
        with pytest.raises(ValueError, match="empty"):
            indexer.build_index([])

    def test_build_index_all_empty_tokens_raises(self):
        """모든 문서의 토큰이 비어있으면 ValueError를 발생시킨다."""
        indexer = self._make_indexer()
        # morphs가 항상 빈 리스트를 반환하도록 설정
        indexer.tokenizer.morphs.side_effect = lambda text: []

        with pytest.raises(ValueError, match="empty token"):
            indexer.build_index(["", "   "])

    def test_rebuild_replaces_existing(self):
        """기존 인덱스가 있으면 교체한다."""
        indexer = self._make_indexer()
        mock_bm25_cls = MagicMock()
        with patch("src.inference.bm25_indexer.BM25Okapi", mock_bm25_cls):
            indexer.build_index(["문서1"])
            indexer.build_index(["문서A", "문서B"])

        assert indexer.doc_count == 2


# ---------------------------------------------------------------------------
# search 테스트
# ---------------------------------------------------------------------------


class TestSearch:
    def _make_ready_indexer(self):
        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            tokenizer = MagicMock()
            tokenizer.morphs.side_effect = lambda text: text.split() if text.strip() else []
            tokenizer.tokenizer_type = "okt"
            mock_cls.return_value = tokenizer
            indexer = BM25Indexer()
        # Simulate built index
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([5.0, 3.0, 0.0, -1.0])
        indexer.bm25 = mock_bm25
        indexer._doc_count = 4
        return indexer

    def test_search_returns_positive_scores(self):
        """양수 점수만 반환한다."""
        indexer = self._make_ready_indexer()
        results = indexer.search("도로 파손", top_k=10)

        # 양수 점수(5.0, 3.0)만 반환
        assert len(results) == 2
        assert all(score > 0 for _, score in results)

    def test_search_sorted_desc(self):
        """점수 내림차순으로 정렬된다."""
        indexer = self._make_ready_indexer()
        results = indexer.search("도로", top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self):
        """빈 쿼리는 빈 결과를 반환한다."""
        indexer = self._make_ready_indexer()
        assert indexer.search("") == []
        assert indexer.search("   ") == []

    def test_search_not_built_raises(self):
        """인덱스 미빌드 시 RuntimeError를 발생시킨다."""
        with patch("src.inference.bm25_indexer.KoreanTokenizer"):
            indexer = BM25Indexer()
        with pytest.raises(RuntimeError, match="not built"):
            indexer.search("테스트")

    def test_search_respects_top_k(self):
        """top_k에 맞게 결과 수를 제한한다."""
        indexer = self._make_ready_indexer()
        results = indexer.search("도로", top_k=1)
        assert len(results) <= 1

    def test_search_tokenized_empty_query(self):
        """토크나이저가 빈 결과를 반환하면 빈 리스트를 반환한다."""
        indexer = self._make_ready_indexer()
        indexer.tokenizer.morphs.side_effect = lambda text: []
        results = indexer.search("a")  # 단일 문자 -> 필터링됨
        assert results == []


# ---------------------------------------------------------------------------
# save / load 테스트
# ---------------------------------------------------------------------------


class _PicklableBM25:
    """pickle 가능한 BM25 대체 객체."""

    def __init__(self):
        self.corpus_size = 2

    def get_scores(self, query):
        return np.array([1.0, 0.5])


class TestSaveLoad:
    def _make_built_indexer(self):
        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            tokenizer = MagicMock()
            tokenizer.morphs.side_effect = lambda text: text.split() if text else []
            tokenizer.tokenizer_type = "okt"
            mock_cls.return_value = tokenizer
            indexer = BM25Indexer()

        # pickle 가능한 객체를 사용
        indexer.bm25 = _PicklableBM25()
        indexer._tokenized_corpus = [["도로", "파손"], ["가로등", "고장"]]
        indexer._doc_count = 2
        return indexer

    def test_save_creates_file(self, tmp_path):
        """인덱스를 파일로 저장한다."""
        indexer = self._make_built_indexer()
        path = str(tmp_path / "index.pkl")
        indexer.save(path)
        assert os.path.exists(path)

    def test_save_not_built_raises(self, tmp_path):
        """인덱스 미빌드 시 RuntimeError를 발생시킨다."""
        with patch("src.inference.bm25_indexer.KoreanTokenizer"):
            indexer = BM25Indexer()
        with pytest.raises(RuntimeError, match="not built"):
            indexer.save(str(tmp_path / "index.pkl"))

    def test_save_creates_parent_dir(self, tmp_path):
        """부모 디렉토리가 없으면 생성한다."""
        indexer = self._make_built_indexer()
        path = str(tmp_path / "subdir" / "deep" / "index.pkl")
        indexer.save(path)
        assert os.path.exists(path)

    def test_load_nonexistent_raises(self, tmp_path):
        """존재하지 않는 파일은 FileNotFoundError를 발생시킨다."""
        with patch("src.inference.bm25_indexer.KoreanTokenizer"):
            indexer = BM25Indexer()
        with pytest.raises(FileNotFoundError):
            indexer.load(str(tmp_path / "nonexistent.pkl"))

    def test_load_corrupt_file_raises(self, tmp_path):
        """손상된 파일은 ValueError를 발생시킨다."""
        path = str(tmp_path / "corrupt.pkl")
        with open(path, "wb") as f:
            f.write(b"not a pickle file")

        with patch("src.inference.bm25_indexer.KoreanTokenizer"):
            indexer = BM25Indexer()
        with pytest.raises(ValueError, match="corrupt"):
            indexer.load(path)

    def test_load_version_mismatch_raises(self, tmp_path):
        """버전 불일치 시 ValueError를 발생시킨다."""
        payload = {
            "version": 999,  # 잘못된 버전
            "bm25": _PicklableBM25(),
            "tokenized_corpus": [],
            "doc_count": 0,
        }
        path = str(tmp_path / "wrong_version.pkl")
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        with patch("src.inference.bm25_indexer.KoreanTokenizer"):
            indexer = BM25Indexer()
        with pytest.raises(ValueError, match="version mismatch"):
            indexer.load(path)

    def test_save_load_roundtrip(self, tmp_path):
        """저장 후 로드하면 동일한 상태를 복원한다."""
        indexer = self._make_built_indexer()
        path = str(tmp_path / "roundtrip.pkl")
        indexer.save(path)

        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            tokenizer = MagicMock()
            tokenizer.tokenizer_type = "okt"
            mock_cls.return_value = tokenizer
            indexer2 = BM25Indexer()

        indexer2.load(path)
        assert indexer2.doc_count == 2
        assert indexer2.is_ready()


# ---------------------------------------------------------------------------
# HMAC 검증 테스트
# ---------------------------------------------------------------------------


class TestHMAC:
    def _make_built_indexer(self):
        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            tokenizer = MagicMock()
            tokenizer.morphs.side_effect = lambda text: text.split() if text else []
            tokenizer.tokenizer_type = "okt"
            mock_cls.return_value = tokenizer
            indexer = BM25Indexer()

        indexer.bm25 = _PicklableBM25()
        indexer._tokenized_corpus = [["테스트"]]
        indexer._doc_count = 1
        return indexer

    def test_save_with_hmac_creates_sig_file(self, tmp_path):
        """HMAC 키 설정 시 .sig 파일을 생성한다."""
        indexer = self._make_built_indexer()
        path = str(tmp_path / "hmac_test.pkl")

        with patch.dict(os.environ, {"BM25_INDEX_HMAC_KEY": "test-secret"}):
            indexer.save(path)

        assert os.path.exists(path + ".sig")

    def test_load_with_hmac_verifies(self, tmp_path):
        """HMAC 검증이 성공하면 정상 로드한다."""
        indexer = self._make_built_indexer()
        path = str(tmp_path / "hmac_verify.pkl")

        with patch.dict(os.environ, {"BM25_INDEX_HMAC_KEY": "test-secret"}):
            indexer.save(path)

        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            tokenizer = MagicMock()
            tokenizer.tokenizer_type = "okt"
            mock_cls.return_value = tokenizer
            indexer2 = BM25Indexer()

        with patch.dict(os.environ, {"BM25_INDEX_HMAC_KEY": "test-secret"}):
            indexer2.load(path)

        assert indexer2.is_ready()

    def test_load_hmac_mismatch_raises(self, tmp_path):
        """HMAC 검증 실패 시 ValueError를 발생시킨다."""
        indexer = self._make_built_indexer()
        path = str(tmp_path / "hmac_bad.pkl")

        with patch.dict(os.environ, {"BM25_INDEX_HMAC_KEY": "key1"}):
            indexer.save(path)

        # 다른 키로 로드 시도
        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            tokenizer = MagicMock()
            tokenizer.tokenizer_type = "okt"
            mock_cls.return_value = tokenizer
            indexer2 = BM25Indexer()

        with patch.dict(os.environ, {"BM25_INDEX_HMAC_KEY": "wrong-key"}):
            with pytest.raises(ValueError, match="HMAC"):
                indexer2.load(path)

    def test_load_missing_sig_file_raises(self, tmp_path):
        """HMAC 키 설정 시 .sig 파일이 없으면 ValueError를 발생시킨다."""
        indexer = self._make_built_indexer()
        path = str(tmp_path / "no_sig.pkl")

        # HMAC 없이 저장
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BM25_INDEX_HMAC_KEY", None)
            indexer.save(path)

        # HMAC 키 설정 후 로드 시도 (sig 파일 없음)
        with patch("src.inference.bm25_indexer.KoreanTokenizer") as mock_cls:
            tokenizer = MagicMock()
            tokenizer.tokenizer_type = "okt"
            mock_cls.return_value = tokenizer
            indexer2 = BM25Indexer()

        with patch.dict(os.environ, {"BM25_INDEX_HMAC_KEY": "some-key"}):
            with pytest.raises(ValueError, match="signature file missing"):
                indexer2.load(path)


# ---------------------------------------------------------------------------
# _extract_complaint_from_template 테스트
# ---------------------------------------------------------------------------


class TestExtractComplaintFromTemplate:
    def test_extract_with_label(self):
        """민원 내용: 라벨이 있으면 추출한다."""
        text = "[|user|]민원 내용: 도로가 파손되었습니다.[|endofturn|]"
        result = BM25Indexer._extract_complaint_from_template(text)
        assert result == "도로가 파손되었습니다."

    def test_extract_without_label(self):
        """라벨 없이 user 블록 전체를 반환한다."""
        text = "[|user|]도로 파손 신고[|endofturn|]"
        result = BM25Indexer._extract_complaint_from_template(text)
        assert result == "도로 파손 신고"

    def test_extract_no_template(self):
        """템플릿이 아닌 텍스트는 그대로 반환한다."""
        text = "일반 텍스트"
        result = BM25Indexer._extract_complaint_from_template(text)
        assert result == "일반 텍스트"

    def test_extract_empty_string(self):
        """빈 문자열은 그대로 반환한다."""
        result = BM25Indexer._extract_complaint_from_template("")
        assert result == ""
