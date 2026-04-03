"""
Mecab 미설치 환경에서 Okt 폴백 검증 테스트 (Issue #154).

Tests cover:
- KoreanTokenizer "auto" 모드에서 Mecab 미설치 시 Okt 자동 폴백
- Okt 폴백 상태에서 형태소 분석 정상 동작
- "mecab" 명시 모드에서 미설치 시 예외 발생
- Okt 폴백 상태에서 BM25Indexer 빌드/검색/저장/로드 전체 동작
- 토크나이저 불일치 경고 로그 검증

Note:
    이 테스트는 Java/Mecab이 설치되지 않은 CI 환경에서도 동작하도록
    konlpy.tag.Mecab과 konlpy.tag.Okt를 모두 mock으로 대체한다.
    Mecab mock은 항상 실패하고, Okt mock은 간단한 형태소 분석을 수행한다.
"""

import pickle
import re
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.inference.bm25_indexer import _STOPWORDS

SAMPLE_DOCS: List[str] = [
    "도로 포장이 파손되어 보행자 안전에 위험합니다",
    "쓰레기 무단투기로 인해 악취가 심합니다",
    "가로등이 고장나서 야간 보행이 어렵습니다",
    "주차 단속이 제대로 이루어지지 않고 있습니다",
    "소음 문제로 주민들이 불편을 겪고 있습니다",
]


def _simple_korean_morphs(text: str) -> List[str]:
    """Okt mock용 간단한 한국어 형태소 분석 시뮬레이션.

    공백 기반 분리 후 조사/어미를 제거하여 실제 Okt와 유사한 결과를 생성한다.
    """
    if not text or not text.strip():
        return []
    # 공백 분리 후 간단한 어미/조사 패턴 제거
    tokens: List[str] = []
    for word in text.split():
        # 조사/어미 패턴 제거
        cleaned = re.sub(r"(이|가|을|를|에|의|로|으로|에서|는|은|도|와|과|하다)$", "", word)
        if cleaned:
            tokens.append(cleaned)
        if cleaned != word:
            suffix = word[len(cleaned) :]
            if suffix:
                tokens.append(suffix)
    return tokens


def _create_mock_okt() -> MagicMock:
    """morphs() 메서드를 가진 Okt mock 객체 생성."""
    mock_okt = MagicMock()
    mock_okt.morphs.side_effect = _simple_korean_morphs
    return mock_okt


@pytest.fixture
def _block_mecab_allow_okt():
    """Mecab import는 실패하고 Okt는 mock으로 성공하는 환경 시뮬레이션.

    konlpy.tag.Mecab은 ImportError를 발생시키고,
    konlpy.tag.Okt는 mock 객체를 반환하도록 패치한다.
    """
    mock_okt_instance = _create_mock_okt()

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "konlpy.tag" and fromlist:
            mock_module = MagicMock()
            if "Mecab" in fromlist:
                # Mecab 속성 접근 시 AttributeError -> ImportError처럼 동작
                del mock_module.Mecab
            if "Okt" in fromlist:
                mock_module.Okt = MagicMock(return_value=mock_okt_instance)
            return mock_module
        return original_import(name, globals, locals, fromlist, level)

    import builtins

    original_import = builtins.__import__

    with patch("builtins.__import__", side_effect=_patched_import):
        yield mock_okt_instance


@pytest.fixture
def _block_all_tokenizers():
    """Mecab과 Okt 모두 import 실패하는 환경 시뮬레이션."""

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "konlpy.tag" and fromlist:
            mock_module = MagicMock()
            if "Mecab" in fromlist:
                del mock_module.Mecab
            if "Okt" in fromlist:
                # Okt 생성 시 실패
                mock_module.Okt = MagicMock(side_effect=RuntimeError("JVM not found"))
            return mock_module
        return original_import(name, globals, locals, fromlist, level)

    import builtins

    original_import = builtins.__import__

    with patch("builtins.__import__", side_effect=_patched_import):
        yield


# ---------------------------------------------------------------------------
# TestOktFallback: Mecab 미설치 시 Okt 자동 폴백 검증
# ---------------------------------------------------------------------------


class TestOktFallback:
    """Mecab 미설치 환경에서 Okt 자동 폴백 검증."""

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_auto_falls_back_to_okt_when_mecab_unavailable(self) -> None:
        """auto 모드에서 Mecab 미설치 시 tokenizer_type이 'okt'로 설정되는지 확인."""
        from src.inference.bm25_indexer import KoreanTokenizer

        tokenizer = KoreanTokenizer("auto")
        assert tokenizer.tokenizer_type == "okt"

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_okt_tokenization_produces_valid_morphemes(self) -> None:
        """Okt 폴백 상태에서 한국어 토큰화가 비어있지 않은 리스트를 반환하는지 확인."""
        from src.inference.bm25_indexer import KoreanTokenizer

        tokenizer = KoreanTokenizer("auto")
        result = tokenizer.morphs("도로 포장이 파손되어 보행자 안전에 위험합니다")

        assert isinstance(result, list)
        assert len(result) > 0
        # 불용어 필터링 확인
        for token in result:
            assert token not in _STOPWORDS, f"불용어 '{token}'이 필터링되지 않음"
        # 단일 문자 필터링 확인
        assert all(len(t) > 1 for t in result)

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_okt_filters_stopwords(self) -> None:
        """Okt 폴백 상태에서 _STOPWORDS에 정의된 불용어가 필터링되는지 확인."""
        from src.inference.bm25_indexer import KoreanTokenizer

        tokenizer = KoreanTokenizer("auto")
        # "합니다"와 "그래서"는 _STOPWORDS에 포함된 불용어
        result = tokenizer.morphs("도로 합니다 그래서 포장 보수")

        stopwords_in_input = {"합니다", "그래서"}
        for token in result:
            assert token not in stopwords_in_input

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_mecab_explicit_raises_when_unavailable(self) -> None:
        """mecab 명시 모드에서 미설치 시 RuntimeError 발생 확인."""
        from src.inference.bm25_indexer import KoreanTokenizer

        with pytest.raises(RuntimeError, match="Mecab is not installed"):
            KoreanTokenizer("mecab")

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_fallback_logs_warning(self) -> None:
        """auto 모드 Mecab -> Okt 폴백 시 warning 로그가 출력되는지 확인."""
        from src.inference.bm25_indexer import KoreanTokenizer

        with patch("src.inference.bm25_indexer.logger") as mock_logger:
            KoreanTokenizer("auto")
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any(
                "okt" in c.lower() for c in warning_calls
            ), "Mecab -> Okt 폴백 시 warning 로그가 출력되어야 함"

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_morphs_empty_input_returns_empty_list(self) -> None:
        """빈 문자열 입력 시 빈 리스트를 반환하는지 확인."""
        from src.inference.bm25_indexer import KoreanTokenizer

        tokenizer = KoreanTokenizer("auto")
        assert tokenizer.morphs("") == []
        assert tokenizer.morphs("   ") == []


# ---------------------------------------------------------------------------
# TestBM25WithOktFallback: Okt 폴백 상태에서 BM25 전체 동작 검증
# ---------------------------------------------------------------------------


class TestBM25WithOktFallback:
    """Okt 폴백 상태에서 BM25 전체 동작 검증."""

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_build_and_search_with_okt(self) -> None:
        """Okt 폴백 상태에서 BM25 인덱스 빌드 및 검색이 정상 동작하는지 확인."""
        from src.inference.bm25_indexer import BM25Indexer

        indexer = BM25Indexer(tokenizer_type="auto")
        assert indexer.tokenizer.tokenizer_type == "okt"

        indexer.build_index(SAMPLE_DOCS)
        assert indexer.is_ready()
        assert indexer.doc_count == len(SAMPLE_DOCS)

        results = indexer.search("도로 포장 파손", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

        # 결과 형식 검증: (index, score) 튜플
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)
            assert score > 0.0

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_save_load_roundtrip_with_okt(self, tmp_path) -> None:
        """Okt로 빌드한 인덱스의 save/load 왕복이 정상 동작하는지 확인."""
        from src.inference.bm25_indexer import BM25Indexer

        indexer = BM25Indexer(tokenizer_type="auto")
        indexer.build_index(SAMPLE_DOCS)

        query = "쓰레기 무단투기 악취"
        original_results = indexer.search(query, top_k=3)

        save_path = str(tmp_path / "okt_bm25.pkl")
        indexer.save(save_path)

        loaded = BM25Indexer(tokenizer_type="auto")
        loaded.load(save_path)
        assert loaded.is_ready()
        assert loaded.doc_count == indexer.doc_count

        loaded_results = loaded.search(query, top_k=3)
        assert original_results == loaded_results

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_tokenizer_mismatch_warning_on_load(self, tmp_path) -> None:
        """mecab으로 빌드된 인덱스를 okt 환경에서 로드 시 warning 로그 발생 확인."""
        from src.inference.bm25_indexer import BM25Indexer

        indexer = BM25Indexer(tokenizer_type="auto")
        indexer.build_index(SAMPLE_DOCS)

        save_path = str(tmp_path / "mecab_index.pkl")
        indexer.save(save_path)

        # pickle 내 tokenizer_type을 "mecab"으로 변경하여 불일치 시뮬레이션
        with open(save_path, "rb") as f:
            payload = pickle.loads(f.read())
        payload["tokenizer_type"] = "mecab"
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

        loaded = BM25Indexer(tokenizer_type="auto")
        with patch("src.inference.bm25_indexer.logger") as mock_logger:
            loaded.load(save_path)
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any(
                "mismatch" in c.lower() for c in warning_calls
            ), "토크나이저 불일치 시 warning 로그가 출력되어야 함"

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_search_relevance_with_okt(self) -> None:
        """Okt 폴백 상태에서 검색 관련성이 유지되는지 확인."""
        from src.inference.bm25_indexer import BM25Indexer

        indexer = BM25Indexer(tokenizer_type="auto")
        indexer.build_index(SAMPLE_DOCS)

        # "가로등 고장 야간" -> 3번째 문서(idx=2)가 가장 관련성 높아야 함
        results = indexer.search("가로등 고장 야간", top_k=3)
        assert len(results) > 0
        top_idx, _ = results[0]
        assert top_idx == 2, f"기대: idx=2, 실제: idx={top_idx}"

    @pytest.mark.usefixtures("_block_mecab_allow_okt")
    def test_search_scores_sorted_descending_with_okt(self) -> None:
        """Okt 폴백 상태에서 검색 결과 점수가 내림차순인지 확인."""
        from src.inference.bm25_indexer import BM25Indexer

        indexer = BM25Indexer(tokenizer_type="auto")
        indexer.build_index(SAMPLE_DOCS)

        results = indexer.search("주민 불편 소음 문제", top_k=5)
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)
