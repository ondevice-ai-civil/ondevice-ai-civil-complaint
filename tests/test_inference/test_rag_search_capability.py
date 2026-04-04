"""rag_search capability 단위 테스트.

RagSearchParams 검증, RagSearchCapability 실행 및 fallback 정책을 검증한다.
"""

from __future__ import annotations

import asyncio

import pytest

try:
    from src.inference.graph.capabilities import RagSearchCapability, RagSearchParams
    from src.inference.graph.capabilities.base import LookupResult

    CAPABILITIES_AVAILABLE = True
except ImportError:
    CAPABILITIES_AVAILABLE = False

requires_capabilities = pytest.mark.skipif(
    not CAPABILITIES_AVAILABLE,
    reason="capabilities 패키지 미구현",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_search_results():
    """SearchResult.model_dump() 형태의 모의 검색 결과."""
    return [
        {
            "doc_id": "CASE-001",
            "source_type": "case",
            "title": "도로 보수 민원 사례",
            "content": "도로 파손으로 인한 민원 처리 결과입니다.",
            "score": 0.85,
            "reliability_score": 0.9,
            "metadata": {"file_path": "/data/cases/case001.json", "page": 1},
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "doc_id": "LAW-042",
            "source_type": "law",
            "title": "도로법 제36조",
            "content": "도로 관리자는 도로 유지·보수를 하여야 한다.",
            "score": 0.72,
            "reliability_score": 1.0,
            "metadata": {"file_path": "/data/laws/law042.json", "page": 3},
            "chunk_index": 2,
            "total_chunks": 5,
        },
    ]


@pytest.fixture
def low_score_results():
    """저신뢰도 검색 결과."""
    return [
        {
            "doc_id": "NOTICE-001",
            "source_type": "notice",
            "title": "공지사항",
            "content": "관련 없는 공지",
            "score": 0.1,
            "reliability_score": 0.5,
            "metadata": {},
            "chunk_index": 0,
            "total_chunks": 1,
        },
        {
            "doc_id": "NOTICE-002",
            "source_type": "notice",
            "title": "공지사항 2",
            "content": "역시 관련 없는 공지",
            "score": 0.15,
            "reliability_score": 0.5,
            "metadata": {},
            "chunk_index": 0,
            "total_chunks": 1,
        },
    ]


def _make_execute_fn(return_value):
    """모의 execute_fn을 생성한다."""

    async def _fn(query, context, session):
        return return_value

    return _fn


def _make_slow_execute_fn(delay: float):
    """지연되는 모의 execute_fn을 생성한다."""

    async def _fn(query, context, session):
        await asyncio.sleep(delay)
        return {"query": query, "results": [], "context_text": ""}

    return _fn


def _make_raising_execute_fn(exc):
    """예외를 발생시키는 모의 execute_fn을 생성한다."""

    async def _fn(query, context, session):
        raise exc

    return _fn


# ===========================================================================
# TestRagSearchParams — 파라미터 validator
# ===========================================================================
@requires_capabilities
class TestRagSearchParams:
    """RagSearchParams validator 검증."""

    def test_valid_params_pass(self):
        params = RagSearchParams(query="민원 처리 절차")
        assert params.validate() is None

    def test_empty_query_fails(self):
        params = RagSearchParams(query="")
        assert params.validate() is not None

    def test_long_query_fails(self):
        params = RagSearchParams(query="가" * 2001)
        error = params.validate()
        assert error is not None
        assert "2000" in error

    def test_alias_normalization_top_k(self):
        params = RagSearchParams.from_context("테스트", {"top_k": 10})
        assert params.top_k == 10

    def test_alias_normalization_rag_top_k(self):
        params = RagSearchParams.from_context("테스트", {"rag_top_k": 8})
        assert params.top_k == 8

    def test_alias_normalization_count(self):
        params = RagSearchParams.from_context("테스트", {"count": 3})
        assert params.top_k == 3

    def test_alias_normalization_filters(self):
        params = RagSearchParams.from_context("테스트", {"filters": ["case", "law"]})
        assert params.source_types == ["case", "law"]

    def test_alias_normalization_source_types(self):
        params = RagSearchParams.from_context("테스트", {"source_types": ["manual"]})
        assert params.source_types == ["manual"]

    def test_alias_normalization_min_confidence(self):
        params = RagSearchParams.from_context("테스트", {"min_confidence": 0.5})
        assert params.min_confidence == 0.5

    def test_alias_normalization_score_threshold(self):
        params = RagSearchParams.from_context("테스트", {"score_threshold": 0.6})
        assert params.min_confidence == 0.6

    def test_top_k_clamped_upper(self):
        params = RagSearchParams.from_context("테스트", {"top_k": 999})
        assert params.top_k == 50

    def test_top_k_clamped_lower(self):
        params = RagSearchParams.from_context("테스트", {"top_k": -1})
        assert params.top_k == 1

    def test_min_confidence_clamped_upper(self):
        params = RagSearchParams.from_context("테스트", {"min_confidence": 5.0})
        assert params.min_confidence == 1.0

    def test_min_confidence_clamped_lower(self):
        params = RagSearchParams.from_context("테스트", {"min_confidence": -0.5})
        assert params.min_confidence == 0.0

    def test_invalid_source_type_fails(self):
        params = RagSearchParams(query="테스트", source_types=["case", "invalid_type"])
        error = params.validate()
        assert error is not None
        assert "invalid_type" in error

    def test_default_source_types(self):
        params = RagSearchParams.from_context("테스트", {})
        assert set(params.source_types) == {"case", "law", "manual", "notice"}

    def test_query_stripped(self):
        params = RagSearchParams.from_context("  공백 포함 쿼리  ", {})
        assert params.query == "공백 포함 쿼리"


# ===========================================================================
# TestRagSearchCapabilityMetadata
# ===========================================================================
@requires_capabilities
class TestRagSearchCapabilityMetadata:
    """RagSearchCapability metadata 검증."""

    @pytest.fixture
    def capability(self):
        return RagSearchCapability(execute_fn=_make_execute_fn({}))

    def test_metadata_name(self, capability):
        assert capability.metadata.name == "rag_search"

    def test_metadata_has_description(self, capability):
        assert capability.metadata.description

    def test_metadata_has_approval_summary(self, capability):
        assert capability.metadata.approval_summary

    def test_metadata_provider(self, capability):
        assert capability.metadata.provider == "local_vectordb"


# ===========================================================================
# TestRagSearchCapabilityExecute — 실행 및 fallback 정책
# ===========================================================================
@requires_capabilities
class TestRagSearchCapabilityExecute:
    """RagSearchCapability.execute() 동작 검증."""

    @pytest.mark.asyncio
    async def test_successful_search_returns_normalized_results(self, mock_search_results):
        """성공 시 정규화된 결과를 반환한다."""
        fn = _make_execute_fn(
            {
                "query": "도로 민원",
                "results": mock_search_results,
                "context_text": "### 로컬 문서 검색 결과:",
            }
        )
        cap = RagSearchCapability(execute_fn=fn)
        result = await cap.execute("도로 민원", {}, None)

        assert result.success is True
        assert len(result.results) == 2
        assert result.empty_reason is None

        first = result.results[0]
        assert "excerpt" in first
        assert "file_path" in first
        assert "page" in first
        assert "score" in first
        assert "source_type" in first
        assert first["source_type"] == "case"
        assert first["file_path"] == "/data/cases/case001.json"
        assert first["page"] == 1

    @pytest.mark.asyncio
    async def test_empty_results_returns_no_match(self):
        """빈 결과는 no_match로 처리한다."""
        fn = _make_execute_fn({"query": "없는 민원", "results": [], "context_text": ""})
        cap = RagSearchCapability(execute_fn=fn)
        result = await cap.execute("없는 민원", {}, None)

        assert result.success is True
        assert result.empty_reason == "no_match"
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_validation_error_on_empty_query(self):
        """빈 query는 validation error를 반환한다."""
        fn = _make_execute_fn({})
        cap = RagSearchCapability(execute_fn=fn)
        result = await cap.execute("", {}, None)

        assert result.success is False
        assert result.empty_reason == "validation_error"
        assert "비어있습니다" in result.error

    @pytest.mark.asyncio
    async def test_error_from_execute_fn(self):
        """execute_fn이 error dict를 반환하면 provider_error로 처리한다."""
        fn = _make_execute_fn({"error": "인덱스 로드 실패"})
        cap = RagSearchCapability(execute_fn=fn)
        result = await cap.execute("테스트", {}, None)

        assert result.success is False
        assert result.empty_reason == "provider_error"
        assert "인덱스 로드 실패" in result.error

    @pytest.mark.asyncio
    async def test_exception_from_execute_fn(self):
        """execute_fn에서 예외가 발생하면 provider_error를 반환한다."""
        fn = _make_raising_execute_fn(RuntimeError("FAISS 오류"))
        cap = RagSearchCapability(execute_fn=fn)
        result = await cap.execute("테스트", {}, None)

        assert result.success is False
        assert result.empty_reason == "provider_error"
        assert "FAISS 오류" in result.error

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self):
        """타임아웃 시 provider_error를 반환한다."""
        from src.inference.graph.capabilities.base import CapabilityMetadata

        slow_fn = _make_slow_execute_fn(delay=1.0)

        class _ShortTimeoutRag(RagSearchCapability):
            @property
            def metadata(self):
                return CapabilityMetadata(
                    name="rag_search",
                    description="test",
                    approval_summary="test",
                    provider="local_vectordb",
                    timeout_sec=0.1,
                )

        cap = _ShortTimeoutRag(execute_fn=slow_fn)
        result = await cap.execute("테스트", {}, None)

        assert result.success is False
        assert result.empty_reason == "provider_error"
        assert "타임아웃" in result.error

    @pytest.mark.asyncio
    async def test_low_confidence_all_results(self, low_score_results):
        """모든 결과가 저신뢰도이면 low_confidence로 분류한다."""
        fn = _make_execute_fn(
            {
                "query": "관련 없는 쿼리",
                "results": low_score_results,
                "context_text": "",
            }
        )
        cap = RagSearchCapability(execute_fn=fn, low_confidence_threshold=0.3)
        result = await cap.execute("관련 없는 쿼리", {}, None)

        assert result.success is True
        assert result.empty_reason == "low_confidence"
        assert len(result.results) == 2  # 원본 결과 유지

    @pytest.mark.asyncio
    async def test_low_confidence_partial_filter(self, mock_search_results):
        """혼합 신뢰도에서 confident 결과만 반환한다."""
        mixed = [
            {**mock_search_results[0], "score": 0.85},  # above threshold
            {**mock_search_results[1], "score": 0.1},  # below threshold
        ]
        fn = _make_execute_fn(
            {
                "query": "도로 민원",
                "results": mixed,
                "context_text": "",
            }
        )
        cap = RagSearchCapability(execute_fn=fn, low_confidence_threshold=0.3)
        result = await cap.execute("도로 민원", {}, None)

        assert result.success is True
        assert result.empty_reason is None
        assert len(result.results) == 1
        assert result.results[0]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_citations_generated(self, mock_search_results):
        """성공 결과에서 citations가 생성된다."""
        fn = _make_execute_fn(
            {
                "query": "도로 민원",
                "results": mock_search_results,
                "context_text": "",
            }
        )
        cap = RagSearchCapability(execute_fn=fn)
        result = await cap.execute("도로 민원", {}, None)

        assert len(result.citations) == 2
        citation = result.citations[0]
        assert "source_type" in citation
        assert "doc_id" in citation
        assert "title" in citation
        assert "score" in citation
        assert "excerpt" in citation

    @pytest.mark.asyncio
    async def test_context_text_passthrough(self, mock_search_results):
        """context_text가 그대로 전달된다."""
        fn = _make_execute_fn(
            {
                "query": "도로 민원",
                "results": mock_search_results,
                "context_text": "### 로컬 문서 검색 결과:",
            }
        )
        cap = RagSearchCapability(execute_fn=fn)
        result = await cap.execute("도로 민원", {}, None)

        assert result.context_text == "### 로컬 문서 검색 결과:"


# ===========================================================================
# TestRagSearchCapabilityCall — __call__ 통합 검증
# ===========================================================================
@requires_capabilities
class TestRagSearchCapabilityCall:
    """RagSearchCapability.__call__() 검증."""

    @pytest.mark.asyncio
    async def test_call_returns_dict_with_latency(self, mock_search_results):
        """__call__()이 latency_ms가 포함된 dict를 반환한다."""
        fn = _make_execute_fn(
            {
                "query": "테스트",
                "results": mock_search_results,
                "context_text": "",
            }
        )
        cap = RagSearchCapability(execute_fn=fn)
        result = await cap(query="테스트", context={}, session=None)

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0
