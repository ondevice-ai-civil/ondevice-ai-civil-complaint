"""draft_civil_response / append_evidence capability 단위·통합 테스트.

Issue #397: draft_civil_response / append_evidence capability 구현.

검증 항목:
- DraftCivilResponseCapability: 정상/에러 응답 LookupResult 변환
- DraftCivilResponseCapability: timeout_sec=30.0, metadata.provider="local_llm"
- AppendEvidenceCapability: 정상/에러/혼합/빈 결과 처리
- AppendEvidenceCapability: timeout_sec=15.0, metadata.provider="local_vectordb+data.go.kr"
- registry에 두 capability가 모두 등록되는지 검증
- executor adapter를 통해 두 capability가 각각 올바르게 실행되는지 검증

외부 서비스(vLLM, API)는 모두 mock 처리한다.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# langgraph가 설치되지 않은 CI 환경 대응:
# capabilities 모듈은 langgraph에 의존하지 않으므로 직접 import 가능.
_LANGGRAPH_AVAILABLE = True
try:
    import langgraph  # noqa: F401
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    for mod_name in [
        "langgraph",
        "langgraph.graph",
        "langgraph.graph.message",
        "langgraph.graph.state",
        "langgraph.checkpoint",
        "langgraph.checkpoint.memory",
        "langgraph.types",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    _lg_graph = sys.modules["langgraph.graph"]
    _lg_graph.END = "END"  # type: ignore[attr-defined]
    _lg_graph.START = "START"  # type: ignore[attr-defined]
    _lg_graph.StateGraph = MagicMock()  # type: ignore[attr-defined]

    _lg_msg = sys.modules["langgraph.graph.message"]
    _lg_msg.add_messages = lambda x: x  # type: ignore[attr-defined]

    _lg_mem = sys.modules["langgraph.checkpoint.memory"]
    _lg_mem.MemorySaver = MagicMock()  # type: ignore[attr-defined]

    _lg_types = sys.modules["langgraph.types"]
    _lg_types.interrupt = MagicMock()  # type: ignore[attr-defined]

# langchain_core도 mock 처리 (설치 안 된 경우)
try:
    import langchain_core  # noqa: F401
except ImportError:
    for mod_name in [
        "langchain_core",
        "langchain_core.messages",
    ]:
        if mod_name not in sys.modules:
            _mock = types.ModuleType(mod_name)
            sys.modules[mod_name] = _mock

    _lc_messages = sys.modules["langchain_core.messages"]
    _lc_messages.AnyMessage = MagicMock()  # type: ignore[attr-defined]
    _lc_messages.AIMessage = MagicMock()  # type: ignore[attr-defined]
    _lc_messages.HumanMessage = MagicMock()  # type: ignore[attr-defined]
    _lc_messages.SystemMessage = MagicMock()  # type: ignore[attr-defined]


from src.inference.graph.capabilities.append_evidence import AppendEvidenceCapability
from src.inference.graph.capabilities.base import CapabilityBase, LookupResult
from src.inference.graph.capabilities.draft_civil_response import DraftCivilResponseCapability
from src.inference.graph.capabilities.registry import (
    MVP_CAPABILITY_IDS,
    build_mvp_registry,
)

# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------


def _make_async_fn(return_value: dict) -> AsyncMock:
    """async (query, context, session) -> dict 시그니처의 mock."""
    fn = AsyncMock(return_value=return_value)
    return fn


@pytest.fixture
def session():
    """테스트용 mock 세션."""
    return MagicMock()


@pytest.fixture
def mock_registry_fns():
    """build_mvp_registry에 주입할 mock 함수 모음."""
    return {
        "rag_search_fn": _make_async_fn({"text": "rag_result", "results": []}),
        "draft_civil_response_fn": _make_async_fn({"text": "민원 답변 초안입니다.", "results": []}),
        "append_evidence_fn": _make_async_fn(
            {
                "text": "근거 보강 결과입니다.",
                "api_citations": [{"source": "data.go.kr"}],
                "rag_results": [{"title": "관련 법령"}],
            }
        ),
    }


@pytest.fixture
def registry(mock_registry_fns):
    """MVP registry fixture."""
    return build_mvp_registry(
        rag_search_fn=mock_registry_fns["rag_search_fn"],
        api_lookup_action=None,
        draft_civil_response_fn=mock_registry_fns["draft_civil_response_fn"],
        append_evidence_fn=mock_registry_fns["append_evidence_fn"],
    )


# ===========================================================================
# TestDraftCivilResponseCapability — 단위 테스트
# ===========================================================================


class TestDraftCivilResponseCapability:
    """DraftCivilResponseCapability 단위 테스트."""

    # -----------------------------------------------------------------------
    # metadata 검증
    # -----------------------------------------------------------------------

    def test_metadata_provider_is_local_llm(self):
        """metadata.provider가 'local_llm'이다."""
        cap = DraftCivilResponseCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.provider == "local_llm"

    def test_metadata_timeout_sec_is_30(self):
        """timeout_sec이 30.0이다."""
        cap = DraftCivilResponseCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.timeout_sec == 30.0

    def test_metadata_name(self):
        """metadata.name이 'draft_civil_response'이다."""
        cap = DraftCivilResponseCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.name == "draft_civil_response"

    def test_metadata_has_description(self):
        """metadata.description이 비어있지 않다."""
        cap = DraftCivilResponseCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.description

    def test_metadata_has_approval_summary(self):
        """metadata.approval_summary가 비어있지 않다."""
        cap = DraftCivilResponseCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.approval_summary

    # -----------------------------------------------------------------------
    # 정상 응답 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_success_sets_context_text(self, session):
        """정상 응답 시 context_text가 raw['text']로 채워진다."""
        fn = _make_async_fn({"text": "민원 답변 초안입니다.", "results": []})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("민원 처리 방법", {}, session)

        assert result.success is True
        assert result.context_text == "민원 답변 초안입니다."

    @pytest.mark.asyncio
    async def test_success_sets_results(self, session):
        """정상 응답 시 results에 raw dict가 포함된다."""
        raw = {"text": "초안", "extra_field": "값"}
        fn = _make_async_fn(raw)
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.success is True
        assert raw in result.results

    @pytest.mark.asyncio
    async def test_success_provider_matches_metadata(self, session):
        """정상 응답 시 result.provider가 metadata.provider와 동일하다."""
        fn = _make_async_fn({"text": "초안"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.provider == "local_llm"

    @pytest.mark.asyncio
    async def test_success_query_is_preserved(self, session):
        """정상 응답 시 result.query가 입력 query와 동일하다."""
        fn = _make_async_fn({"text": "초안"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("입력된 민원", {}, session)

        assert result.query == "입력된 민원"

    @pytest.mark.asyncio
    async def test_success_no_error(self, session):
        """정상 응답 시 error가 None이다."""
        fn = _make_async_fn({"text": "초안"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.error is None

    # -----------------------------------------------------------------------
    # 에러 응답 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_response_sets_success_false(self, session):
        """에러 응답 시 success=False이다."""
        fn = _make_async_fn({"error": "vLLM 연결 실패"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_response_maps_error_field(self, session):
        """에러 응답 시 error 필드가 raw['error']로 매핑된다."""
        fn = _make_async_fn({"error": "vLLM 연결 실패"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.error == "vLLM 연결 실패"

    @pytest.mark.asyncio
    async def test_error_response_sets_empty_reason(self, session):
        """에러 응답 시 empty_reason이 'provider_error'이다."""
        fn = _make_async_fn({"error": "타임아웃"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.empty_reason == "provider_error"

    @pytest.mark.asyncio
    async def test_error_response_provider_is_set(self, session):
        """에러 응답 시에도 provider가 올바르게 설정된다."""
        fn = _make_async_fn({"error": "오류"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.provider == "local_llm"

    # -----------------------------------------------------------------------
    # __call__ → execute() → LookupResult → to_dict() 흐름 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_call_returns_dict(self, session):
        """__call__이 dict를 반환한다 (RegistryExecutorAdapter 호환)."""
        fn = _make_async_fn({"text": "초안"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap(query="테스트", context={}, session=session)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_call_includes_success_field(self, session):
        """__call__ 결과 dict에 'success' 필드가 포함된다."""
        fn = _make_async_fn({"text": "초안"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap(query="테스트", context={}, session=session)

        assert "success" in result

    @pytest.mark.asyncio
    async def test_call_includes_latency_ms(self, session):
        """__call__ 결과 dict에 'latency_ms' 필드가 포함된다."""
        fn = _make_async_fn({"text": "초안"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap(query="테스트", context={}, session=session)

        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_call_includes_context_text(self, session):
        """__call__ 결과 dict에 'context_text' 필드가 포함된다."""
        fn = _make_async_fn({"text": "초안 본문"})
        cap = DraftCivilResponseCapability(execute_fn=fn)

        result = await cap(query="테스트", context={}, session=session)

        assert result["context_text"] == "초안 본문"

    @pytest.mark.asyncio
    async def test_execute_fn_called_with_correct_args(self, session):
        """execute_fn이 올바른 인자로 호출된다."""
        fn = _make_async_fn({"text": "초안"})
        cap = DraftCivilResponseCapability(execute_fn=fn)
        ctx = {"session_id": "test-session"}

        await cap.execute("민원 처리", ctx, session)

        fn.assert_awaited_once_with(query="민원 처리", context=ctx, session=session)


# ===========================================================================
# TestAppendEvidenceCapability — 단위 테스트
# ===========================================================================


class TestAppendEvidenceCapability:
    """AppendEvidenceCapability 단위 테스트."""

    # -----------------------------------------------------------------------
    # metadata 검증
    # -----------------------------------------------------------------------

    def test_metadata_provider_is_local_vectordb_and_data_go_kr(self):
        """metadata.provider가 'local_vectordb+data.go.kr'이다."""
        cap = AppendEvidenceCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.provider == "local_vectordb+data.go.kr"

    def test_metadata_timeout_sec_is_15(self):
        """timeout_sec이 15.0이다."""
        cap = AppendEvidenceCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.timeout_sec == 15.0

    def test_metadata_name(self):
        """metadata.name이 'append_evidence'이다."""
        cap = AppendEvidenceCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.name == "append_evidence"

    def test_metadata_has_description(self):
        """metadata.description이 비어있지 않다."""
        cap = AppendEvidenceCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.description

    def test_metadata_has_approval_summary(self):
        """metadata.approval_summary가 비어있지 않다."""
        cap = AppendEvidenceCapability(execute_fn=_make_async_fn({}))
        assert cap.metadata.approval_summary

    # -----------------------------------------------------------------------
    # 정상 응답 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_success_sets_context_text(self, session):
        """정상 응답 시 context_text가 raw['text']로 채워진다."""
        raw = {"text": "근거 보강 결과입니다.", "api_citations": [], "rag_results": []}
        fn = _make_async_fn(raw)
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("근거 추가", {}, session)

        assert result.success is True
        assert result.context_text == "근거 보강 결과입니다."

    @pytest.mark.asyncio
    async def test_success_maps_api_citations(self, session):
        """정상 응답 시 citations 필드가 raw['api_citations']로 매핑된다."""
        raw = {
            "text": "결과",
            "api_citations": [{"source": "data.go.kr", "url": "http://example.com"}],
            "rag_results": [],
        }
        fn = _make_async_fn(raw)
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.citations == raw["api_citations"]

    @pytest.mark.asyncio
    async def test_success_maps_rag_results(self, session):
        """정상 응답 시 results 필드가 raw['rag_results']로 매핑된다."""
        raw = {
            "text": "결과",
            "api_citations": [],
            "rag_results": [{"title": "관련 법령", "content": "법령 내용"}],
        }
        fn = _make_async_fn(raw)
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.results == raw["rag_results"]

    @pytest.mark.asyncio
    async def test_success_provider_matches_metadata(self, session):
        """정상 응답 시 result.provider가 metadata.provider와 동일하다."""
        fn = _make_async_fn({"text": "결과", "api_citations": [], "rag_results": []})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.provider == "local_vectordb+data.go.kr"

    # -----------------------------------------------------------------------
    # rag+api 혼합 결과 처리 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_mixed_rag_and_api_citations(self, session):
        """rag+api 혼합 결과가 각 필드에 올바르게 분리된다."""
        raw = {
            "text": "혼합 결과",
            "api_citations": [
                {"source": "data.go.kr", "title": "공공 데이터 API"},
                {"source": "data.go.kr", "title": "통계 API"},
            ],
            "rag_results": [
                {"title": "법령 1", "content": "법령 내용"},
                {"title": "법령 2", "content": "추가 내용"},
            ],
        }
        fn = _make_async_fn(raw)
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("혼합 테스트", {}, session)

        assert result.success is True
        assert len(result.citations) == 2
        assert len(result.results) == 2
        assert result.citations[0]["title"] == "공공 데이터 API"
        assert result.results[0]["title"] == "법령 1"

    @pytest.mark.asyncio
    async def test_mixed_result_context_text_preserved(self, session):
        """혼합 결과에서 context_text가 올바르게 보존된다."""
        raw = {
            "text": "혼합 컨텍스트 텍스트",
            "api_citations": [{"source": "api"}],
            "rag_results": [{"title": "법령"}],
        }
        fn = _make_async_fn(raw)
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.context_text == "혼합 컨텍스트 텍스트"

    # -----------------------------------------------------------------------
    # 빈 결과 처리 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_results_still_success(self, session):
        """api_citations와 rag_results 모두 비어있어도 success=True이다."""
        fn = _make_async_fn({"text": "", "api_citations": [], "rag_results": []})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_results_citations_is_empty_list(self, session):
        """빈 결과 시 citations가 빈 리스트이다."""
        fn = _make_async_fn({"text": "", "api_citations": [], "rag_results": []})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.citations == []

    @pytest.mark.asyncio
    async def test_empty_results_rag_results_is_empty_list(self, session):
        """빈 결과 시 results가 빈 리스트이다."""
        fn = _make_async_fn({"text": "", "api_citations": [], "rag_results": []})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.results == []

    @pytest.mark.asyncio
    async def test_missing_api_citations_key_defaults_to_empty(self, session):
        """raw에 'api_citations' 키가 없으면 citations가 빈 리스트이다."""
        fn = _make_async_fn({"text": "결과"})  # api_citations 키 없음
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.citations == []

    @pytest.mark.asyncio
    async def test_missing_rag_results_key_defaults_to_empty(self, session):
        """raw에 'rag_results' 키가 없으면 results가 빈 리스트이다."""
        fn = _make_async_fn({"text": "결과"})  # rag_results 키 없음
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.results == []

    # -----------------------------------------------------------------------
    # 에러 응답 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_response_sets_success_false(self, session):
        """에러 응답 시 success=False이다."""
        fn = _make_async_fn({"error": "벡터DB 연결 실패"})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_response_maps_error_field(self, session):
        """에러 응답 시 error 필드가 raw['error']로 매핑된다."""
        fn = _make_async_fn({"error": "벡터DB 연결 실패"})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.error == "벡터DB 연결 실패"

    @pytest.mark.asyncio
    async def test_error_response_sets_empty_reason(self, session):
        """에러 응답 시 empty_reason이 'provider_error'이다."""
        fn = _make_async_fn({"error": "타임아웃"})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap.execute("테스트", {}, session)

        assert result.empty_reason == "provider_error"

    # -----------------------------------------------------------------------
    # __call__ → execute() → LookupResult → to_dict() 흐름 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_call_returns_dict(self, session):
        """__call__이 dict를 반환한다 (RegistryExecutorAdapter 호환)."""
        fn = _make_async_fn({"text": "결과", "api_citations": [], "rag_results": []})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap(query="테스트", context={}, session=session)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_call_includes_latency_ms(self, session):
        """__call__ 결과 dict에 'latency_ms' 필드가 포함된다."""
        fn = _make_async_fn({"text": "결과", "api_citations": [], "rag_results": []})
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap(query="테스트", context={}, session=session)

        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_call_to_dict_includes_citations(self, session):
        """__call__ 결과 dict에 'citations' 필드가 포함된다."""
        raw = {
            "text": "결과",
            "api_citations": [{"source": "data.go.kr"}],
            "rag_results": [],
        }
        fn = _make_async_fn(raw)
        cap = AppendEvidenceCapability(execute_fn=fn)

        result = await cap(query="테스트", context={}, session=session)

        assert "citations" in result
        assert len(result["citations"]) == 1

    @pytest.mark.asyncio
    async def test_execute_fn_called_with_correct_args(self, session):
        """execute_fn이 올바른 인자로 호출된다."""
        fn = _make_async_fn({"text": "결과", "api_citations": [], "rag_results": []})
        cap = AppendEvidenceCapability(execute_fn=fn)
        ctx = {"session_id": "test-session"}

        await cap.execute("근거 추가", ctx, session)

        fn.assert_awaited_once_with(query="근거 추가", context=ctx, session=session)


# ===========================================================================
# TestRegistryIntegration — 경로 분리 통합 테스트
# ===========================================================================


class TestRegistryIntegration:
    """registry에 두 capability가 올바르게 등록되고 실행되는지 검증."""

    def test_draft_civil_response_in_registry(self, registry):
        """draft_civil_response가 registry에 등록되어 있다."""
        assert "draft_civil_response" in registry

    def test_append_evidence_in_registry(self, registry):
        """append_evidence가 registry에 등록되어 있다."""
        assert "append_evidence" in registry

    def test_draft_civil_response_is_capability_base(self, registry):
        """registry의 draft_civil_response가 CapabilityBase 인스턴스이다."""
        assert isinstance(registry["draft_civil_response"], CapabilityBase)

    def test_append_evidence_is_capability_base(self, registry):
        """registry의 append_evidence가 CapabilityBase 인스턴스이다."""
        assert isinstance(registry["append_evidence"], CapabilityBase)

    def test_draft_civil_response_metadata_name_matches_key(self, registry):
        """draft_civil_response의 metadata.name이 registry key와 동일하다."""
        cap = registry["draft_civil_response"]
        assert cap.metadata.name == "draft_civil_response"

    def test_append_evidence_metadata_name_matches_key(self, registry):
        """append_evidence의 metadata.name이 registry key와 동일하다."""
        cap = registry["append_evidence"]
        assert cap.metadata.name == "append_evidence"

    def test_all_mvp_capabilities_registered(self, registry):
        """4개 MVP capability가 모두 등록되어 있다."""
        assert set(registry.keys()) == MVP_CAPABILITY_IDS

    def test_draft_civil_response_provider_in_registry(self, registry):
        """registry에서 가져온 draft_civil_response의 provider가 'local_llm'이다."""
        cap = registry["draft_civil_response"]
        assert cap.metadata.provider == "local_llm"

    def test_append_evidence_provider_in_registry(self, registry):
        """registry에서 가져온 append_evidence의 provider가 'local_vectordb+data.go.kr'이다."""
        cap = registry["append_evidence"]
        assert cap.metadata.provider == "local_vectordb+data.go.kr"

    def test_draft_civil_response_timeout_in_registry(self, registry):
        """registry에서 가져온 draft_civil_response의 timeout_sec이 30.0이다."""
        cap = registry["draft_civil_response"]
        assert cap.metadata.timeout_sec == 30.0

    def test_append_evidence_timeout_in_registry(self, registry):
        """registry에서 가져온 append_evidence의 timeout_sec이 15.0이다."""
        cap = registry["append_evidence"]
        assert cap.metadata.timeout_sec == 15.0


# ===========================================================================
# TestExecutorAdapterIntegration — executor adapter를 통한 실행 검증
# ===========================================================================


class TestExecutorAdapterIntegration:
    """RegistryExecutorAdapter를 통해 두 capability가 올바르게 실행되는지 검증."""

    def _make_adapter(self, registry):
        from src.inference.graph.executor_adapter import RegistryExecutorAdapter

        session_store = MagicMock()
        session_store.get_or_create.return_value = MagicMock()
        return RegistryExecutorAdapter(
            tool_registry=registry,
            session_store=session_store,
        )

    # -----------------------------------------------------------------------
    # list_tools 검증
    # -----------------------------------------------------------------------

    def test_list_tools_includes_draft_civil_response(self, registry):
        """list_tools()에 draft_civil_response가 포함된다."""
        adapter = self._make_adapter(registry)
        assert "draft_civil_response" in adapter.list_tools()

    def test_list_tools_includes_append_evidence(self, registry):
        """list_tools()에 append_evidence가 포함된다."""
        adapter = self._make_adapter(registry)
        assert "append_evidence" in adapter.list_tools()

    # -----------------------------------------------------------------------
    # get_tool_metadata 검증
    # -----------------------------------------------------------------------

    def test_get_tool_metadata_draft_civil_response(self, registry):
        """adapter.get_tool_metadata()가 draft_civil_response의 올바른 metadata를 반환한다."""
        adapter = self._make_adapter(registry)
        meta = adapter.get_tool_metadata("draft_civil_response")

        assert meta is not None
        assert meta["name"] == "draft_civil_response"
        assert meta["provider"] == "local_llm"
        assert meta["description"]
        assert meta["approval_summary"]

    def test_get_tool_metadata_append_evidence(self, registry):
        """adapter.get_tool_metadata()가 append_evidence의 올바른 metadata를 반환한다."""
        adapter = self._make_adapter(registry)
        meta = adapter.get_tool_metadata("append_evidence")

        assert meta is not None
        assert meta["name"] == "append_evidence"
        assert meta["provider"] == "local_vectordb+data.go.kr"
        assert meta["description"]
        assert meta["approval_summary"]

    # -----------------------------------------------------------------------
    # execute 성공 경로 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_draft_civil_response_success(self, registry):
        """adapter를 통해 draft_civil_response 실행 시 success=True를 반환한다."""
        adapter = self._make_adapter(registry)
        result = await adapter.execute(
            "draft_civil_response",
            "민원 답변 초안 작성",
            {"session_id": "test"},
        )

        assert result["success"] is True
        assert "context_text" in result

    @pytest.mark.asyncio
    async def test_execute_append_evidence_success(self, registry):
        """adapter를 통해 append_evidence 실행 시 success=True를 반환한다."""
        adapter = self._make_adapter(registry)
        result = await adapter.execute(
            "append_evidence",
            "근거 추가",
            {"session_id": "test"},
        )

        assert result["success"] is True
        assert "citations" in result

    @pytest.mark.asyncio
    async def test_execute_draft_civil_response_has_latency(self, registry):
        """adapter를 통한 draft_civil_response 실행 결과에 latency_ms가 있다."""
        adapter = self._make_adapter(registry)
        result = await adapter.execute(
            "draft_civil_response",
            "테스트",
            {"session_id": "test"},
        )

        # adapter 자체도 latency_ms를 주입한다
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_execute_append_evidence_has_latency(self, registry):
        """adapter를 통한 append_evidence 실행 결과에 latency_ms가 있다."""
        adapter = self._make_adapter(registry)
        result = await adapter.execute(
            "append_evidence",
            "테스트",
            {"session_id": "test"},
        )

        assert "latency_ms" in result

    # -----------------------------------------------------------------------
    # 경로 분리 검증: 두 capability가 서로 독립적으로 실행된다
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_draft_and_evidence_execute_independently(self, mock_registry_fns):
        """draft_civil_response와 append_evidence 실행이 서로 독립적이다."""
        registry = build_mvp_registry(
            rag_search_fn=mock_registry_fns["rag_search_fn"],
            api_lookup_action=None,
            draft_civil_response_fn=mock_registry_fns["draft_civil_response_fn"],
            append_evidence_fn=mock_registry_fns["append_evidence_fn"],
        )
        adapter = self._make_adapter(registry)

        # draft_civil_response 실행
        draft_result = await adapter.execute(
            "draft_civil_response",
            "민원 답변",
            {"session_id": "s1"},
        )

        # append_evidence 실행
        evidence_result = await adapter.execute(
            "append_evidence",
            "근거 추가",
            {"session_id": "s1"},
        )

        # 각 mock fn이 정확히 1번씩 호출되었는지 확인
        mock_registry_fns["draft_civil_response_fn"].assert_awaited_once()
        mock_registry_fns["append_evidence_fn"].assert_awaited_once()

        # 각 결과가 독립적으로 올바른 값을 가지는지 확인
        assert draft_result["success"] is True
        assert evidence_result["success"] is True

    @pytest.mark.asyncio
    async def test_draft_result_contains_text(self, registry):
        """adapter를 통한 draft_civil_response 결과에 'context_text'가 포함된다."""
        adapter = self._make_adapter(registry)
        result = await adapter.execute(
            "draft_civil_response",
            "테스트 민원",
            {"session_id": "test"},
        )

        assert result.get("context_text") == "민원 답변 초안입니다."

    @pytest.mark.asyncio
    async def test_evidence_result_contains_citations(self, registry):
        """adapter를 통한 append_evidence 결과에 'citations'가 포함된다."""
        adapter = self._make_adapter(registry)
        result = await adapter.execute(
            "append_evidence",
            "테스트 민원",
            {"session_id": "test"},
        )

        assert result.get("citations") == [{"source": "data.go.kr"}]

    # -----------------------------------------------------------------------
    # 에러 경로 검증
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_draft_civil_response_error_path(self):
        """draft_civil_response가 에러를 반환할 때 adapter가 success=False를 반환한다."""
        error_fns = {
            "rag_search_fn": _make_async_fn({"results": []}),
            "draft_civil_response_fn": _make_async_fn({"error": "LLM 오류 발생"}),
            "append_evidence_fn": _make_async_fn(
                {"text": "", "api_citations": [], "rag_results": []}
            ),
        }
        registry = build_mvp_registry(
            rag_search_fn=error_fns["rag_search_fn"],
            api_lookup_action=None,
            draft_civil_response_fn=error_fns["draft_civil_response_fn"],
            append_evidence_fn=error_fns["append_evidence_fn"],
        )

        session_store = MagicMock()
        session_store.get_or_create.return_value = MagicMock()

        from src.inference.graph.executor_adapter import RegistryExecutorAdapter

        adapter = RegistryExecutorAdapter(
            tool_registry=registry,
            session_store=session_store,
        )

        result = await adapter.execute(
            "draft_civil_response",
            "테스트",
            {"session_id": "test"},
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_append_evidence_error_path(self):
        """append_evidence가 에러를 반환할 때 adapter가 success=False를 반환한다."""
        error_fns = {
            "rag_search_fn": _make_async_fn({"results": []}),
            "draft_civil_response_fn": _make_async_fn({"text": "초안"}),
            "append_evidence_fn": _make_async_fn({"error": "벡터DB 장애"}),
        }
        registry = build_mvp_registry(
            rag_search_fn=error_fns["rag_search_fn"],
            api_lookup_action=None,
            draft_civil_response_fn=error_fns["draft_civil_response_fn"],
            append_evidence_fn=error_fns["append_evidence_fn"],
        )

        session_store = MagicMock()
        session_store.get_or_create.return_value = MagicMock()

        from src.inference.graph.executor_adapter import RegistryExecutorAdapter

        adapter = RegistryExecutorAdapter(
            tool_registry=registry,
            session_store=session_store,
        )

        result = await adapter.execute(
            "append_evidence",
            "테스트",
            {"session_id": "test"},
        )

        assert result["success"] is False
        assert "error" in result
