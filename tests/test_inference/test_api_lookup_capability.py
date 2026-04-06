"""api_lookup capability 통합 테스트.

Part A(capabilities 패키지)와 Part B(executor_adapter metadata 지원)를 검증한다.
Part A가 미완료 상태에서도 import 오류 없이 skip되도록 guard를 둔다.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

try:
    from src.inference.graph.capabilities.base import (
        CapabilityBase,
        CapabilityMetadata,
        LookupResult,
    )

    CAPABILITIES_BASE_AVAILABLE = True
except ImportError:
    CAPABILITIES_BASE_AVAILABLE = False

try:
    from src.inference.graph.capabilities import (
        ApiLookupCapability,
        ApiLookupParams,
    )

    CAPABILITIES_AVAILABLE = True
except ImportError:
    CAPABILITIES_AVAILABLE = False

# ---------------------------------------------------------------------------
# Part A 전체(ApiLookupCapability) 필요 테스트 → skip guard
# ---------------------------------------------------------------------------
requires_capabilities = pytest.mark.skipif(
    not CAPABILITIES_AVAILABLE,
    reason="capabilities 패키지 미구현 (Part A 대기)",
)


# ===========================================================================
# TestApiLookupParams — 파라미터 validator
# ===========================================================================
@requires_capabilities
class TestApiLookupParams:
    """ApiLookupParams validator 검증."""

    def test_valid_params_pass(self):
        """정상 query는 검증을 통과한다."""
        params = ApiLookupParams(query="민원 처리 절차")
        error = params.validate()
        assert error is None

    def test_empty_query_fails(self):
        """빈 query는 validate() 오류 문자열을 반환한다."""
        params = ApiLookupParams(query="")
        error = params.validate()
        assert error is not None

    def test_long_query_fails(self):
        """501자 이상 query는 validate() 오류 문자열을 반환한다."""
        params = ApiLookupParams(query="가" * 501)
        error = params.validate()
        assert error is not None

    def test_alias_normalization(self):
        """context alias(count, score_threshold)가 ret_count, min_score로 정규화된다."""
        params = ApiLookupParams.from_context(
            query="테스트",
            context={"count": 3, "score_threshold": 1.0},
        )
        assert params.ret_count == 3
        assert params.min_score == 1

    def test_ret_count_clamped(self):
        """ret_count가 상한(20)으로 클램핑된다."""
        params = ApiLookupParams.from_context(query="테스트", context={"count": 999})
        assert params.ret_count <= 20

    def test_min_score_clamped(self):
        """min_score가 하한(0)으로 클램핑된다."""
        params = ApiLookupParams.from_context(query="테스트", context={"min_score": -5})
        assert params.min_score >= 0


# ===========================================================================
# TestApiLookupCapabilityMetadata
# ===========================================================================
@requires_capabilities
class TestApiLookupCapabilityMetadata:
    """ApiLookupCapability metadata 검증."""

    @pytest.fixture
    def capability(self):
        return ApiLookupCapability(action=None)

    def test_metadata_name(self, capability):
        assert capability.metadata.name == "api_lookup"

    def test_metadata_has_description(self, capability):
        assert capability.metadata.description

    def test_metadata_has_approval_summary(self, capability):
        assert capability.metadata.approval_summary


# ===========================================================================
# TestApiLookupCapabilityNoAction — action=None 시 동작
# ===========================================================================
@requires_capabilities
class TestApiLookupCapabilityNoAction:
    """action이 None일 때의 동작을 검증한다."""

    @pytest.fixture
    def capability_no_action(self):
        return ApiLookupCapability(action=None)

    @pytest.mark.asyncio
    async def test_empty_result_when_no_action(self, capability_no_action):
        """action이 None이면 빈 결과를 반환한다."""
        result = await capability_no_action(
            query="테스트 민원",
            context={},
            session=None,
        )
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_validation_error_returned(self, capability_no_action):
        """빈 query면 success=False를 반환한다."""
        result = await capability_no_action(query="", context={}, session=None)
        assert result["success"] is False
        assert result["error"]


# ===========================================================================
# TestRegistryExecutorAdapterMetadata — get_tool_metadata 검증
# ===========================================================================
class TestRegistryExecutorAdapterMetadata:
    """RegistryExecutorAdapter.get_tool_metadata() 검증."""

    def _make_adapter(self, tool_registry=None):
        from src.inference.graph.executor_adapter import RegistryExecutorAdapter

        return RegistryExecutorAdapter(
            tool_registry=tool_registry or {},
            session_store=MagicMock(),
        )

    def test_get_tool_metadata_returns_none_for_unknown(self):
        """등록되지 않은 tool은 None을 반환한다."""
        adapter = self._make_adapter()
        assert adapter.get_tool_metadata("unknown") is None

    def test_get_tool_metadata_plain_callable(self):
        """일반 callable 등록 시 이름만 포함된 기본 metadata를 반환한다."""

        async def dummy(query, context, session):
            return {"success": True}

        adapter = self._make_adapter({"dummy_tool": dummy})
        meta = adapter.get_tool_metadata("dummy_tool")
        assert meta is not None
        assert meta["name"] == "dummy_tool"
        assert meta["description"] == ""

    @pytest.mark.skipif(
        not CAPABILITIES_BASE_AVAILABLE,
        reason="capabilities.base 미사용 가능",
    )
    def test_get_tool_metadata_with_capability(self):
        """CapabilityBase 인스턴스 등록 시 metadata를 올바르게 반환한다."""

        class _StubCapability(CapabilityBase):
            @property
            def metadata(self) -> CapabilityMetadata:
                return CapabilityMetadata(
                    name="api_lookup",
                    description="공공 API 조회",
                    approval_summary="공공 데이터 조회",
                    provider="data.go.kr",
                )

            async def execute(self, query, context, session):
                return LookupResult(success=True, query=query)

        cap = _StubCapability()
        adapter = self._make_adapter({"api_lookup": cap})
        meta = adapter.get_tool_metadata("api_lookup")
        assert meta is not None
        assert meta["name"] == "api_lookup"
        assert meta["description"] == "공공 API 조회"
        assert meta["approval_summary"] == "공공 데이터 조회"
        assert meta["provider"] == "data.go.kr"

    @requires_capabilities
    def test_get_tool_metadata_with_api_lookup_capability(self):
        """실제 ApiLookupCapability 등록 시 metadata를 올바르게 반환한다."""
        cap = ApiLookupCapability(action=None)
        adapter = self._make_adapter({"api_lookup": cap})
        meta = adapter.get_tool_metadata("api_lookup")
        assert meta is not None
        assert meta["name"] == "api_lookup"
        assert meta["description"]

    def test_list_tools_returns_registered_names(self):
        """list_tools()가 등록된 tool 이름을 반환한다."""

        async def a(query, context, session):
            return {}

        async def b(query, context, session):
            return {}

        adapter = self._make_adapter({"tool_a": a, "tool_b": b})
        names = adapter.list_tools()
        assert set(names) == {"tool_a", "tool_b"}


# ===========================================================================
# TestLookupResult — LookupResult.to_dict() 검증
# ===========================================================================
@pytest.mark.skipif(
    not CAPABILITIES_BASE_AVAILABLE,
    reason="capabilities.base 미구현",
)
class TestLookupResult:
    """LookupResult.to_dict() 검증."""

    def test_to_dict_success(self):
        """성공 결과가 올바른 dict로 변환된다."""
        result = LookupResult(
            success=True,
            query="테스트",
            results=[{"title": "사례1"}],
            context_text="컨텍스트",
            provider="data.go.kr",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["query"] == "테스트"
        assert d["count"] == 1
        assert d["provider"] == "data.go.kr"
        assert d["error"] is None

    def test_to_dict_failure(self):
        """실패 결과의 error 필드가 포함된다."""
        result = LookupResult(
            success=False,
            query="테스트",
            error="API 오류",
            empty_reason="provider_error",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "API 오류"
        assert d["count"] == 0

    def test_to_dict_empty_reason(self):
        """empty_reason이 dict에 포함된다."""
        result = LookupResult(success=True, query="테스트", empty_reason="no_match")
        d = result.to_dict()
        assert d["empty_reason"] == "no_match"


# ===========================================================================
# TestCapabilityBaseCall — CapabilityBase.__call__() latency 측정 검증
# ===========================================================================
@pytest.mark.skipif(
    not CAPABILITIES_BASE_AVAILABLE,
    reason="capabilities.base 미구현",
)
class TestCapabilityBaseCall:
    """CapabilityBase.__call__() latency 측정 및 to_dict() 변환 검증."""

    @pytest.mark.asyncio
    async def test_call_returns_dict_with_latency(self):
        """__call__()이 latency_ms가 포함된 dict를 반환한다."""

        class _Stub(CapabilityBase):
            @property
            def metadata(self) -> CapabilityMetadata:
                return CapabilityMetadata(
                    name="stub",
                    description="stub",
                    approval_summary="stub",
                    provider="stub",
                )

            async def execute(self, query, context, session):
                return LookupResult(success=True, query=query)

        stub = _Stub()
        result = await stub(query="테스트", context={}, session=None)
        assert result["success"] is True
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0


# ===========================================================================
# TestApiLookupCapabilityWithAction — action 있을 때의 동작 검증
# ===========================================================================
@requires_capabilities
class TestApiLookupCapabilityWithAction:
    """action이 주어졌을 때 ApiLookupCapability 동작 검증."""

    @pytest.mark.asyncio
    async def test_validation_error_when_empty_query(self):
        """action이 있어도 빈 query는 validation error를 반환한다."""
        action = MagicMock()
        cap = ApiLookupCapability(action=action)
        result = await cap(query="", context={}, session=None)
        assert result["success"] is False
        assert result["error"]

    @pytest.mark.asyncio
    async def test_result_when_action_returns_results(self):
        """action이 결과를 반환하면 LookupResult로 변환된다."""

        async def _fetch(q, ctx, **kwargs):
            return {
                "results": [{"title": "사례1"}],
                "context_text": "컨텍스트",
                "citations": [],
                "query": q,
            }

        action = MagicMock()
        action.fetch_similar_cases = _fetch
        cap = ApiLookupCapability(action=action)
        result = await cap(query="민원 처리", context={}, session=None)
        assert result["success"] is True
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_empty_results_returns_no_match(self):
        """action이 빈 results를 반환하면 no_match로 처리한다."""

        async def _fetch(q, ctx, **kwargs):
            return {"results": [], "citations": [], "query": q}

        action = MagicMock()
        action.fetch_similar_cases = _fetch
        cap = ApiLookupCapability(action=action)
        result = await cap(query="민원 처리", context={}, session=None)
        assert result["success"] is True
        assert result["count"] == 0
        assert result["empty_reason"] == "no_match"

    @pytest.mark.asyncio
    async def test_action_exception_returns_failure(self):
        """action에서 예외 발생 시 success=False를 반환한다."""

        async def _fetch(q, ctx, **kwargs):
            raise RuntimeError("연결 오류")

        action = MagicMock()
        action.fetch_similar_cases = _fetch
        cap = ApiLookupCapability(action=action)
        result = await cap(query="민원 처리", context={}, session=None)
        assert result["success"] is False
        assert "연결 오류" in result["error"]

    @pytest.mark.asyncio
    async def test_action_none_payload_returns_failure(self):
        """action이 results=None인 payload를 반환하면 failure."""

        async def _fetch(q, ctx, **kwargs):
            return {"results": None}

        action = MagicMock()
        action.fetch_similar_cases = _fetch
        cap = ApiLookupCapability(action=action)
        result = await cap(query="민원 처리", context={}, session=None)
        assert result["success"] is False
