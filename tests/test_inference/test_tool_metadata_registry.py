"""tool metadata registry 단위 테스트.

Issue #416: tool metadata registry 및 LangGraph executor binding 정리.

검증 항목:
- MVP capability stable identifier 일관성
- build_mvp_registry()가 모든 capability를 CapabilityBase로 반환
- planner metadata와 executor binding이 같은 소스에서 나옴
- approval summary와 session log가 동일한 identifier 사용
- 비MVP capability가 registry 수준에서 차단됨
- get_tool_descriptions_for_planner()가 올바른 형식 반환
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# langgraph가 설치되지 않은 CI 환경 대응:
# graph/__init__.py -> builder -> langgraph 의존성을 우회한다.
# capabilities 모듈 자체는 langgraph에 의존하지 않으므로 직접 import 가능.
_LANGGRAPH_AVAILABLE = True
try:
    import langgraph  # noqa: F401
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    # langgraph 관련 모듈을 mock으로 등록
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

    # langgraph.graph에서 필요한 속성 설정
    _lg_graph = sys.modules["langgraph.graph"]
    _lg_graph.END = "END"  # type: ignore[attr-defined]
    _lg_graph.START = "START"  # type: ignore[attr-defined]
    _lg_graph.StateGraph = MagicMock()  # type: ignore[attr-defined]

    # langgraph.graph.message
    _lg_msg = sys.modules["langgraph.graph.message"]
    _lg_msg.add_messages = lambda x: x  # type: ignore[attr-defined]

    # langgraph.checkpoint.memory
    _lg_mem = sys.modules["langgraph.checkpoint.memory"]
    _lg_mem.MemorySaver = MagicMock()  # type: ignore[attr-defined]

    # langgraph.types
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

from src.inference.graph.capabilities.base import CapabilityBase, CapabilityMetadata
from src.inference.graph.capabilities.registry import (
    MVP_CAPABILITY_IDS,
    build_mvp_registry,
    get_all_metadata,
    get_mvp_capability_ids,
    is_mvp_capability,
)

# ---------------------------------------------------------------------------
# fixture: 테스트용 mock closure
# ---------------------------------------------------------------------------


def _make_mock_fn(name: str) -> AsyncMock:
    """async (query, context, session) -> dict 시그니처의 mock."""
    fn = AsyncMock(return_value={"text": f"mock_{name}", "results": []})
    fn.__name__ = name
    return fn


@pytest.fixture
def mock_fns():
    return {
        "rag_search_fn": _make_mock_fn("rag_search"),
        "draft_civil_response_fn": _make_mock_fn("draft_civil_response"),
        "append_evidence_fn": _make_mock_fn("append_evidence"),
    }


@pytest.fixture
def registry(mock_fns):
    return build_mvp_registry(
        rag_search_fn=mock_fns["rag_search_fn"],
        api_lookup_action=None,
        draft_civil_response_fn=mock_fns["draft_civil_response_fn"],
        append_evidence_fn=mock_fns["append_evidence_fn"],
    )


# ---------------------------------------------------------------------------
# MVP_CAPABILITY_IDS 일관성
# ---------------------------------------------------------------------------


class TestMvpCapabilityIds:
    """MVP capability stable identifier 검증."""

    def test_expected_ids(self):
        """MVP capability 8개가 정확히 등록되어 있다."""
        expected = {
            "rag_search",
            "api_lookup",
            "draft_civil_response",
            "append_evidence",
            "issue_detector",
            "stats_lookup",
            "keyword_analyzer",
            "demographics_lookup",
        }
        assert MVP_CAPABILITY_IDS == expected

    def test_get_mvp_capability_ids_returns_same(self):
        """get_mvp_capability_ids()가 MVP_CAPABILITY_IDS와 동일한 집합을 반환한다."""
        assert get_mvp_capability_ids() == MVP_CAPABILITY_IDS

    def test_is_mvp_capability_true(self):
        for name in MVP_CAPABILITY_IDS:
            assert is_mvp_capability(name) is True

    def test_is_mvp_capability_false(self):
        assert is_mvp_capability("non_existent_tool") is False
        assert is_mvp_capability("") is False
        assert is_mvp_capability("RAG_SEARCH") is False  # case sensitive


# ---------------------------------------------------------------------------
# build_mvp_registry
# ---------------------------------------------------------------------------


class TestBuildMvpRegistry:
    """build_mvp_registry() 팩토리 함수 검증."""

    def test_returns_all_four_capabilities(self, registry):
        """4개 capability 모두 반환된다."""
        assert set(registry.keys()) == MVP_CAPABILITY_IDS

    def test_all_are_capability_base(self, registry):
        """모든 값이 CapabilityBase 인스턴스이다."""
        for name, cap in registry.items():
            assert isinstance(cap, CapabilityBase), f"{name}은 CapabilityBase가 아님"

    def test_all_have_metadata(self, registry):
        """모든 capability가 CapabilityMetadata를 가진다."""
        for name, cap in registry.items():
            meta = cap.metadata
            assert isinstance(meta, CapabilityMetadata), f"{name}의 metadata 타입 불일치"
            assert meta.name == name, f"metadata.name({meta.name}) != registry key({name})"
            assert meta.description, f"{name}의 description이 비어있음"
            assert meta.approval_summary, f"{name}의 approval_summary가 비어있음"
            assert meta.provider, f"{name}의 provider가 비어있음"

    def test_metadata_name_matches_registry_key(self, registry):
        """metadata.name과 registry key가 동일하다 (stable identifier 일관성)."""
        for key, cap in registry.items():
            assert cap.metadata.name == key

    def test_all_are_callable(self, registry):
        """모든 capability가 callable이다 (RegistryExecutorAdapter 호환)."""
        for name, cap in registry.items():
            assert callable(cap), f"{name}은 callable이 아님"


# ---------------------------------------------------------------------------
# get_all_metadata
# ---------------------------------------------------------------------------


class TestGetAllMetadata:
    """planner용 metadata 목록 검증."""

    def test_returns_list_of_dicts(self, registry):
        result = get_all_metadata(registry)
        assert isinstance(result, list)
        assert len(result) == 8

    def test_each_metadata_has_required_fields(self, registry):
        """각 metadata dict에 필수 필드가 포함되어 있다."""
        required_fields = {"name", "description", "approval_summary", "provider", "timeout_sec"}
        for meta_dict in get_all_metadata(registry):
            assert required_fields.issubset(meta_dict.keys()), (
                f"{meta_dict.get('name', '?')}에 필수 필드 누락: "
                f"{required_fields - set(meta_dict.keys())}"
            )

    def test_names_match_mvp_ids(self, registry):
        """metadata 목록의 name 집합이 MVP_CAPABILITY_IDS와 동일하다."""
        names = {m["name"] for m in get_all_metadata(registry)}
        assert names == MVP_CAPABILITY_IDS


# ---------------------------------------------------------------------------
# RegistryExecutorAdapter 통합 검증
# ---------------------------------------------------------------------------


class TestRegistryExecutorAdapterIntegration:
    """RegistryExecutorAdapter가 CapabilityBase registry와 올바르게 동작하는지 검증."""

    def _make_adapter(self, registry):
        from src.inference.graph.executor_adapter import RegistryExecutorAdapter

        session_store = MagicMock()
        session_store.get_or_create.return_value = MagicMock()
        return RegistryExecutorAdapter(
            tool_registry=registry,
            session_store=session_store,
        )

    def test_list_tools(self, registry):
        adapter = self._make_adapter(registry)
        tools = adapter.list_tools()
        assert set(tools) == MVP_CAPABILITY_IDS

    def test_get_tool_metadata_returns_full_metadata(self, registry):
        """모든 tool이 CapabilityBase이므로 풍부한 metadata가 반환된다."""
        adapter = self._make_adapter(registry)
        for name in MVP_CAPABILITY_IDS:
            meta = adapter.get_tool_metadata(name)
            assert meta is not None
            assert meta["name"] == name
            assert meta["description"] != ""
            assert meta["approval_summary"] != ""

    def test_get_tool_metadata_unknown_returns_none(self, registry):
        adapter = self._make_adapter(registry)
        assert adapter.get_tool_metadata("unknown_tool") is None

    def test_get_tool_descriptions_for_planner(self, registry):
        """planner용 tool descriptions 메서드가 올바른 목록을 반환한다."""
        adapter = self._make_adapter(registry)
        descriptions = adapter.get_tool_descriptions_for_planner()
        assert len(descriptions) == 8
        names = {d["name"] for d in descriptions}
        assert names == MVP_CAPABILITY_IDS

    @pytest.mark.asyncio
    async def test_non_mvp_capability_blocked(self, registry):
        """비MVP capability 실행이 차단된다."""
        adapter = self._make_adapter(registry)
        result = await adapter.execute("non_existent_tool", "query", {})
        assert result["success"] is False
        assert "비MVP" in result["error"]


# ---------------------------------------------------------------------------
# plan_validator 연동 검증
# ---------------------------------------------------------------------------


class TestPlanValidatorIntegration:
    """plan_validator의 MVP_CAPABILITIES가 registry와 일치하는지 검증."""

    def test_plan_validator_uses_registry_source(self):
        from src.inference.graph.plan_validator import MVP_CAPABILITIES

        assert MVP_CAPABILITIES == MVP_CAPABILITY_IDS


# ---------------------------------------------------------------------------
# planner_adapter 연동 검증
# ---------------------------------------------------------------------------


class TestPlannerAdapterIntegration:
    """LLMPlannerAdapter의 system prompt가 registry 도구 목록을 동적 반영하는지 검증."""

    def test_system_prompt_contains_all_registry_tools(self):
        from src.inference.graph.planner_adapter import LLMPlannerAdapter

        prompt = LLMPlannerAdapter._build_system_prompt()
        for tool_id in MVP_CAPABILITY_IDS:
            assert tool_id in prompt, f"system prompt에 {tool_id}가 포함되어야 합니다"


# ---------------------------------------------------------------------------
# 개별 capability wrapper 검증
# ---------------------------------------------------------------------------


class TestCapabilityWrappers:
    """thin wrapper capability들의 execute/metadata 검증."""

    @pytest.mark.asyncio
    async def test_rag_search_delegates_to_fn(self, mock_fns):
        from src.inference.graph.capabilities.rag_search import RagSearchCapability

        cap = RagSearchCapability(execute_fn=mock_fns["rag_search_fn"])
        result = await cap.execute("테스트", {}, None)
        mock_fns["rag_search_fn"].assert_awaited_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_draft_civil_response_delegates_to_fn(self, mock_fns):
        from src.inference.graph.capabilities.draft_civil_response import (
            DraftCivilResponseCapability,
        )

        cap = DraftCivilResponseCapability(execute_fn=mock_fns["draft_civil_response_fn"])
        result = await cap.execute("테스트", {}, None)
        mock_fns["draft_civil_response_fn"].assert_awaited_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_append_evidence_delegates_to_fn(self, mock_fns):
        from src.inference.graph.capabilities.append_evidence import AppendEvidenceCapability

        cap = AppendEvidenceCapability(execute_fn=mock_fns["append_evidence_fn"])
        result = await cap.execute("테스트", {}, None)
        mock_fns["append_evidence_fn"].assert_awaited_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_rag_search_error_handling(self):
        from src.inference.graph.capabilities.rag_search import RagSearchCapability

        error_fn = AsyncMock(return_value={"error": "검색 실패"})
        cap = RagSearchCapability(execute_fn=error_fn)
        result = await cap.execute("테스트", {}, None)
        assert result.success is False
        assert result.error == "검색 실패"

    @pytest.mark.asyncio
    async def test_capability_call_returns_dict(self, mock_fns):
        """__call__이 dict를 반환한다 (RegistryExecutorAdapter 호환)."""
        from src.inference.graph.capabilities.rag_search import RagSearchCapability

        cap = RagSearchCapability(execute_fn=mock_fns["rag_search_fn"])
        result = await cap("테스트", {}, None)
        assert isinstance(result, dict)
        assert "success" in result
        assert "latency_ms" in result
