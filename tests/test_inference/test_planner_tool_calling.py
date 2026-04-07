"""Planner 네이티브 tool calling 단위 테스트."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# langgraph/langchain mock setup — CI 환경에서 미설치 대응
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMPlannerToolCalling:
    """LLMPlannerAdapter 네이티브 tool calling 검증."""

    @pytest.mark.asyncio
    async def test_native_tool_calls_parsed(self):
        """LLM이 tool_calls를 반환하면 ToolPlan으로 파싱된다."""
        from src.inference.graph.planner_adapter import LLMPlannerAdapter

        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"name": "rag_search", "args": {"query": "소음 민원"}},
            {"name": "draft_civil_response", "args": {"query": "소음 민원 답변", "adapter": "civil"}},
        ]
        mock_response.content = ""

        mock_llm = MagicMock()
        mock_llm_with_tools = AsyncMock(return_value=mock_response)
        mock_llm.bind_tools = MagicMock(return_value=MagicMock(ainvoke=mock_llm_with_tools))

        adapter = LLMPlannerAdapter(llm=mock_llm)

        mock_msg = MagicMock()
        mock_msg.content = "소음 민원 답변해줘"

        plan = await adapter.plan(messages=[mock_msg], context={})

        assert "rag_search" in plan.tools
        assert "draft_civil_response" in plan.tools
        assert plan.tool_args.get("draft_civil_response", {}).get("adapter") == "civil"
        assert "tool_calling" in plan.adapter_mode

    @pytest.mark.asyncio
    async def test_json_fallback(self):
        """tool_calls가 없으면 JSON 텍스트 fallback으로 파싱된다."""
        from src.inference.graph.planner_adapter import LLMPlannerAdapter

        json_response = json.dumps({
            "task_type": "draft_response",
            "goal": "민원 답변 작성",
            "reason": "사용자가 답변을 요청함",
            "tools": ["rag_search", "draft_civil_response"],
        })

        mock_response = MagicMock()
        mock_response.tool_calls = []  # empty
        mock_response.content = json_response

        mock_llm = MagicMock()
        mock_llm_with_tools = AsyncMock(return_value=mock_response)
        mock_llm.bind_tools = MagicMock(return_value=MagicMock(ainvoke=mock_llm_with_tools))

        adapter = LLMPlannerAdapter(llm=mock_llm)

        mock_msg = MagicMock()
        mock_msg.content = "답변해줘"

        plan = await adapter.plan(messages=[mock_msg], context={})

        assert plan.tools == ["rag_search", "draft_civil_response"]
        assert plan.adapter_mode == "llm"


class TestDirectEnginePlannerToolCalling:
    """DirectEnginePlannerAdapter Hermes tool calling 검증."""

    def test_parse_hermes_tool_calls(self):
        """Hermes <tool_call> 태그를 올바르게 파싱한다."""
        from src.inference.graph.planner_adapter import DirectEnginePlannerAdapter

        text = (
            '<tool_call>{"name": "rag_search", "arguments": {"query": "소음 민원"}}</tool_call>\n'
            '<tool_call>{"name": "draft_civil_response", "arguments": {"query": "답변", "adapter": "civil"}}</tool_call>'
        )

        result = DirectEnginePlannerAdapter._parse_hermes_tool_calls(text)

        assert len(result) == 2
        assert result[0]["name"] == "rag_search"
        assert result[1]["name"] == "draft_civil_response"
        assert result[1]["arguments"]["adapter"] == "civil"

    def test_parse_hermes_empty(self):
        """tool_call 태그가 없으면 빈 리스트를 반환한다."""
        from src.inference.graph.planner_adapter import DirectEnginePlannerAdapter

        result = DirectEnginePlannerAdapter._parse_hermes_tool_calls("그냥 텍스트")
        assert result == []

    def test_parse_hermes_malformed_json(self):
        """잘못된 JSON은 무시한다."""
        from src.inference.graph.planner_adapter import DirectEnginePlannerAdapter

        text = '<tool_call>{bad json}</tool_call>\n<tool_call>{"name": "rag_search", "arguments": {}}</tool_call>'

        result = DirectEnginePlannerAdapter._parse_hermes_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "rag_search"


class TestToolPlanWithArgs:
    """ToolPlan.tool_args 검증."""

    def test_tool_plan_default_args(self):
        """tool_args 기본값은 빈 dict."""
        from src.inference.graph.state import TaskType, ToolPlan

        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="test",
            reason="test",
            tools=["rag_search"],
        )
        assert plan.tool_args == {}

    def test_tool_plan_with_adapter_args(self):
        """tool_args에 adapter 정보가 포함된다."""
        from src.inference.graph.state import TaskType, ToolPlan

        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="test",
            reason="test",
            tools=["draft_civil_response"],
            tool_args={"draft_civil_response": {"query": "test", "adapter": "civil"}},
        )
        assert plan.tool_args["draft_civil_response"]["adapter"] == "civil"


class TestBuildToolDefinitions:
    """build_tool_definitions() 검증."""

    def test_returns_openai_format(self):
        """OpenAI tool definition 형식을 반환한다."""
        from src.inference.graph.capabilities.registry import build_mvp_registry, build_tool_definitions

        registry = build_mvp_registry(
            rag_search_fn=AsyncMock(return_value={"text": "", "results": []}),
            api_lookup_action=None,
            draft_civil_response_fn=AsyncMock(return_value={"text": "", "results": []}),
            append_evidence_fn=AsyncMock(return_value={"text": "", "results": []}),
        )

        definitions = build_tool_definitions(registry)

        assert len(definitions) == 8
        for defn in definitions:
            assert defn["type"] == "function"
            assert "name" in defn["function"]
            assert "description" in defn["function"]
            assert "parameters" in defn["function"]

    def test_adapter_tools_have_adapter_param(self):
        """draft_civil_response, append_evidence에 adapter 파라미터가 있다."""
        from src.inference.graph.capabilities.registry import build_mvp_registry, build_tool_definitions

        registry = build_mvp_registry(
            rag_search_fn=AsyncMock(return_value={"text": "", "results": []}),
            api_lookup_action=None,
            draft_civil_response_fn=AsyncMock(return_value={"text": "", "results": []}),
            append_evidence_fn=AsyncMock(return_value={"text": "", "results": []}),
        )

        definitions = build_tool_definitions(registry)
        adapter_tools = {
            d["function"]["name"]: d
            for d in definitions
            if d["function"]["name"] in ("draft_civil_response", "append_evidence")
        }

        for name, defn in adapter_tools.items():
            params = defn["function"]["parameters"]
            assert "adapter" in params.get("properties", {}), f"{name}에 adapter 파라미터 없음"
