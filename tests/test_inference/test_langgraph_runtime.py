from types import SimpleNamespace

import pytest

from src.inference.actions.base import ActionResult, BaseAction
from src.inference.langgraph_runtime import (
    GovOnLangGraphRuntime,
    LangGraphModelAdapter,
    LangGraphToolAdapter,
    build_graph_tool_adapters,
    build_langgraph_runtime,
)
from src.inference.runtime_config import RuntimeConfig
from src.inference.session_context import SessionContext
from src.inference.tool_router import ToolType


class FakeCompiledGraph:
    def __init__(self, nodes, edges, conditional):
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional

    async def ainvoke(self, state):
        current = self._edges["__start__"][0]
        working = dict(state)

        while current != "__end__":
            node_fn = self._nodes[current]
            update = node_fn(working)
            if hasattr(update, "__await__"):
                update = await update
            working.update(update or {})

            if current in self._conditional:
                condition_fn, mapping = self._conditional[current]
                branch = condition_fn(working)
                current = mapping[branch]
                continue

            current = self._edges.get(current, ["__end__"])[0]

        return working


class FakeStateGraph:
    def __init__(self, _state_type):
        self._nodes = {}
        self._edges = {}
        self._conditional = {}

    def add_node(self, name, node):
        self._nodes[name] = node

    def add_edge(self, start, end):
        self._edges.setdefault(start, []).append(end)

    def add_conditional_edges(self, node_name, selector, mapping):
        self._conditional[node_name] = (selector, mapping)

    def compile(self):
        return FakeCompiledGraph(self._nodes, self._edges, self._conditional)


class FakeModelAdapter:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, user_prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
            }
        )
        return "LangGraph 응답 초안"


class EchoAction(BaseAction):
    def __init__(self):
        super().__init__("echo_action")

    async def execute(self, query, context, session):
        return ActionResult(
            success=True,
            data={"query": query, "session_id": session.session_id},
            source="echo_action",
            citations=[],
            context_text=f"action-context::{query}",
        )


@pytest.fixture(autouse=True)
def fake_langgraph_import(monkeypatch):
    from src.inference import langgraph_runtime as module

    monkeypatch.setattr(
        module,
        "_import_langgraph_components",
        lambda: ("__start__", "__end__", FakeStateGraph),
    )


def test_build_graph_tool_adapters_wraps_tool_registry_and_actions():
    async def fake_search(query, context, session):
        return {"success": True, "data": {"query": query}}

    adapters = build_graph_tool_adapters(
        tool_registry={ToolType.SEARCH: fake_search},
        action_registry={"minwon_analysis": EchoAction()},
    )

    assert set(adapters.keys()) == {"search", "minwon_analysis"}
    assert isinstance(adapters["search"], LangGraphToolAdapter)
    assert isinstance(adapters["minwon_analysis"], LangGraphToolAdapter)


@pytest.mark.asyncio
async def test_langgraph_runtime_compiles_and_invokes_tool_route():
    model_adapter = FakeModelAdapter()
    runtime = GovOnLangGraphRuntime(
        model_adapter=model_adapter,
        tools={
            "minwon_analysis": LangGraphToolAdapter.from_action(
                name="minwon_analysis",
                action=EchoAction(),
            )
        },
    )

    session = SessionContext(session_id="session-lg-001")
    result = await runtime.ainvoke(
        query="포트홀 민원 답변 초안 작성",
        session=session,
        requested_action="minwon_analysis",
    )

    assert result["route"] == "tool"
    assert result["action_result"]["success"] is True
    assert result["context_text"] == "action-context::포트홀 민원 답변 초안 작성"
    assert result["final_text"] == "LangGraph 응답 초안"
    assert "action-context::포트홀 민원 답변 초안 작성" in model_adapter.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_build_langgraph_runtime_uses_runtime_config():
    runtime_config = RuntimeConfig.from_env()
    model_adapter = FakeModelAdapter()

    runtime = build_langgraph_runtime(
        runtime_config=runtime_config,
        tool_registry={},
        action_registry={"minwon_analysis": EchoAction()},
        model_adapter=model_adapter,
    )

    result = await runtime.ainvoke(
        query="유사 민원 사례를 정리해줘",
        session=SessionContext(session_id="session-lg-002"),
        requested_action="minwon_analysis",
    )

    assert result["final_text"] == "LangGraph 응답 초안"
    assert "minwon_analysis" in runtime.tools


@pytest.mark.asyncio
async def test_langgraph_model_adapter_connects_to_vllm_endpoint(monkeypatch):
    captured = {}

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        async def ainvoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="endpoint-ok")

    from src.inference import langgraph_runtime as module

    monkeypatch.setattr(
        module,
        "_import_langchain_openai_components",
        lambda: (FakeChatOpenAI, FakeMessage, FakeMessage),
    )

    adapter = LangGraphModelAdapter.from_vllm_endpoint(
        model="govon-test-model",
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        temperature=0.2,
        max_tokens=256,
    )

    text = await adapter.ainvoke("사용자 질문", system_prompt="시스템 프롬프트")

    assert text == "endpoint-ok"
    assert captured["kwargs"]["model"] == "govon-test-model"
    assert captured["kwargs"]["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["kwargs"]["api_key"] == "EMPTY"
    assert len(captured["messages"]) == 2
