"""LangGraph 기반 runtime foundation 모듈.

Issue: #415

이 모듈은 GovOn의 기존 runtime/tool 계층 위에 LangGraph를 연결하기 위한
최소 기반을 제공한다.

- vLLM/OpenAI-compatible endpoint에 연결되는 모델 어댑터
- 기존 ToolType registry / BaseAction 계층을 graph node에서 호출할 수 있는 어댑터
- graph runtime compile / invoke를 검증할 수 있는 최소 상태 그래프

주의:
- 본 모듈은 #415 범위에 맞춰 foundation integration까지만 담당한다.
- 정교한 state routing/guardrail은 #410, full graph runtime 전환은 #409,
  checkpoint/recovery는 #418에서 이어진다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Protocol, TypedDict

from loguru import logger

from .actions.base import BaseAction
from .runtime_config import RuntimeConfig
from .session_context import SessionContext
from .tool_router import ToolType

LangGraphToolCallable = Callable[[str, Dict[str, Any], SessionContext], Awaitable[Dict[str, Any]]]

_DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8001/v1"
_DEFAULT_SYSTEM_PROMPT = (
    "당신은 GovOn 행정 업무 어시스턴트입니다. "
    "근거가 주어지면 이를 우선 반영하고, 근거가 없으면 추측을 줄여 간결하게 답하세요."
)


class LangGraphDependencyError(RuntimeError):
    """LangGraph 또는 adapter 의존성이 없을 때 발생한다."""


class SupportsModelAdapter(Protocol):
    """LangGraph runtime이 요구하는 최소 모델 어댑터 프로토콜."""

    async def ainvoke(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """시스템/사용자 프롬프트를 입력받아 최종 텍스트를 반환한다."""


class LangGraphState(TypedDict, total=False):
    """GovOn LangGraph runtime의 최소 상태 스키마."""

    query: str
    session: SessionContext
    session_id: str
    session_summary: str
    requested_action: Optional[str]
    available_actions: List[str]
    route: str
    action_result: Dict[str, Any]
    citations: List[Dict[str, Any]]
    context_text: str
    final_text: str
    error: Optional[str]


def _import_langgraph_components() -> tuple[Any, Any, Any]:
    """LangGraph 구성요소를 지연 import한다."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise LangGraphDependencyError(
            "LangGraph runtime을 사용하려면 'langgraph' 패키지가 필요합니다."
        ) from exc

    return START, END, StateGraph


def _import_langchain_openai_components() -> tuple[Any, Any, Any]:
    """langchain-openai 구성요소를 지연 import한다."""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise LangGraphDependencyError(
            "LangGraph 모델 어댑터를 사용하려면 'langchain-openai'가 필요합니다."
        ) from exc

    return ChatOpenAI, HumanMessage, SystemMessage


def _coerce_text_content(content: Any) -> str:
    """LangChain 응답 content를 단일 문자열로 정규화한다."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content") or ""
                if text_value:
                    parts.append(str(text_value))
            else:
                text_value = getattr(item, "text", "") or getattr(item, "content", "")
                if text_value:
                    parts.append(str(text_value))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


@dataclass
class LangGraphModelAdapter:
    """vLLM/OpenAI-compatible endpoint를 LangGraph에서 쓰기 위한 모델 어댑터."""

    client: Any
    model: str
    base_url: str
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 512
    _human_message_cls: Any = field(default=None, repr=False)
    _system_message_cls: Any = field(default=None, repr=False)

    @classmethod
    def from_vllm_endpoint(
        cls,
        *,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> "LangGraphModelAdapter":
        """vLLM OpenAI-compatible endpoint를 사용하는 ChatOpenAI 어댑터를 만든다."""

        ChatOpenAI, HumanMessage, SystemMessage = _import_langchain_openai_components()
        client = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return cls(
            client=client,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            _human_message_cls=HumanMessage,
            _system_message_cls=SystemMessage,
        )

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RuntimeConfig,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "LangGraphModelAdapter":
        """RuntimeConfig와 환경변수를 기준으로 endpoint 어댑터를 초기화한다."""

        resolved_base_url = base_url or os.getenv("LANGGRAPH_MODEL_BASE_URL", _DEFAULT_VLLM_BASE_URL)
        resolved_api_key = api_key or os.getenv("LANGGRAPH_MODEL_API_KEY", "EMPTY")

        return cls.from_vllm_endpoint(
            model=runtime_config.model.model_path,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            temperature=runtime_config.generation.temperature,
            max_tokens=runtime_config.generation.max_tokens,
        )

    async def ainvoke(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """시스템/사용자 프롬프트를 endpoint에 전달하고 텍스트를 반환한다."""

        messages: List[Any] = []
        if system_prompt:
            if self._system_message_cls is not None:
                messages.append(self._system_message_cls(content=system_prompt))
            else:
                messages.append({"role": "system", "content": system_prompt})

        if self._human_message_cls is not None:
            messages.append(self._human_message_cls(content=user_prompt))
        else:
            messages.append({"role": "user", "content": user_prompt})

        response = await self.client.ainvoke(messages)
        return _coerce_text_content(getattr(response, "content", response))


@dataclass
class LangGraphToolAdapter:
    """기존 tool/action callable을 LangGraph node에서 쓰기 위한 어댑터."""

    name: str
    runner: LangGraphToolCallable
    description: str = ""

    @classmethod
    def from_callable(
        cls,
        *,
        name: str,
        runner: LangGraphToolCallable,
        description: str = "",
    ) -> "LangGraphToolAdapter":
        return cls(name=name, runner=runner, description=description)

    @classmethod
    def from_action(
        cls,
        *,
        name: str,
        action: BaseAction,
        description: str = "",
    ) -> "LangGraphToolAdapter":
        return cls(name=name, runner=action, description=description or action.__class__.__name__)

    async def run(
        self,
        query: str,
        context: Dict[str, Any],
        session: SessionContext,
    ) -> Dict[str, Any]:
        """LangGraph node에서 호출할 실행 진입점."""

        return await self.runner(query, context, session)


def build_graph_tool_adapters(
    *,
    tool_registry: Optional[Mapping[Any, LangGraphToolCallable]] = None,
    action_registry: Optional[Mapping[str, BaseAction]] = None,
) -> Dict[str, LangGraphToolAdapter]:
    """기존 tool registry / action registry를 LangGraph adapter 맵으로 변환한다."""

    adapters: Dict[str, LangGraphToolAdapter] = {}

    for raw_name, runner in (tool_registry or {}).items():
        name = raw_name.value if isinstance(raw_name, ToolType) else str(raw_name)
        adapters[name] = LangGraphToolAdapter.from_callable(name=name, runner=runner)

    for name, action in (action_registry or {}).items():
        adapters[name] = LangGraphToolAdapter.from_action(name=name, action=action)

    return adapters


@dataclass
class GovOnLangGraphRuntime:
    """GovOn용 LangGraph foundation runtime."""

    model_adapter: SupportsModelAdapter
    tools: Mapping[str, LangGraphToolAdapter]
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    graph: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.graph = self._compile_graph()

    def _compile_graph(self) -> Any:
        """LangGraph state graph를 compile한다."""

        START, END, StateGraph = _import_langgraph_components()
        builder = StateGraph(LangGraphState)

        builder.add_node("decide", self._decide_node)
        builder.add_node("run_action", self._run_action_node)
        builder.add_node("respond", self._respond_node)

        builder.add_edge(START, "decide")
        builder.add_conditional_edges(
            "decide",
            self._route_from_state,
            {
                "tool": "run_action",
                "respond": "respond",
            },
        )
        builder.add_edge("run_action", "respond")
        builder.add_edge("respond", END)
        return builder.compile()

    def _decide_node(self, state: LangGraphState) -> Dict[str, Any]:
        """요청된 action이 있으면 tool route, 없으면 direct response route를 선택한다."""

        requested_action = state.get("requested_action")
        if requested_action and requested_action in self.tools:
            return {"route": "tool"}

        if requested_action and requested_action not in self.tools:
            return {
                "route": "respond",
                "error": f"등록되지 않은 action입니다: {requested_action}",
            }

        return {"route": "respond"}

    @staticmethod
    def _route_from_state(state: LangGraphState) -> str:
        """조건 분기용 route selector."""

        return "tool" if state.get("route") == "tool" else "respond"

    async def _run_action_node(self, state: LangGraphState) -> Dict[str, Any]:
        """선택된 action/tool을 실행한다."""

        session = state.get("session") or SessionContext(session_id=state.get("session_id", ""))
        requested_action = state.get("requested_action")

        if not requested_action or requested_action not in self.tools:
            return {"error": "실행할 action이 선택되지 않았습니다."}

        adapter = self.tools[requested_action]
        context = {
            "session_context": state.get("session_summary", ""),
            "available_actions": state.get("available_actions", []),
            "route": state.get("route", ""),
        }
        result = await adapter.run(state["query"], context, session)

        return {
            "action_result": result,
            "citations": result.get("citations", []),
            "context_text": result.get("context_text", ""),
            "error": result.get("error"),
        }

    async def _respond_node(self, state: LangGraphState) -> Dict[str, Any]:
        """모델 어댑터로 최종 응답을 생성한다."""

        user_prompt = self._build_user_prompt(state)
        final_text = await self.model_adapter.ainvoke(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
        )
        return {"final_text": final_text}

    def _build_user_prompt(self, state: LangGraphState) -> str:
        """action 결과와 세션 문맥을 포함한 사용자 프롬프트를 구성한다."""

        lines = [f"사용자 요청:\n{state['query']}"]

        session_summary = state.get("session_summary")
        if session_summary:
            lines.append(f"세션 요약:\n{session_summary}")

        context_text = state.get("context_text")
        if context_text:
            lines.append(f"참고 컨텍스트:\n{context_text}")

        citations = state.get("citations") or []
        if citations:
            citation_lines = []
            for citation in citations[:5]:
                title = citation.get("title", "")
                url = citation.get("url", "")
                snippet = citation.get("snippet", "")
                citation_lines.append(f"- {title} | {url} | {snippet}".strip())
            lines.append("출처:\n" + "\n".join(citation_lines))

        error = state.get("error")
        if error:
            lines.append(f"주의사항:\n{error}")

        lines.append("위 정보를 바탕으로 한국어로 간결하고 실무형 응답을 생성하세요.")
        return "\n\n".join(lines)

    async def ainvoke(
        self,
        *,
        query: str,
        session: Optional[SessionContext] = None,
        requested_action: Optional[str] = None,
    ) -> LangGraphState:
        """초기 상태를 만들어 graph를 실행한다."""

        session_obj = session or SessionContext()
        initial_state: LangGraphState = {
            "query": query,
            "session": session_obj,
            "session_id": session_obj.session_id,
            "session_summary": session_obj.build_context_summary(),
            "requested_action": requested_action,
            "available_actions": sorted(self.tools.keys()),
        }
        return await self.graph.ainvoke(initial_state)


def build_langgraph_runtime(
    *,
    runtime_config: RuntimeConfig,
    tool_registry: Optional[Mapping[Any, LangGraphToolCallable]] = None,
    action_registry: Optional[Mapping[str, BaseAction]] = None,
    model_adapter: Optional[SupportsModelAdapter] = None,
) -> GovOnLangGraphRuntime:
    """RuntimeConfig와 기존 registry를 사용해 GovOn LangGraph runtime을 만든다."""

    resolved_model_adapter = model_adapter or LangGraphModelAdapter.from_runtime_config(runtime_config)
    adapters = build_graph_tool_adapters(
        tool_registry=tool_registry,
        action_registry=action_registry,
    )
    logger.info(f"LangGraph runtime foundation 구성 완료: tools={list(adapters.keys())}")
    return GovOnLangGraphRuntime(
        model_adapter=resolved_model_adapter,
        tools=adapters,
    )
