# LangGraph Runtime Design for Issue #415

**Status**: Accepted  
**Date**: 2026-04-05  
**Issue**: #415 — [Task 4.0] LangGraph runtime 기반 및 planner/executor adapter 구성  
**Implements**: ADR-006 Section 3 (Approval-Gated Task Loop) via LangGraph StateGraph  
**Target Consumer**: engineering-ai-engineer agent (구현 담당)

---

## 1. Scope & Non-Scope

### In Scope (MVP)
- LangGraph `StateGraph` 기반 6-node graph: `session_load -> planner -> approval_wait -> tool_execute -> synthesis -> persist`
- Planner adapter interface: `LLMPlannerAdapter` 기본, `RegexPlannerAdapter` CI fallback
- Executor adapter interface (tool registry 연동)
- `interrupt()` 기반 human-in-the-loop approval gate
- `SqliteSaver` 기반 graph checkpoint (GovOn 기존 SQLite와 별도 DB 파일)
- Smoke test: graph 초기화 및 단일 사이클 검증

### Out of Scope (Non-MVP)
- 분산 checkpoint 엔진
- 승인 없는 자율 루프
- 원격 멀티테넌트
- 복잡한 graph checkpoint와 GovOn sessions DB 통합
- 웹 UI 연동

---

## 2. Dependency Pinning

### 2.1 requirements.txt 변경

현재 `requirements.txt`에서 다음 라인을 교체/추가한다:

```diff
- langgraph>=0.2.0
+ langgraph==1.1.4
+ langgraph-checkpoint==4.0.1
+ langgraph-checkpoint-sqlite==3.0.3
  langchain-openai>=0.3.0
+ langchain-core>=1.2.0,<2.0.0
```

**근거**: `langgraph>=0.2.0`은 1.x와 0.x의 API 차이가 매우 크다. 1.1.4는 현재 최신 stable이며, `interrupt()` API가 안정화된 버전이다. checkpoint 패키지는 `SqliteSaver`를 위해 필수다.

### 2.2 설치 검증 명령

```bash
pip install langgraph==1.1.4 langgraph-checkpoint-sqlite==3.0.3
python -c "from langgraph.graph import StateGraph; print('OK')"
```

---

## 3. Graph State Schema

### 3.1 파일 위치

`src/inference/graph/state.py` (신규 생성)

### 3.2 State 정의

```python
"""GovOn LangGraph state schema."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TaskType(str, Enum):
    DRAFT_RESPONSE = "draft_response"       # 민원 답변 초안 작성
    REVISE_RESPONSE = "revise_response"     # 답변 수정
    APPEND_EVIDENCE = "append_evidence"     # 근거 보강 (v4에서 삭제됨 — evidence augmentation은 follow-up으로 처리)
    LOOKUP_STATS = "lookup_stats"           # 통계/사례 조회


@dataclass
class ToolPlan:
    """planner가 생성하는 구조화된 실행 계획."""
    task_type: TaskType
    goal: str                               # 사용자에게 보여줄 작업 설명 (한국어)
    reason: str                             # 왜 이 작업이 필요한지
    tools: List[str]                        # 실행할 tool 이름 목록 (순서대로)
    # 예: ["rag_search", "api_lookup", "draft_response"]


class GovOnState:
    """LangGraph StateGraph의 typed dict 형태로 사용할 state.

    실제 구현 시 TypedDict로 선언한다.
    """
    pass


# --- 실제 구현용 TypedDict ---

from typing import TypedDict


class GovOnGraphState(TypedDict, total=False):
    """GovOn LangGraph graph state.

    모든 노드가 공유하는 state object.
    planner와 executor가 동일한 state를 읽고 쓴다.
    """

    # --- 세션 식별 ---
    session_id: str
    request_id: str

    # --- 메시지 히스토리 (LangGraph add_messages reducer 사용) ---
    messages: Annotated[Sequence[AnyMessage], add_messages]

    # --- planner 출력 ---
    task_type: str                          # TaskType.value
    goal: str                               # 승인 프롬프트에 표시할 작업 설명
    reason: str                             # 작업 이유
    planned_tools: List[str]                # 실행 예정 tool 이름 리스트

    # --- approval gate ---
    approval_status: str                    # ApprovalStatus.value
    # interrupt()에서 사용자 입력을 받으면 이 필드가 갱신된다

    # --- executor 출력 ---
    tool_results: Dict[str, Any]            # {tool_name: result_dict, ...}
    accumulated_context: Dict[str, Any]     # tool 간 전달되는 누적 컨텍스트

    # --- synthesis 출력 ---
    final_text: str                         # 최종 사용자 응답 텍스트

    # --- 메타데이터 ---
    error: Optional[str]
    total_latency_ms: float
```

### 3.3 State 설계 결정 근거

| 필드 | 근거 |
|---|---|
| `messages` | `add_messages` reducer를 사용해 LangGraph가 메시지를 자동 병합한다. planner LLM 호출 시 대화 맥락으로 사용. |
| `planned_tools` | planner가 결정한 tool 목록을 approval_wait과 tool_execute가 공유해야 한다. |
| `approval_status` | `interrupt()` 후 사용자 입력을 받아 갱신. `REJECTED`면 graph가 `END`로 분기. |
| `tool_results` | executor가 tool별 결과를 dict로 누적. synthesis 노드가 읽는다. |
| `accumulated_context` | 기존 `AgentLoop`의 `accumulated` dict를 계승. tool 간 데이터 전달용. |

---

## 4. Graph Topology

### 4.1 파일 위치

`src/inference/graph/builder.py` (신규 생성)

### 4.2 노드 구성

```
START
  |
  v
[session_load] ---> [planner] ---> [approval_wait] ---> [tool_execute] ---> [synthesis] ---> [persist] ---> END
                                        |
                                        | (rejected)
                                        v
                                       END
```

### 4.3 각 노드 상세

#### Node 1: `session_load`

**역할**: 세션 컨텍스트를 로드하고 state를 초기화한다.

**입력**: `session_id`, 사용자 메시지 (messages[-1])  
**출력**: `messages`에 세션 히스토리 추가, `accumulated_context`에 세션 요약 삽입

```python
async def session_load(state: GovOnGraphState) -> dict:
    """세션 로드 노드.

    SessionStore에서 기존 세션을 불러오거나 새 세션을 생성한다.
    대화 히스토리와 tool 사용 기록을 state에 주입한다.
    """
    session_store: SessionStore  # graph config에서 주입
    session = session_store.get_or_create(state.get("session_id"))

    context_summary = session.build_context_summary()
    return {
        "session_id": session.session_id,
        "accumulated_context": {
            "session_context": context_summary,
            "query": state["messages"][-1].content,
        },
    }
```

#### Node 2: `planner`

**역할**: 사용자 요청을 분석하고 실행 계획을 생성한다. 기존 `ToolRouter`의 정규식 로직을 LLM 호출로 교체.

**입력**: `messages`, `accumulated_context`  
**출력**: `task_type`, `goal`, `reason`, `planned_tools`

```python
async def planner(state: GovOnGraphState) -> dict:
    """Planner 노드.

    PlannerAdapter를 호출하여 구조화된 실행 계획을 생성한다.
    """
    planner_adapter: PlannerAdapter  # graph config에서 주입
    plan: ToolPlan = await planner_adapter.plan(
        messages=state["messages"],
        context=state.get("accumulated_context", {}),
    )
    return {
        "task_type": plan.task_type.value,
        "goal": plan.goal,
        "reason": plan.reason,
        "planned_tools": plan.tools,
    }
```

#### Node 3: `approval_wait`

**역할**: 사용자 승인을 요청하고 `interrupt()`로 대기한다. ADR-006의 핵심 요구사항.

**입력**: `goal`, `reason`, `planned_tools`  
**출력**: `approval_status`

```python
from langgraph.types import interrupt, Command

async def approval_wait(state: GovOnGraphState) -> dict:
    """Human-in-the-loop 승인 게이트.

    interrupt()를 호출하여 graph 실행을 일시 정지한다.
    FastAPI 엔드포인트가 사용자 응답을 받아 graph를 resume한다.
    """
    approval_request = {
        "type": "approval_request",
        "goal": state["goal"],
        "reason": state["reason"],
        "planned_tools": state["planned_tools"],
        "prompt": f"다음 작업을 수행하겠습니다:\n\n"
                  f"  {state['goal']}\n\n"
                  f"  이유: {state['reason']}\n"
                  f"  사용할 도구: {', '.join(state['planned_tools'])}\n\n"
                  f"승인하시겠습니까? (승인/거절)",
    }

    # interrupt()는 graph 실행을 멈추고, resume 시 반환값이 된다
    user_response = interrupt(approval_request)

    # user_response는 resume 시 전달되는 값
    # 예: {"approved": True} 또는 {"approved": False}
    if isinstance(user_response, dict) and user_response.get("approved"):
        return {"approval_status": ApprovalStatus.APPROVED.value}
    else:
        return {"approval_status": ApprovalStatus.REJECTED.value}
```

#### Node 4: `tool_execute`

**역할**: 승인된 계획의 tool들을 순차 실행한다.

**입력**: `planned_tools`, `accumulated_context`, `messages`  
**출력**: `tool_results`, `accumulated_context` 갱신

```python
async def tool_execute(state: GovOnGraphState) -> dict:
    """Tool executor 노드.

    ExecutorAdapter를 통해 planned_tools를 순차 실행하고
    결과를 accumulated_context에 누적한다.
    """
    executor_adapter: ExecutorAdapter  # graph config에서 주입
    accumulated = dict(state.get("accumulated_context", {}))
    tool_results: Dict[str, Any] = {}

    for tool_name in state["planned_tools"]:
        result = await executor_adapter.execute(
            tool_name=tool_name,
            query=accumulated.get("query", ""),
            context=accumulated,
        )
        tool_results[tool_name] = result
        accumulated[tool_name] = result if result.get("success", True) else {}

    return {
        "tool_results": tool_results,
        "accumulated_context": accumulated,
    }
```

#### Node 5: `synthesis`

**역할**: tool 결과를 종합하여 최종 응답을 생성한다.

**입력**: `tool_results`, `accumulated_context`, `task_type`  
**출력**: `final_text`, `messages`에 assistant 응답 추가

```python
from langchain_core.messages import AIMessage

async def synthesis(state: GovOnGraphState) -> dict:
    """결과 종합 노드.

    기존 AgentLoop._extract_final_text() 로직을 계승한다.
    task_type에 따라 적절한 형식으로 최종 텍스트를 구성한다.
    """
    accumulated = state.get("accumulated_context", {})
    task_type = state.get("task_type", "")

    final_text = _extract_final_text(accumulated, task_type)

    return {
        "final_text": final_text,
        "messages": [AIMessage(content=final_text)],
    }
```

#### Node 6: `persist`

**역할**: 세션에 대화 기록과 tool 사용 기록을 저장한다.

**입력**: 전체 state  
**출력**: (없음 -- side effect로 DB에 저장)

```python
async def persist(state: GovOnGraphState) -> dict:
    """영속화 노드.

    SessionStore에 대화 턴과 tool 실행 기록을 저장한다.
    기존 SessionContext._persist_turn / _persist_tool_run 로직 계승.
    """
    session_store: SessionStore  # graph config에서 주입
    session = session_store.get_or_create(state["session_id"])

    # 사용자 입력 저장
    user_msg = state["messages"][0]  # 마지막 사용자 메시지
    session.add_turn("user", user_msg.content)

    # tool 실행 기록 저장
    for tool_name, result in state.get("tool_results", {}).items():
        session.add_tool_run(
            tool=tool_name,
            success=result.get("success", True),
            latency_ms=result.get("latency_ms", 0.0),
            error=result.get("error"),
        )

    # 어시스턴트 응답 저장
    session.add_turn("assistant", state.get("final_text", ""))

    return {}
```

### 4.4 Conditional Edge: approval 분기

```python
def route_after_approval(state: GovOnGraphState) -> str:
    """approval_wait 이후 분기 조건.

    approved -> tool_execute
    rejected -> END
    """
    if state.get("approval_status") == ApprovalStatus.APPROVED.value:
        return "tool_execute"
    return "__end__"
```

### 4.5 Graph Builder

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

def build_govon_graph(
    planner_adapter: "PlannerAdapter",
    executor_adapter: "ExecutorAdapter",
    session_store: "SessionStore",
    checkpointer: AsyncSqliteSaver | None = None,
) -> StateGraph:
    """GovOn MVP StateGraph를 구성한다.

    Parameters
    ----------
    planner_adapter : PlannerAdapter
        planner LLM adapter 인스턴스.
    executor_adapter : ExecutorAdapter
        tool executor adapter 인스턴스.
    session_store : SessionStore
        GovOn 세션 저장소.
    checkpointer : AsyncSqliteSaver | None
        LangGraph checkpoint 저장소.
        None이면 MemorySaver를 사용한다.

    Returns
    -------
    CompiledGraph
        컴파일된 LangGraph.
    """
    from langgraph.checkpoint.memory import MemorySaver

    graph = StateGraph(GovOnGraphState)

    # --- 노드 등록 ---
    graph.add_node("session_load", session_load)
    graph.add_node("planner", planner)
    graph.add_node("approval_wait", approval_wait)
    graph.add_node("tool_execute", tool_execute)
    graph.add_node("synthesis", synthesis)
    graph.add_node("persist", persist)

    # --- 엣지 ---
    graph.add_edge(START, "session_load")
    graph.add_edge("session_load", "planner")
    graph.add_edge("planner", "approval_wait")
    graph.add_conditional_edges(
        "approval_wait",
        route_after_approval,
        {
            "tool_execute": "tool_execute",
            "__end__": END,
        },
    )
    graph.add_edge("tool_execute", "synthesis")
    graph.add_edge("synthesis", "persist")
    graph.add_edge("persist", END)

    # --- 컴파일 ---
    saver = checkpointer or MemorySaver()
    compiled = graph.compile(checkpointer=saver)

    return compiled
```

---

## 5. Planner Adapter Interface

### 5.1 파일 위치

`src/inference/graph/planner_adapter.py` (신규 생성)

### 5.2 인터페이스 정의

```python
"""Planner adapter: 사용자 요청을 구조화된 실행 계획으로 변환."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from langchain_core.messages import AnyMessage

from .state import TaskType, ToolPlan


class PlannerAdapter(ABC):
    """Planner 추상 인터페이스.

    모든 planner 구현체는 이 인터페이스를 따른다.
    """

    @abstractmethod
    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        """사용자 메시지와 컨텍스트를 받아 실행 계획을 반환한다.

        Parameters
        ----------
        messages : Sequence[AnyMessage]
            LangGraph state의 message history.
        context : Dict[str, Any]
            accumulated_context (세션 요약, 이전 tool 결과 등).

        Returns
        -------
        ToolPlan
            task_type, goal, reason, tools를 포함한 구조화된 계획.
        """
        ...
```

### 5.3 MVP 구현체: LLMPlannerAdapter

`LLMPlannerAdapter`는 **LGAI-EXAONE/EXAONE-4.0-32B-AWQ**의 네이티브 tool calling을
vLLM OpenAI-compatible endpoint를 통해 활용하는 기본 planner다.

EXAONE 4.0은 tool calling을 네이티브로 지원한다 (BFCL 65.2).
vLLM은 `--enable-auto-tool-choice --tool-call-parser hermes` 옵션으로 서빙된다.
`LLMPlannerAdapter`는 `ChatOpenAI(base_url=..., model=<model_path>)`로 이 endpoint에 연결한다.

per-request LoRA 매핑:
- `draft_response` task → vLLM에 `civil-adapter` 모델 이름으로 요청
- evidence augmentation follow-up → vLLM에 `legal-adapter` 모델 이름으로 요청
- 나머지 (`rag_search`, `api_lookup`, `synthesis`) → base model 사용

```python
class LLMPlannerAdapter(PlannerAdapter):
    """EXAONE 4.0-32B-AWQ 네이티브 tool calling 기반 planner.

    vLLM OpenAI-compatible endpoint (--enable-auto-tool-choice --tool-call-parser hermes)를
    통해 LGAI-EXAONE/EXAONE-4.0-32B-AWQ의 tool calling 능력을 활용한다.
    langchain-openai ChatOpenAI를 사용하여 ToolPlan을 생성한다.
    """

    AVAILABLE_TOOLS = ["rag_search", "api_lookup", "draft_response", "synthesis"]

    SYSTEM_PROMPT = (
        "당신은 GovOn 민원 답변 보조 시스템의 작업 계획기입니다.\n"
        "사용자의 요청을 분석하여 다음 JSON 형식으로 실행 계획을 출력하세요:\n\n"
        '{"task_type": "<draft_response|revise_response|lookup_stats>",\n'
        ' "goal": "<사용자에게 보여줄 작업 설명 (한국어, 1-2문장)>",\n'
        ' "reason": "<이 작업이 필요한 이유 (한국어, 1문장)>",\n'
        ' "tools": ["<tool1>", "<tool2>", ...]}\n\n'
        f"사용 가능한 도구: {AVAILABLE_TOOLS}\n"
        "규칙:\n"
        "- draft_response: rag_search, api_lookup, draft_response 순서\n"
        "- revise_response: rag_search, api_lookup, draft_response 순서\n"
        "- lookup_stats: api_lookup 단독\n"
        "- JSON만 출력하세요. 다른 텍스트 없이.\n"
    )

    def __init__(self, llm) -> None:
        """
        Parameters
        ----------
        llm : BaseChatModel
            langchain-openai ChatOpenAI 또는 호환 LLM.
            로컬 vLLM을 OpenAI-compatible endpoint로 연결 가능.
        """
        self._llm = llm

    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        from langchain_core.messages import HumanMessage, SystemMessage
        import json

        plan_messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_prompt(messages, context)),
        ]

        response = await self._llm.ainvoke(plan_messages)
        parsed = json.loads(response.content)

        return ToolPlan(
            task_type=TaskType(parsed["task_type"]),
            goal=parsed["goal"],
            reason=parsed["reason"],
            tools=parsed["tools"],
        )

    @staticmethod
    def _build_user_prompt(
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> str:
        parts = []
        if context.get("session_context"):
            parts.append(f"[세션 맥락]\n{context['session_context']}")
        user_query = messages[-1].content if messages else ""
        parts.append(f"[사용자 요청]\n{user_query}")
        return "\n\n".join(parts)
```

### 5.4 CI Fallback: RegexPlannerAdapter

운영 환경의 기본 planner는 **`LLMPlannerAdapter`** 다.
`RegexPlannerAdapter`는 CI/테스트 환경(`SKIP_MODEL_LOAD=true`)에서 LLM 없이
graph가 동작하도록 보장하는 fallback 전용 구현체다.
운영 배포에서 정규식 기반 라우팅은 사용하지 않는다.

```python
class RegexPlannerAdapter(PlannerAdapter):
    """기존 정규식 ToolRouter를 PlannerAdapter 인터페이스로 래핑.

    CI fallback 전용: SKIP_MODEL_LOAD=true 환경에서 LLM 없이 graph를 실행한다.
    운영 환경에서는 LLMPlannerAdapter가 도구를 선택한다.
    """

    def __init__(self) -> None:
        from src.inference.tool_router import ToolRouter
        self._router = ToolRouter()

    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        query = messages[-1].content if messages else ""
        has_context = bool(context.get("session_context"))

        execution_plan = self._router.plan(query, has_context=has_context)

        # ToolRouter의 결과를 ToolPlan으로 변환
        task_type = self._infer_task_type(execution_plan.tool_names)

        return ToolPlan(
            task_type=task_type,
            goal=f"요청 처리: {execution_plan.reason}",
            reason=execution_plan.reason,
            tools=execution_plan.tool_names,
        )

    @staticmethod
    def _infer_task_type(tool_names: list[str]) -> TaskType:
        if "draft_response" in tool_names:
            return TaskType.DRAFT_RESPONSE
        if tool_names == ["api_lookup"]:
            return TaskType.LOOKUP_STATS
        return TaskType.DRAFT_RESPONSE
```

---

## 6. Executor Adapter Interface

### 6.1 파일 위치

`src/inference/graph/executor_adapter.py` (신규 생성)

### 6.2 인터페이스 정의

```python
"""Executor adapter: tool registry에서 tool을 조회하고 실행."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from loguru import logger


# Tool metadata 계약: 각 tool이 registry에 등록될 때 제공해야 하는 정보
TOOL_METADATA_SCHEMA = {
    "name": str,            # tool 식별자 (예: "rag_search")
    "description": str,     # tool 설명 (planner에게 제공)
    "callable": Callable,   # async (query, context, session) -> dict
    "timeout_sec": float,   # 실행 제한 시간 (기본 30.0)
}


class ExecutorAdapter(ABC):
    """Tool executor 추상 인터페이스."""

    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """tool을 실행하고 결과를 반환한다.

        Parameters
        ----------
        tool_name : str
            실행할 tool 이름.
        query : str
            사용자 요청 텍스트.
        context : Dict[str, Any]
            누적 컨텍스트 (이전 tool 결과 포함).

        Returns
        -------
        Dict[str, Any]
            tool 실행 결과. 최소 {"success": bool, ...} 형태.
            실패 시 {"success": False, "error": str}.
        """
        ...

    @abstractmethod
    def list_tools(self) -> list[str]:
        """등록된 tool 이름 목록을 반환한다."""
        ...
```

### 6.3 MVP 구현체: RegistryExecutorAdapter

```python
import asyncio
import time


class RegistryExecutorAdapter(ExecutorAdapter):
    """기존 vLLMEngineManager._init_agent_loop() 의 tool_registry를 재사용하는 executor.

    tool_registry: Dict[str, Callable] 형태로 주입받는다.
    기존 AgentLoop._execute_tool() 로직을 계승한다.
    """

    def __init__(
        self,
        tool_registry: Dict[str, Callable],
        session_store: "SessionStore",
        default_timeout: float = 30.0,
    ) -> None:
        self._tools = tool_registry
        self._session_store = session_store
        self._default_timeout = default_timeout

    async def execute(
        self,
        tool_name: str,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_fn = self._tools.get(tool_name)
        if tool_fn is None:
            return {"success": False, "error": f"등록되지 않은 tool: {tool_name}"}

        session = self._session_store.get_or_create(
            context.get("session_id")
        )
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool_fn(query=query, context=context, session=session),
                timeout=self._default_timeout,
            )
            latency = (time.monotonic() - start) * 1000
            if isinstance(result, dict):
                result["latency_ms"] = latency
                result.setdefault("success", True)
                return result
            return {"success": True, "result": result, "latency_ms": latency}
        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            return {"success": False, "error": f"tool {tool_name} 타임아웃", "latency_ms": latency}
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"[Executor] tool {tool_name} 오류: {exc}", exc_info=True)
            return {"success": False, "error": str(exc), "latency_ms": latency}

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
```

---

## 7. 기존 코드 통합 전략

### 7.1 디렉토리 구조

```
src/inference/
    graph/                          # <-- 신규 패키지
        __init__.py
        state.py                    # GovOnGraphState, TaskType, ApprovalStatus, ToolPlan
        builder.py                  # build_govon_graph(), 노드 함수들, 라우팅
        planner_adapter.py          # PlannerAdapter ABC, LLMPlannerAdapter, RegexPlannerAdapter
        executor_adapter.py         # ExecutorAdapter ABC, RegistryExecutorAdapter
    agent_loop.py                   # 기존 유지 (deprecated, 제거하지 않음)
    tool_router.py                  # 기존 유지 (RegexPlannerAdapter가 래핑)
    session_context.py              # 기존 유지 (graph 노드에서 직접 사용)
    api_server.py                   # 수정: graph 초기화 추가
    actions/                        # 기존 유지
```

### 7.2 AgentLoop / ToolRouter 처리

- **삭제하지 않는다.** 기존 `/v1/agent/run` 및 `/v1/agent/stream` 엔드포인트는 당분간 유지한다.
- 새 엔드포인트 `/v2/agent/run` 및 `/v2/agent/stream`을 추가한다.
- `ToolRouter`는 `RegexPlannerAdapter`에 의해 재사용된다.
- 마이그레이션 완료 후 v1 엔드포인트를 제거하는 것은 별도 이슈로 처리한다.

### 7.3 api_server.py 통합

`vLLMEngineManager`에 graph 초기화 메서드를 추가한다:

```python
# api_server.py에 추가할 코드

class vLLMEngineManager:
    def __init__(self):
        # ... 기존 코드 ...
        self.graph = None           # <-- 추가
        self._init_agent_loop()
        self._init_graph()          # <-- 추가

    def _init_graph(self) -> None:
        """LangGraph StateGraph를 초기화한다."""
        from src.inference.graph.builder import build_govon_graph
        from src.inference.graph.planner_adapter import LLMPlannerAdapter
        from src.inference.graph.executor_adapter import RegistryExecutorAdapter

        # 기존 _init_agent_loop에서 생성한 tool_registry를 재사용
        tool_registry = self._build_tool_registry()

        # 운영: LLMPlannerAdapter (CI fallback: SKIP_MODEL_LOAD=true 시 RegexPlannerAdapter)
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY",
                         model=MODEL_PATH, temperature=0.0)
        planner = LLMPlannerAdapter(llm=llm, registry=tool_registry)
        executor = RegistryExecutorAdapter(
            tool_registry=tool_registry,
            session_store=self.session_store,
        )

        # SqliteSaver: GovOn 세션 DB와 별도 파일 사용
        import os
        from pathlib import Path
        checkpoint_db = str(
            Path(os.getenv("GOVON_HOME", Path.home() / ".govon")) / "graph_checkpoints.sqlite3"
        )

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = AsyncSqliteSaver.from_conn_string(checkpoint_db)

        self.graph = build_govon_graph(
            planner_adapter=planner,
            executor_adapter=executor,
            session_store=self.session_store,
            checkpointer=checkpointer,
        )

    def _build_tool_registry(self) -> Dict[str, Callable]:
        """tool registry를 dict[str, Callable]로 반환.

        기존 _init_agent_loop의 tool 정의를 추출하여 재사용한다.
        """
        # 기존 _init_agent_loop 내부의 _rag_search_tool, _api_lookup_tool 등을
        # 이 메서드로 추출한다. tool_name을 str key로 사용.
        # (구현 에이전트가 기존 _init_agent_loop에서 함수들을 추출하여 리팩터링)
        ...
```

### 7.4 새 FastAPI 엔드포인트

```python
# api_server.py에 추가

@app.post("/v2/agent/run")
async def v2_agent_run(
    request: AgentRunRequest,
    _: str = Depends(verify_api_key),
):
    """LangGraph 기반 agent 실행.

    1단계: graph.invoke()로 planner까지 실행
    2단계: interrupt()에서 멈춤 -> approval_request 반환
    3단계: /v2/agent/approve 에서 resume
    """
    if not manager.graph:
        raise HTTPException(status_code=503, detail="Graph 미초기화")

    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": request.session_id or str(uuid.uuid4())}}
    initial_state = {
        "session_id": request.session_id,
        "request_id": str(uuid.uuid4()),
        "messages": [HumanMessage(content=request.query)],
    }

    # graph.invoke()는 interrupt()에서 멈추고 중간 상태를 반환
    result = await manager.graph.ainvoke(initial_state, config=config)

    # interrupt 상태 확인
    graph_state = await manager.graph.aget_state(config)
    if graph_state.next:  # interrupt 대기 중
        # approval_request 정보를 클라이언트에 반환
        return {
            "status": "awaiting_approval",
            "thread_id": config["configurable"]["thread_id"],
            "approval_request": graph_state.tasks[0].interrupts[0].value,
        }

    # interrupt 없이 완료된 경우 (rejected 등)
    return {
        "status": "completed",
        "thread_id": config["configurable"]["thread_id"],
        "text": result.get("final_text", ""),
    }


@app.post("/v2/agent/approve")
async def v2_agent_approve(
    thread_id: str,
    approved: bool,
    _: str = Depends(verify_api_key),
):
    """interrupt된 graph를 resume한다."""
    if not manager.graph:
        raise HTTPException(status_code=503, detail="Graph 미초기화")

    config = {"configurable": {"thread_id": thread_id}}

    from langgraph.types import Command

    # resume: interrupt()의 반환값으로 사용자 응답을 전달
    result = await manager.graph.ainvoke(
        Command(resume={"approved": approved}),
        config=config,
    )

    return {
        "status": "completed",
        "text": result.get("final_text", ""),
        "tool_results": result.get("tool_results", {}),
    }
```

### 7.5 Checkpoint 전략

| 항목 | 결정 |
|---|---|
| Checkpoint 저장소 | `AsyncSqliteSaver` (별도 파일 `~/.govon/graph_checkpoints.sqlite3`) |
| GovOn 세션 DB | 기존 `~/.govon/sessions.sqlite3` 유지 (분리) |
| 이유 | LangGraph checkpoint 스키마와 GovOn 세션 스키마가 다르다. 통합은 복잡성 대비 이득이 적다. 세션 데이터(대화, tool log)는 `persist` 노드에서 기존 `SessionStore`를 통해 저장한다. |
| Trade-off | DB 파일이 2개가 되지만, 각각의 lifecycle이 독립적이라 관리가 단순하다. |

---

## 8. Synthesis 로직 상세

기존 `AgentLoop._extract_final_text()`를 독립 함수로 추출한다.

### 파일 위치

`src/inference/graph/builder.py` 내부 private 함수

```python
def _extract_final_text(accumulated: Dict[str, Any], task_type: str) -> str:
    """tool 결과를 종합하여 최종 텍스트를 생성.

    기존 AgentLoop._extract_final_text()를 계승하되,
    task_type을 기반으로 분기한다.
    """
    # 1. draft_response의 직접 텍스트가 있으면 사용
    for key in ("draft_response",):
        payload = accumulated.get(key, {})
        if isinstance(payload, dict) and payload.get("text"):
            return str(payload["text"])

    # 2. planned_tools 순서대로 텍스트 탐색
    for key, payload in accumulated.items():
        if isinstance(payload, dict) and payload.get("text"):
            return str(payload["text"])

    # 3. 개별 결과 조합
    parts: list[str] = []

    rag_data = accumulated.get("rag_search", {})
    if rag_data.get("results"):
        lines = ["[로컬 문서 근거]"]
        for item in rag_data["results"][:3]:
            title = item.get("title", "")
            content = str(item.get("content", ""))[:120]
            lines.append(f"- {title}: {content}")
        parts.append("\n".join(lines))

    api_data = accumulated.get("api_lookup", {})
    if api_data.get("context_text"):
        parts.append(api_data["context_text"])

    return "\n\n".join(parts) if parts else "요청을 처리할 수 없습니다."
```

---

## 9. 전체 파일 목록 및 생성/수정 계획

| 파일 | 작업 | 설명 |
|---|---|---|
| `src/inference/graph/__init__.py` | 신규 | 패키지 init. 주요 클래스 re-export. |
| `src/inference/graph/state.py` | 신규 | `GovOnGraphState`, `TaskType`, `ApprovalStatus`, `ToolPlan` |
| `src/inference/graph/builder.py` | 신규 | `build_govon_graph()`, 6개 노드 함수, `route_after_approval()`, `_extract_final_text()` |
| `src/inference/graph/planner_adapter.py` | 신규 | `PlannerAdapter` ABC, `LLMPlannerAdapter`, `RegexPlannerAdapter` |
| `src/inference/graph/executor_adapter.py` | 신규 | `ExecutorAdapter` ABC, `RegistryExecutorAdapter` |
| `src/inference/api_server.py` | 수정 | `_init_graph()`, `_build_tool_registry()`, `/v2/agent/run`, `/v2/agent/approve` 추가 |
| `requirements.txt` | 수정 | langgraph 버전 고정, checkpoint 패키지 추가 |
| `tests/test_graph_smoke.py` | 신규 | Smoke test |

---

## 10. Smoke Test 계획

### 10.1 파일 위치

`tests/test_graph_smoke.py` (신규 생성)

### 10.2 테스트 시나리오

```python
"""LangGraph graph 초기화 및 기본 사이클 smoke test.

SKIP_MODEL_LOAD=true 환경에서 LLM 없이 실행 가능해야 한다.
"""

import asyncio
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from src.inference.graph.state import GovOnGraphState, ApprovalStatus
from src.inference.graph.builder import build_govon_graph
from src.inference.graph.planner_adapter import RegexPlannerAdapter
from src.inference.graph.executor_adapter import ExecutorAdapter
from src.inference.session_context import SessionStore


class StubExecutorAdapter(ExecutorAdapter):
    """테스트용 스텁. 모든 tool 호출에 고정 결과를 반환."""

    async def execute(self, tool_name, query, context):
        return {
            "success": True,
            "text": f"[stub] {tool_name} result for: {query}",
            "latency_ms": 1.0,
        }

    def list_tools(self):
        return ["rag_search", "api_lookup", "draft_response", "synthesis"]


@pytest.fixture
def graph():
    """CI fallback: RegexPlannerAdapter + StubExecutor로 graph를 구성.
    실제 운영은 LLMPlannerAdapter를 사용한다."""
    planner = RegexPlannerAdapter()  # CI fallback: 실제 운영은 LLMPlannerAdapter
    executor = StubExecutorAdapter()
    store = SessionStore(db_path=":memory:")
    return build_govon_graph(
        planner_adapter=planner,
        executor_adapter=executor,
        session_store=store,
        checkpointer=MemorySaver(),
    )


class TestGraphSmoke:
    """Graph 초기화 smoke test."""

    def test_graph_compiles(self, graph):
        """graph가 에러 없이 컴파일된다."""
        assert graph is not None

    def test_graph_has_expected_nodes(self, graph):
        """graph에 6개 노드가 존재한다."""
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"session_load", "planner", "approval_wait", "tool_execute", "synthesis", "persist"}
        # START와 END는 별도이므로 expected가 node_names의 부분집합인지 확인
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    @pytest.mark.asyncio
    async def test_graph_runs_to_approval_interrupt(self, graph):
        """graph가 approval_wait에서 interrupt된다."""
        config = {"configurable": {"thread_id": "smoke-test-1"}}
        initial = {
            "session_id": "test-session",
            "request_id": "test-request",
            "messages": [HumanMessage(content="이 민원에 대한 답변 초안 작성해줘")],
        }
        result = await graph.ainvoke(initial, config=config)

        # interrupt 상태 확인
        state = await graph.aget_state(config)
        assert state.next, "Graph should be interrupted at approval_wait"

    @pytest.mark.asyncio
    async def test_graph_completes_after_approval(self, graph):
        """승인 후 graph가 끝까지 실행된다."""
        from langgraph.types import Command

        config = {"configurable": {"thread_id": "smoke-test-2"}}
        initial = {
            "session_id": "test-session-2",
            "request_id": "test-request-2",
            "messages": [HumanMessage(content="답변 초안 작성해줘")],
        }

        # 1단계: interrupt까지 실행
        await graph.ainvoke(initial, config=config)

        # 2단계: 승인으로 resume
        result = await graph.ainvoke(
            Command(resume={"approved": True}),
            config=config,
        )

        assert result.get("final_text"), "final_text should be non-empty after approval"
        assert result.get("approval_status") == ApprovalStatus.APPROVED.value

    @pytest.mark.asyncio
    async def test_graph_ends_on_rejection(self, graph):
        """거절 시 graph가 tool_execute 없이 종료된다."""
        from langgraph.types import Command

        config = {"configurable": {"thread_id": "smoke-test-3"}}
        initial = {
            "session_id": "test-session-3",
            "request_id": "test-request-3",
            "messages": [HumanMessage(content="답변 초안 작성해줘")],
        }

        # 1단계: interrupt까지 실행
        await graph.ainvoke(initial, config=config)

        # 2단계: 거절로 resume
        result = await graph.ainvoke(
            Command(resume={"approved": False}),
            config=config,
        )

        assert result.get("approval_status") == ApprovalStatus.REJECTED.value
        assert not result.get("tool_results"), "No tools should run after rejection"
```

### 10.3 실행 방법

```bash
# 의존성 설치
pip install langgraph==1.1.4 langgraph-checkpoint-sqlite==3.0.3

# smoke test 실행
SKIP_MODEL_LOAD=true pytest tests/test_graph_smoke.py -v
```

---

## 11. 구현 순서 권고

구현 에이전트는 다음 순서로 작업한다:

| 순서 | 작업 | 검증 |
|---|---|---|
| 1 | `requirements.txt` 수정 및 `pip install` | `python -c "from langgraph.graph import StateGraph"` |
| 2 | `src/inference/graph/__init__.py` 생성 | import 성공 |
| 3 | `src/inference/graph/state.py` 생성 | `from src.inference.graph.state import GovOnGraphState` |
| 4 | `src/inference/graph/planner_adapter.py` 생성 | `RegexPlannerAdapter` 단독 테스트 |
| 5 | `src/inference/graph/executor_adapter.py` 생성 | `RegistryExecutorAdapter` 단독 테스트 |
| 6 | `src/inference/graph/builder.py` 생성 | `build_govon_graph()` 호출 성공 |
| 7 | `tests/test_graph_smoke.py` 생성 및 실행 | 4개 테스트 전부 통과 |
| 8 | `src/inference/api_server.py` 수정 | `/v2/agent/run` 호출 가능 |

---

## 12. Architectural Decision Record

### ADR: LangGraph StateGraph 도입

**Status**: Proposed

**Context**: 현재 `AgentLoop` + `ToolRouter`는 정규식 기반으로 tool을 선택한다. 승인 게이트가 없고, 세션 간 graph 상태 복원이 불가능하다. 이슈 #415는 LangGraph 기반 runtime으로 전환하여 planner LLM과 human-in-the-loop 승인을 도입할 것을 요구한다.

**Decision**: LangGraph 1.1.4 `StateGraph`를 도입하되, 기존 `AgentLoop`/`ToolRouter`를 삭제하지 않고 v2 엔드포인트를 병행한다. Planner와 Executor를 adapter 패턴으로 추상화하여 LLM planner와 regex fallback을 교환 가능하게 한다. Checkpoint는 `AsyncSqliteSaver`로 별도 DB 파일에 저장한다.

**Consequences**:
- 장점: `interrupt()` 기반 승인 게이트가 프레임워크 수준에서 지원됨. graph 상태가 자동 직렬화/복원됨. planner 교체가 adapter 인터페이스로 쉬움.
- 단점: langgraph 의존성 추가 (약 30MB). checkpoint DB 파일이 별도로 생성됨. LangGraph API 변경 시 추가 마이그레이션 필요.
- 쉬워지는 것: 승인 게이트, 세션 중간 복원, planner 교체.
- 어려워지는 것: LangGraph 버전 업그레이드 시 호환성, 디버깅 시 graph 내부 상태 추적.
