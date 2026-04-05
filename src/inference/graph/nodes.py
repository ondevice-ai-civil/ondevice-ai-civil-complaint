"""GovOn LangGraph 노드 함수 모음.

Issue #415: LangGraph runtime 기반 및 planner/executor adapter 구성.

6개 노드를 정의한다:
  session_load -> planner -> approval_wait -> tool_execute -> synthesis -> persist

각 노드는 `GovOnGraphState`를 입력으로 받고 상태 업데이트 dict를 반환한다.
I/O가 필요한 노드는 async 함수이며, `approval_wait` 노드는 `interrupt()`를
사용하는 human-in-the-loop 승인 게이트이므로 sync 함수로 유지한다.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from loguru import logger

from src.inference.query_builder import (
    build_query_variants,
    build_runtime_query_context,
    resolve_tool_query,
)

from .plan_validator import PlanValidationError, ToolPlanValidator
from .state import ApprovalStatus, GovOnGraphState

if TYPE_CHECKING:
    from src.inference.session_context import SessionStore

    from .executor_adapter import ExecutorAdapter
    from .planner_adapter import PlannerAdapter


async def session_load_node(
    state: GovOnGraphState,
    *,
    session_store: "SessionStore",
) -> dict:
    """세션 로드 노드.

    SessionStore에서 기존 세션을 불러오거나 새 세션을 생성한다.
    대화 히스토리와 tool 사용 기록을 accumulated_context에 주입한다.

    Parameters
    ----------
    state : GovOnGraphState
        현재 graph state. `session_id`와 `messages`를 읽는다.
    session_store : SessionStore
        graph config에서 closure로 주입되는 세션 저장소.

    Returns
    -------
    dict
        `session_id`와 `accumulated_context`를 갱신한다.
    """
    _start = time.monotonic()

    session_id: str | None = state.get("session_id")
    session = session_store.get_or_create(session_id)

    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    accumulated_context = build_runtime_query_context(session, query)

    _latency_ms = round((time.monotonic() - _start) * 1000, 2)
    logger.debug(
        f"[session_load] session_id={session.session_id} "
        f"query_len={len(query)} latency_ms={_latency_ms}"
    )

    return {
        "session_id": session.session_id,
        "accumulated_context": accumulated_context,
        "node_latencies": {"session_load": _latency_ms},
    }


async def planner_node(
    state: GovOnGraphState,
    *,
    planner_adapter: "PlannerAdapter",
) -> dict:
    """Planner 노드.

    PlannerAdapter를 호출하여 구조화된 실행 계획을 생성한다.
    LLMPlannerAdapter가 기본으로 사용된다. SKIP_MODEL_LOAD=true 환경에서는
    RegexPlannerAdapter가 CI fallback으로 동작한다.

    Parameters
    ----------
    state : GovOnGraphState
        현재 graph state. `messages`와 `accumulated_context`를 읽는다.
    planner_adapter : PlannerAdapter
        graph config에서 closure로 주입되는 planner 어댑터.

    Returns
    -------
    dict
        `task_type`, `goal`, `reason`, `planned_tools`를 갱신한다.
    """
    _start = time.monotonic()

    messages = state.get("messages", [])
    context = state.get("accumulated_context", {})

    plan = await planner_adapter.plan(messages=messages, context=context)

    validator = ToolPlanValidator()
    try:
        validator.validate(plan)
    except PlanValidationError as e:
        _latency_ms = round((time.monotonic() - _start) * 1000, 2)
        logger.warning(f"[planner] validation 실패: {e} latency_ms={_latency_ms}")
        return {
            **validator.make_fallback_plan(e),
            "task_type": "",
            "node_latencies": {"planner": _latency_ms},
        }

    logger.info(
        f"[planner] task_type={plan.task_type.value} "
        f"tools={plan.tools} reason={plan.reason} adapter_mode={plan.adapter_mode}"
    )

    query_variants = build_query_variants(
        context.get("query", ""),
        tool_names=plan.tools,
        context=context,
    )

    _latency_ms = round((time.monotonic() - _start) * 1000, 2)
    logger.debug(f"[planner] latency_ms={_latency_ms}")

    return {
        "task_type": plan.task_type.value,
        "goal": plan.goal,
        "reason": plan.reason,
        "planned_tools": plan.tools,
        "tool_summaries": plan.tool_summaries,
        "adapter_mode": plan.adapter_mode,
        "accumulated_context": {
            **context,
            "query_variants": query_variants,
        },
        "node_latencies": {"planner": _latency_ms},
    }


def approval_wait_node(state: GovOnGraphState) -> dict:
    """Human-in-the-loop 승인 게이트.

    `interrupt()`를 호출하여 graph 실행을 일시 정지한다.
    FastAPI `/v2/agent/approve` 엔드포인트가 사용자 응답을 받아 graph를 resume한다.

    `interrupt()`는 LangGraph가 지원하는 human-in-the-loop 메커니즘이다.
    graph 실행이 멈추고, `Command(resume=...)` 호출로 재개될 때
    `interrupt()`의 반환값으로 사용자 입력이 전달된다.

    Parameters
    ----------
    state : GovOnGraphState
        현재 graph state. `goal`, `reason`, `planned_tools`를 읽는다.

    Returns
    -------
    dict
        `approval_status`를 갱신한다.
    """
    tool_summaries: List[str] = state.get("tool_summaries") or []
    planned_tools: List[str] = state.get("planned_tools", [])

    # CLI 표시용 도구 설명: tool_summaries가 있으면 사용, 없으면 tool 이름 그대로
    display_tools = tool_summaries if tool_summaries else planned_tools

    approval_request = {
        "type": "approval_request",
        "goal": state.get("goal", ""),
        "reason": state.get("reason", ""),
        "planned_tools": planned_tools,
        "tool_summaries": display_tools,
        "prompt": (
            f"다음 작업을 수행하겠습니다:\n\n"
            f"  {state.get('goal', '')}\n\n"
            f"  이유: {state.get('reason', '')}\n"
            f"  사용할 도구:\n"
            + "".join(f"    - {s}\n" for s in display_tools)
            + "\n승인하시겠습니까? (승인/거절)"
        ),
    }

    logger.info(f"[approval_wait] interrupt 호출: tools={planned_tools}")

    # interrupt()는 graph 실행을 멈추고, resume 시 반환값이 된다.
    # 예: {"approved": True} 또는 {"approved": False}
    user_response = interrupt(approval_request)

    if isinstance(user_response, dict) and user_response.get("approved"):
        logger.info("[approval_wait] 승인됨")
        return {"approval_status": ApprovalStatus.APPROVED.value}

    # cancel 신호가 있으면 interrupt_reason을 "user_cancel"로 설정
    interrupt_reason = None
    if isinstance(user_response, dict) and user_response.get("cancel"):
        logger.info("[approval_wait] 사용자 취소 (cancel)")
        interrupt_reason = "user_cancel"
    else:
        logger.info("[approval_wait] 거절됨")

    return {
        "approval_status": ApprovalStatus.REJECTED.value,
        "interrupt_reason": interrupt_reason,
    }


async def tool_execute_node(
    state: GovOnGraphState,
    *,
    executor_adapter: "ExecutorAdapter",
) -> dict:
    """Tool executor 노드.

    ExecutorAdapter를 통해 `planned_tools`를 두 단계로 실행하고
    결과를 `accumulated_context`에 누적한다.

    실행 전략:
    - Phase 1 (병렬): `rag_search`, `api_lookup` 등 INDEPENDENT_TOOLS는
      `asyncio.gather()`로 동시에 실행한다.
    - Phase 2 (순차): 나머지 의존 도구(draft_civil_response 등)는 Phase 1
      결과가 누적된 accumulated_context를 사용하여 순서대로 실행한다.

    Parameters
    ----------
    state : GovOnGraphState
        현재 graph state. `planned_tools`, `accumulated_context`를 읽는다.
    executor_adapter : ExecutorAdapter
        graph config에서 closure로 주입되는 executor 어댑터.

    Returns
    -------
    dict
        `tool_results`, `accumulated_context`, `node_latencies`를 갱신한다.
        `node_latencies`에는 `"tool_execute"` 키로 노드 전체 레이턴시,
        `"tool:<tool_name>"` 키로 개별 도구 레이턴시가 포함된다.
    """
    _start = time.monotonic()

    # approval guard: 승인 없이 tool 실행 차단
    approval_status = state.get("approval_status", "")
    if approval_status != ApprovalStatus.APPROVED.value:
        _latency_ms = round((time.monotonic() - _start) * 1000, 2)
        logger.warning(
            f"[tool_execute] 승인되지 않은 상태에서 실행 시도 차단: approval_status={approval_status!r}"
        )
        return {
            "tool_results": {},
            "accumulated_context": dict(state.get("accumulated_context", {})),
            "error": f"tool 실행 차단: 승인 필요 (현재 상태: {approval_status!r})",
            "node_latencies": {"tool_execute": _latency_ms},
        }

    planned_tools: list[str] = state.get("planned_tools", [])
    accumulated: Dict[str, Any] = dict(state.get("accumulated_context", {}))

    # planned_tools가 비어있는 경우 (validation 실패 fallback 등)
    if not planned_tools:
        logger.warning("[tool_execute] planned_tools가 비어있어 실행 건너뜀")
        return {
            "tool_results": {},
            "accumulated_context": accumulated,
            "node_latencies": {"tool_execute": round((time.monotonic() - _start) * 1000, 2)},
        }

    tool_results: Dict[str, Any] = {}
    tool_latencies: Dict[str, float] = {}

    # --- 독립 도구와 의존 도구를 분리하여 병렬/순차 실행 ---
    # 새로운 독립 capability를 추가할 때는 이 집합에도 등록해야 한다.
    # 독립 도구란 다른 도구의 실행 결과(accumulated_context)에 의존하지 않아
    # 병렬 실행이 안전한 capability를 의미한다.
    INDEPENDENT_TOOLS = {"rag_search", "api_lookup"}

    independent = [t for t in planned_tools if t in INDEPENDENT_TOOLS]
    dependent = [t for t in planned_tools if t not in INDEPENDENT_TOOLS]

    # Phase 1: 독립 도구 병렬 실행
    # 주의: _run_tool 클로저 내에서 accumulated는 읽기 전용으로 취급해야 한다.
    # 병렬 실행 중 accumulated를 변경하면 race condition이 발생할 수 있다.
    # accumulated 갱신은 gather() 완료 후 메인 루프에서 순서대로 수행한다.
    if independent:

        async def _run_tool(name: str) -> tuple[str, Dict[str, Any], float]:
            t0 = time.monotonic()
            execution_query = resolve_tool_query(name, accumulated)
            logger.info(f"[tool_execute] 병렬 실행: {name}")
            result = await executor_adapter.execute(
                tool_name=name,
                query=execution_query,
                context=dict(accumulated),
            )
            latency = round((time.monotonic() - t0) * 1000, 2)
            return name, result, latency

        results = await asyncio.gather(
            *[_run_tool(name) for name in independent],
            return_exceptions=True,
        )
        for i, item in enumerate(results):
            if isinstance(item, Exception):
                failed_tool = independent[i]
                logger.opt(exception=item).error(
                    f"[tool_execute] 병렬 실행 실패: tool={failed_tool}"
                )
                continue
            name, result, latency = item
            tool_results[name] = result
            tool_latencies[name] = latency
            if result.get("success", True):
                accumulated[name] = result

    # Phase 2: 의존 도구 순차 실행 (accumulated context 필요)
    for name in dependent:
        t0 = time.monotonic()
        execution_query = resolve_tool_query(name, accumulated)
        logger.info(f"[tool_execute] 순차 실행: {name}")
        try:
            result = await executor_adapter.execute(
                tool_name=name,
                query=execution_query,
                context=accumulated,
            )
        except Exception as exc:
            latency = round((time.monotonic() - t0) * 1000, 2)
            logger.opt(exception=exc).error(f"[tool_execute] 순차 실행 실패: tool={name}")
            tool_results[name] = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            tool_latencies[name] = latency
            continue
        latency = round((time.monotonic() - t0) * 1000, 2)
        tool_results[name] = result
        tool_latencies[name] = latency
        if result.get("success", True):
            accumulated[name] = result

    _latency_ms = round((time.monotonic() - _start) * 1000, 2)
    logger.info(
        f"[tool_execute] 완료: {list(tool_results.keys())} "
        f"latency_ms={_latency_ms} per_tool={tool_latencies}"
    )

    # node_latencies에 노드 전체 레이턴시와 개별 tool 레이턴시를 함께 기록한다.
    # tool 레이턴시는 "tool:<tool_name>" 접두사로 구분한다.
    merged_latencies: Dict[str, float] = {"tool_execute": _latency_ms}
    for tool_name, tool_lat in tool_latencies.items():
        merged_latencies[f"tool:{tool_name}"] = tool_lat

    return {
        "tool_results": tool_results,
        "accumulated_context": accumulated,
        "node_latencies": merged_latencies,
    }


async def synthesis_node(state: GovOnGraphState) -> dict:
    """결과 종합 노드.

    tool_results와 accumulated_context를 종합하여 최종 응답 텍스트를 생성한다.
    기존 AgentLoop._extract_final_text() 로직을 계승한다.

    Parameters
    ----------
    state : GovOnGraphState
        현재 graph state. `tool_results`, `accumulated_context`, `task_type`을 읽는다.

    Returns
    -------
    dict
        `final_text`와 `messages`(AIMessage 추가)를 갱신한다.
    """
    _start = time.monotonic()

    accumulated = state.get("accumulated_context", {})
    task_type = state.get("task_type", "")

    final_text = _extract_final_text(accumulated, task_type)

    _latency_ms = round((time.monotonic() - _start) * 1000, 2)
    logger.info(f"[synthesis] final_text_len={len(final_text)} latency_ms={_latency_ms}")

    return {
        "final_text": final_text,
        "messages": [AIMessage(content=final_text)],
        "node_latencies": {"synthesis": _latency_ms},
    }


async def persist_node(
    state: GovOnGraphState,
    *,
    session_store: "SessionStore",
) -> dict:
    """영속화 노드.

    SessionStore에 대화 턴과 tool 실행 기록을 저장한다.
    기존 SessionContext.add_turn / add_tool_run 로직을 계승한다.

    Parameters
    ----------
    state : GovOnGraphState
        현재 graph state. 전체 state를 읽어 저장한다.
    session_store : SessionStore
        graph config에서 closure로 주입되는 세션 저장소.

    Returns
    -------
    dict
        side effect로 DB에 저장하고, `node_latencies` 업데이트를 반환한다.
    """
    _start = time.monotonic()

    session_id: str | None = state.get("session_id")
    session = session_store.get_or_create(session_id)

    # 사용자 입력 저장 (messages[0]이 최초 사용자 메시지)
    messages = state.get("messages", [])
    if messages:
        user_msg = messages[0]
        session.add_turn("user", user_msg.content)

    # --- graph_run 기록 (plan + approval + executed capabilities) ---
    request_id: str = state.get("request_id", "")
    approval_status: str = state.get("approval_status", "")
    planned_tools: List[str] = state.get("planned_tools", [])
    tool_results: Dict[str, Any] = state.get("tool_results", {})

    # 승인된 경우에만 실행된 도구 목록을 기록, 거절 시 빈 리스트
    executed_capabilities: List[str] = (
        [name for name in planned_tools if name in tool_results]
        if approval_status == ApprovalStatus.APPROVED.value
        else []
    )

    plan_summary = (
        f"[{state.get('task_type', '')}] {state.get('goal', '')} "
        f"| 이유: {state.get('reason', '')} | tools: {planned_tools}"
    )

    total_latency_ms = sum(r.get("latency_ms", 0.0) for r in tool_results.values())

    # interrupt_reason이 있으면 "interrupted", 거절이면 "rejected", 그 외 "completed"
    interrupt_reason: str | None = state.get("interrupt_reason")
    if interrupt_reason:
        graph_status = "interrupted"
    elif approval_status == ApprovalStatus.REJECTED.value:
        graph_status = "rejected"
    else:
        graph_status = "completed"

    session.add_graph_run(
        request_id=request_id,
        plan_summary=plan_summary,
        approval_status=approval_status,
        executed_capabilities=executed_capabilities,
        status=graph_status,
        total_latency_ms=total_latency_ms,
    )

    # tool 실행 기록 저장 (graph_run_request_id로 연결)
    for name, result in tool_results.items():
        session.add_tool_run(
            tool=name,
            success=result.get("success", True),
            graph_run_request_id=request_id,
            latency_ms=result.get("latency_ms", 0.0),
            error=result.get("error"),
        )

    # 어시스턴트 응답 저장
    final_text = state.get("final_text", "")
    if final_text:
        session.add_turn("assistant", final_text)

    _latency_ms = round((time.monotonic() - _start) * 1000, 2)
    logger.debug(
        f"[persist] session_id={session.session_id} "
        f"graph_run={request_id} saved latency_ms={_latency_ms}"
    )

    return {"node_latencies": {"persist": _latency_ms}}


def _extract_final_text(accumulated: Dict[str, Any], task_type: str) -> str:
    """tool 결과를 종합하여 최종 텍스트를 생성한다.

    기존 AgentLoop._extract_final_text()를 계승하되,
    task_type을 기반으로 분기한다.

    Parameters
    ----------
    accumulated : Dict[str, Any]
        tool 결과가 누적된 컨텍스트 dict.
    task_type : str
        TaskType.value (예: "draft_response").

    Returns
    -------
    str
        최종 응답 텍스트.
    """
    # 1. append_evidence 또는 draft_civil_response의 직접 텍스트가 있으면 사용
    for key in ("append_evidence", "draft_civil_response"):
        payload = accumulated.get(key, {})
        if isinstance(payload, dict) and payload.get("text"):
            return str(payload["text"])

    # 2. 모든 accumulated 결과에서 텍스트 탐색
    for key, payload in accumulated.items():
        if key in ("session_context", "query"):
            continue
        if isinstance(payload, dict) and payload.get("text"):
            return str(payload["text"])

    # 3. 개별 결과 조합 — evidence 필드가 있으면 우선 사용
    parts: list[str] = []

    # evidence 필드 기반 출처 목록 생성 (source-specific branching 없이)
    all_evidence_items: list[dict] = []
    for key, payload in accumulated.items():
        if key in ("session_context", "query"):
            continue
        if isinstance(payload, dict):
            ev = payload.get("evidence")
            if isinstance(ev, dict) and ev.get("items"):
                all_evidence_items.extend(ev["items"])

    if all_evidence_items:
        lines = ["[참조 근거]"]
        for item in all_evidence_items[:5]:
            source_type = item.get("source_type", "")
            title = item.get("title", "")
            excerpt = item.get("excerpt", "")[:120]
            label = (
                "[로컬]" if source_type == "rag" else "[외부]" if source_type == "api" else "[생성]"
            )
            if title:
                lines.append(f"- {label} {title}: {excerpt}")
            elif excerpt:
                lines.append(f"- {label} {excerpt}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    # evidence가 없는 경우 legacy fallback
    if not parts:
        rag_data = accumulated.get("rag_search", {})
        if isinstance(rag_data, dict) and rag_data.get("results"):
            lines = ["[로컬 문서 근거]"]
            for item in rag_data["results"][:3]:
                title = item.get("title", "")
                content = str(item.get("content", ""))[:120]
                lines.append(f"- {title}: {content}")
            parts.append("\n".join(lines))

        api_data = accumulated.get("api_lookup", {})
        if isinstance(api_data, dict) and api_data.get("context_text"):
            parts.append(api_data["context_text"])

    return "\n\n".join(parts) if parts else "요청을 처리할 수 없습니다."
