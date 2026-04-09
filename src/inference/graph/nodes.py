"""GovOn LangGraph 노드 함수 모음.

v4 아키텍처: ReAct + ToolNode 기반.
LLM이 자율적으로 도구 호출을 결정하며, 정적 planner/executor/synthesis를 제거한다.

노드 구성:
  session_load → agent → approval_wait → (ToolNode) → persist

각 노드 팩토리는 의존성을 클로저로 주입받아 노드 함수를 반환한다.
"""

from __future__ import annotations

import json
import time

# v3 전용 import
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.types import interrupt
from loguru import logger

from .state import ApprovalStatus, GovOnGraphState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CONSECUTIVE_REJECTIONS = 2

if TYPE_CHECKING:
    from src.inference.session_context import SessionStore


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


_BASE_SYSTEM_PROMPT = (
    "You are GovOn, a Korean civil complaint response assistant.\n"
    "Analyze the user's request and actively use the available tools to provide accurate, data-backed answers.\n\n"
    "Decision rules:\n"
    "1. For complaint response drafts: ALWAYS call the appropriate adapter tool "
    "(public_admin_adapter for administrative matters, legal_adapter for legal questions).\n"
    "2. For data or statistics: Use api_lookup, stats_lookup, issue_detector, "
    "keyword_analyzer, or demographics_lookup as needed.\n"
    "3. Call MULTIPLE tools in parallel when the query requires different types of information.\n"
    "4. Only respond directly WITHOUT tools for simple greetings or questions requiring no data.\n"
    "5. After collecting tool results, synthesize them into a comprehensive Korean answer.\n"
    "6. If a previous turn already contains a draft, you may add legal citations "
    "or data without re-calling the adapter.\n"
)

_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# session_load
# ---------------------------------------------------------------------------


def make_session_load_node(session_store: "SessionStore"):
    """세션 로드 + 컨텍스트 윈도우 관리 노드 팩토리.

    멀티턴 대화에서 messages가 누적되면 LLM 컨텍스트 윈도우(max_model_len=8192)를
    초과할 수 있다. session_load에서 토큰 예산 기반으로 오래된 메시지를 정리한다.

    EXAONE-4.0-32B-AWQ 스펙: max_position_embeddings=131072
    현재 vLLM 설정: --max-model-len=8192 (GPU 메모리 제약)

    토큰 예산 분배 (max_model_len=8192):
      시스템 프롬프트:     ~500 토큰
      도구 스키마 (7개):   ~1,000 토큰
      대화 이력 (messages): ~4,000 토큰
      LLM 응답 (max_tokens): ~2,048 토큰
      안전 마진:           ~644 토큰
    """

    # 메시지에 할당 가능한 토큰 예산
    # max_model_len(8192) - max_tokens(2048) - overhead(1500) - margin(644) = 4000
    MAX_MESSAGE_TOKENS = 4000
    # 최근 N개 메시지는 무조건 보존 (현재 턴의 맥락 유지)
    KEEP_RECENT = 6

    def _estimate_tokens(msg: Any) -> int:
        """메시지의 대략적 토큰 수 추정.

        한국어: 1글자 ≈ 2-3 토큰 (보수적으로 2 적용)
        영어/JSON: 1문자 ≈ 0.75 토큰
        ToolMessage는 JSON이 많으므로 별도 계수.
        """
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
            # ToolMessage는 JSON(영어/숫자 위주), 나머지는 한국어 위주
            if getattr(msg, "type", "") == "tool":
                return max(len(content), 10)  # JSON: ~1 token/char
            return max(len(content) * 2, 10)  # 한국어: ~2 tokens/char
        return 100

    async def session_load_node(state: GovOnGraphState) -> dict:
        t0 = time.monotonic()
        session_id = state.get("session_id", "")
        messages = list(state.get("messages", []))

        result: Dict[str, Any] = {}

        # 메시지가 적으면 관리 불필요
        if len(messages) > KEEP_RECENT:
            recent = messages[-KEEP_RECENT:]
            older = messages[:-KEEP_RECENT]

            recent_tokens = sum(_estimate_tokens(m) for m in recent)
            remaining_budget = MAX_MESSAGE_TOKENS - recent_tokens

            if remaining_budget <= 0:
                trimmed = recent
            else:
                kept_older: List = []
                for msg in reversed(older):
                    tokens = _estimate_tokens(msg)
                    if remaining_budget - tokens < 0:
                        break
                    kept_older.insert(0, msg)
                    remaining_budget -= tokens
                trimmed = kept_older + recent

            if len(trimmed) < len(messages):
                logger.info(
                    f"[session_load] context window trim: "
                    f"{len(messages)} → {len(trimmed)} messages"
                )
                # RemoveMessage로 오래된 메시지를 state에서 제거
                from langchain_core.messages import RemoveMessage

                trimmed_ids = {getattr(m, "id", None) for m in trimmed} - {None}
                removals = []
                for msg in messages:
                    msg_id = getattr(msg, "id", None)
                    if msg_id and msg_id not in trimmed_ids:
                        removals.append(RemoveMessage(id=msg_id))
                if removals:
                    result["messages"] = removals

        latency = round((time.monotonic() - t0) * 1000, 2)
        logger.debug(f"[session_load] session_id={session_id} latency_ms={latency}")
        result["node_latencies"] = {"session_load": latency}
        return result

    return session_load_node


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------


def make_agent_node(llm, tools: list):
    """ReAct agent 노드 팩토리.

    LLM에 도구를 바인딩하고, 매 호출마다 system prompt + messages를
    전달하여 도구 호출 또는 최종 응답을 생성한다.

    연속 거부 감지 시 도구 바인딩 없이 LLM을 호출하여
    tool_calls가 포함되지 않는 순수 텍스트 응답을 보장한다.
    """
    llm_without_tools = llm  # 원본 LLM (도구 바인딩 없음)
    llm_with_tools = llm.bind_tools(tools)

    def _count_consecutive_rejections(messages: list) -> int:
        """messages를 역순 순회하여 연속 거부 피드백 메시지 수를 카운트한다.

        거부 사이클: AIMessage(tool_calls) → 거부 HumanMessage 가 반복된다.
        tool_calls가 있는 AIMessage는 거부 사이클의 일부이므로 건너뛰고,
        tool_calls가 없는 AIMessage(최종 응답)를 만나면 이전 사이클이므로 중단한다.
        """
        count = 0
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and "사용자가 도구 실행을 거부했습니다" in (
                msg.content or ""
            ):
                count += 1
            elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                # tool_calls 없는 AI 응답 = 이전 사이클 완료, 중단
                break
            # tool_calls 있는 AIMessage는 거부 사이클의 일부 → 건너뜀
        return count

    async def agent_node(state: GovOnGraphState) -> dict:
        t0 = time.monotonic()
        messages = state.get("messages", [])

        # 연속 거부 횟수에 따라 system prompt에 도구 호출 금지 힌트 추가
        rejection_count = _count_consecutive_rejections(messages)
        system_content = _SYSTEM_PROMPT
        if rejection_count >= _MAX_CONSECUTIVE_REJECTIONS:
            system_content += (
                "\n\n[중요] 사용자가 도구 실행을 연속으로 거부했습니다. "
                "더 이상 도구를 호출하지 말고, 현재 가진 정보만으로 직접 답변하세요."
            )
            logger.warning(
                f"[agent] 연속 거부 {rejection_count}회 감지 — 도구 호출 없이 직접 응답 유도"
            )

        system = SystemMessage(content=system_content)

        # 연속 거부 시 도구 바인딩 없이 호출하여 순수 텍스트 응답 보장
        active_llm = (
            llm_without_tools if rejection_count >= _MAX_CONSECUTIVE_REJECTIONS else llm_with_tools
        )
        response = await active_llm.ainvoke([system] + list(messages))
        latency = round((time.monotonic() - t0) * 1000, 2)
        logger.debug(
            f"[agent] tool_calls={len(getattr(response, 'tool_calls', []))} "
            f"latency_ms={latency}"
        )
        return {
            "messages": [response],
            "node_latencies": {"agent": latency},
        }

    return agent_node


# ---------------------------------------------------------------------------
# approval_wait
# ---------------------------------------------------------------------------


def make_approval_wait_node(approval_map: dict[str, bool]):
    """Human-in-the-loop 승인 게이트 노드 팩토리.

    interrupt()로 graph 실행을 일시 정지하고 사용자 승인을 대기한다.
    거부 시 HumanMessage를 추가하여 agent가 대안을 제시하도록 한다.
    """

    async def approval_wait_node(state: GovOnGraphState, config: RunnableConfig) -> dict:
        # interrupt()가 내부적으로 get_config()를 호출하므로 ContextVar에 config 주입
        var_child_runnable_config.set(config)

        messages = state.get("messages", [])
        if not messages:
            return {"approval_status": ApprovalStatus.REJECTED.value}

        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", [])
        tool_names = [tc["name"] for tc in tool_calls]

        approval_required = [name for name in tool_names if approval_map.get(name, False)]

        logger.info(
            f"[approval_wait] interrupt: tools={tool_names} "
            f"approval_required={approval_required}"
        )

        response = interrupt(
            {
                "tools": tool_names,
                "planned_tools": tool_names,
                "message": f"다음 도구를 실행합니다: {', '.join(tool_names)}",
                "approval_required": approval_required,
            }
        )

        cancelled = response.get("cancel", False) if isinstance(response, dict) else False
        if cancelled:
            logger.info("[approval_wait] 취소됨")
            return {"approval_status": "cancelled"}

        approved = response.get("approved", False) if isinstance(response, dict) else bool(response)

        if approved:
            logger.info("[approval_wait] 승인됨")
            return {"approval_status": ApprovalStatus.APPROVED.value}

        logger.info("[approval_wait] 거절됨")
        rejection_msg = HumanMessage(
            content="사용자가 도구 실행을 거부했습니다. 다른 방법을 제안하세요."
        )
        return {
            "approval_status": ApprovalStatus.REJECTED.value,
            "messages": [rejection_msg],
        }

    return approval_wait_node


# ---------------------------------------------------------------------------
# route_after_approval
# ---------------------------------------------------------------------------


def route_after_approval(state: GovOnGraphState) -> str:
    """approval_wait 이후 분기.

    승인이면 tools로, 취소이면 persist로, 거부이면 agent로 돌려보내 대안을 제시하게 한다.
    """
    status = state.get("approval_status", "")
    if status == ApprovalStatus.APPROVED.value:
        return "tools"
    if status == "cancelled":
        return "persist"
    return "agent"


# ---------------------------------------------------------------------------
# persist
# ---------------------------------------------------------------------------


def make_persist_node(session_store: "SessionStore"):
    """영속화 노드 팩토리.

    messages에서 최종 텍스트와 evidence를 추출하고 SessionStore에 저장한다.
    """

    async def persist_node(state: GovOnGraphState) -> dict:
        t0 = time.monotonic()
        messages = state.get("messages", [])

        # 최종 텍스트: agent_node_v3가 명시적으로 저장한 경우 우선 사용,
        # 없으면 messages에서 tool_calls 없는 마지막 AIMessage 추출
        final_text = state.get("final_text") or ""
        if not final_text:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    final_text = msg.content or ""
                    break
                elif (
                    hasattr(msg, "type")
                    and msg.type == "ai"
                    and not getattr(msg, "tool_calls", None)
                    and hasattr(msg, "content")
                ):
                    final_text = msg.content
                    break

        # 현재 턴 시작점: 마지막 HumanMessage (거부 메시지 제외) 이후부터
        turn_start = 0
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if (
                hasattr(msg, "type")
                and msg.type == "human"
                and "사용자가 도구 실행을 거부했습니다" not in (msg.content or "")
            ):
                turn_start = i
                break

        # evidence: 현재 턴의 ToolMessage에서만 수집 (stale evidence 방지)
        evidence_items: List[Dict[str, Any]] = []
        for msg in messages[turn_start:]:
            if hasattr(msg, "type") and msg.type == "tool":
                try:
                    data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(data, dict):
                        ev = data.get("evidence", {})
                        if isinstance(ev, dict) and ev.get("items"):
                            evidence_items.extend(ev["items"])
                except (json.JSONDecodeError, TypeError):
                    pass

        # SessionStore에 마지막 턴만 저장 (중복 방지)
        session_id = state.get("session_id", "")
        if session_id and session_store:
            session = session_store.get_or_create(session_id)
            # 역순 순회하여 마지막 Human 쿼리와 AI 응답만 추출
            last_human = None
            last_ai = None
            for msg in reversed(messages):
                if (
                    last_ai is None
                    and hasattr(msg, "type")
                    and msg.type == "ai"
                    and not getattr(msg, "tool_calls", None)
                ):
                    last_ai = msg
                elif (
                    last_human is None
                    and hasattr(msg, "type")
                    and msg.type == "human"
                    and "사용자가 도구 실행을 거부했습니다" not in (msg.content or "")
                ):
                    last_human = msg
                if last_human and last_ai:
                    break
            if last_human:
                session.add_turn("user", last_human.content)
            if last_ai:
                session.add_turn("assistant", last_ai.content)

        latency = round((time.monotonic() - t0) * 1000, 2)
        logger.debug(
            f"[persist] final_text_len={len(final_text)} "
            f"evidence={len(evidence_items)} latency_ms={latency}"
        )

        return {
            "final_text": final_text,
            "evidence_items": evidence_items[:10],
            "node_latencies": {"persist": latency},
        }

    return persist_node


# ---------------------------------------------------------------------------
# v3 ReAct 전용 상수
# ---------------------------------------------------------------------------

_V3_SYSTEM_PROMPT = (
    _BASE_SYSTEM_PROMPT
    + "7. If tool results are insufficient, call additional tools in the next iteration.\n"
)

_V3_FORCE_ANSWER_SUFFIX = (
    "\n\n[IMPORTANT] Maximum iterations reached. "
    "Do NOT call any more tools. Write the best answer you can based on the information collected so far."
)


# ---------------------------------------------------------------------------
# v3 agent 노드
# ---------------------------------------------------------------------------


def make_agent_node_v3(llm, tools: list):
    """v3 ReAct agent 노드 팩토리.

    iteration_count를 추적하고 max_iterations 초과 시 강제 final answer를 생성한다.
    pending_tool_calls를 state에 저장하여 tools_node_v3에서 실행한다.
    """
    llm_without_tools = llm
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node_v3(state: GovOnGraphState) -> dict:
        t0 = time.monotonic()
        messages = state.get("messages", [])
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 10)

        # max_iterations 도달 시 도구 바인딩 없이 호출하여 강제 종료
        force_answer = iteration >= max_iter
        system_content = _V3_SYSTEM_PROMPT
        if force_answer:
            system_content += _V3_FORCE_ANSWER_SUFFIX
            logger.warning(
                f"[agent_v3] iteration {iteration} >= max {max_iter} — 강제 final answer"
            )

        system = SystemMessage(content=system_content)
        active_llm = llm_without_tools if force_answer else llm_with_tools
        response = await active_llm.ainvoke([system] + list(messages))

        tool_calls = getattr(response, "tool_calls", None) or []
        latency = round((time.monotonic() - t0) * 1000, 2)

        result: dict = {
            "messages": [response],
            "node_latencies": {"agent": latency},
        }

        if tool_calls and not force_answer:
            # 도구 호출이 있으면 pending_tool_calls에 저장하고 iteration 증가
            result["pending_tool_calls"] = tool_calls
            result["iteration_count"] = iteration + 1
        else:
            # 최종 답변 — pending_tool_calls 비우기, final_text 명시적 저장
            result["pending_tool_calls"] = []
            if response.content:
                result["final_text"] = response.content

        logger.debug(
            f"[agent_v3] iteration={iteration} tool_calls={len(tool_calls)} "
            f"force_answer={force_answer} latency_ms={latency}"
        )
        return result

    return agent_node_v3


# ---------------------------------------------------------------------------
# v3 tools 노드
# ---------------------------------------------------------------------------


def make_tools_node_v3(tool_node_fn):
    """v3 tools 노드 팩토리.

    기존 ToolNode를 래핑하여 실행 후 tool_call_history에 메타데이터를 기록한다.

    Parameters
    ----------
    tool_node_fn : callable
        LangGraph ToolNode 인스턴스 또는 동등한 async callable.
    """

    async def tools_node_v3(state: GovOnGraphState, config: RunnableConfig) -> dict:
        t0 = time.monotonic()
        iteration = state.get("iteration_count", 0)
        pending = state.get("pending_tool_calls", [])

        # ToolNode 실행 (기존 로직 재사용)
        if hasattr(tool_node_fn, "ainvoke"):
            result = await tool_node_fn.ainvoke(state, config)
        else:
            result = await tool_node_fn(state)

        latency = round((time.monotonic() - t0) * 1000, 2)

        # tool_call_history 기록
        history_entries = []
        now = datetime.now(timezone.utc).isoformat()
        tool_messages = result.get("messages", [])

        # tool_call_id → ToolMessage 매핑 (인덱스 매칭 대신 ID 기반)
        msg_by_id: dict = {}
        for msg in tool_messages:
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id:
                msg_by_id[tc_id] = msg

        for tc in pending:
            tool_name = tc.get("name", "unknown")
            tc_id = tc.get("id")
            # ToolMessage에서 성공 여부 판단 (JSON 파싱 후 success 필드 확인)
            success = True
            msg = msg_by_id.get(tc_id) if tc_id else None
            if msg is not None:
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if parsed.get("success") is False:
                            success = False
                    except (json.JSONDecodeError, TypeError):
                        pass

            history_entries.append(
                {
                    "iteration": iteration,
                    "tool": tool_name,
                    "node_latency_ms": latency,
                    "success": success,
                    "timestamp": now,
                }
            )

        result["tool_call_history"] = history_entries
        result["pending_tool_calls"] = []  # 실행 완료 후 비우기
        result["node_latencies"] = {"tools": latency}

        logger.debug(f"[tools_v3] executed={len(pending)} latency_ms={latency}")
        return result

    return tools_node_v3
