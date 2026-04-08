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
from typing import TYPE_CHECKING, Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

_SYSTEM_PROMPT = (
    "당신은 GovOn 민원 답변 보조 시스템입니다.\n"
    "사용자의 요청을 분석하여 적절한 도구를 호출하거나 직접 답변하세요.\n\n"
    "판단 원칙:\n"
    "- 각 도구의 설명을 읽고 필요한 것만 호출하세요.\n"
    "- 답변 초안이 필요하면 도메인에 맞는 어댑터 도구를 선택하세요.\n"
    "- 근거나 출처가 필요하면 검색 도구를 호출하세요.\n"
    "- 이전 대화에서 이미 초안이 있으면 어댑터 재호출이 불필요할 수 있습니다.\n"
    "- 충분한 정보가 모이면 최종 답변을 직접 작성하세요.\n"
    "- 불필요한 도구는 호출하지 마세요.\n"
)


# ---------------------------------------------------------------------------
# session_load
# ---------------------------------------------------------------------------


def make_session_load_node(session_store: "SessionStore"):
    """세션 로드 노드 팩토리.

    checkpointer가 messages를 복원하므로 최소한의 세션 식별만 수행한다.
    """

    async def session_load_node(state: GovOnGraphState) -> dict:
        t0 = time.monotonic()
        session_id = state.get("session_id", "")
        latency = round((time.monotonic() - t0) * 1000, 2)
        logger.debug(f"[session_load] session_id={session_id} latency_ms={latency}")
        return {
            "session_id": session_id,
            "node_latencies": {"session_load": latency},
        }

    return session_load_node


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------


def make_agent_node(llm, tools: list):
    """ReAct agent 노드 팩토리.

    LLM에 도구를 바인딩하고, 매 호출마다 system prompt + messages를
    전달하여 도구 호출 또는 최종 응답을 생성한다.
    """
    llm_with_tools = llm.bind_tools(tools)

    def _count_consecutive_rejections(messages: list) -> int:
        """messages를 역순 순회하여 연속 거부 피드백 메시지 수를 카운트한다."""
        count = 0
        for msg in reversed(messages):
            if (
                isinstance(msg, HumanMessage)
                and "사용자가 도구 실행을 거부했습니다" in (msg.content or "")
            ):
                count += 1
            elif isinstance(msg, AIMessage):
                # AI 응답을 만나면 그 이전의 거부는 이전 사이클이므로 중단
                break
            else:
                break
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
        response = await llm_with_tools.ainvoke([system] + list(messages))
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

    def approval_wait_node(state: GovOnGraphState) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"approval_status": ApprovalStatus.REJECTED.value}

        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", [])
        tool_names = [tc["name"] for tc in tool_calls]

        approval_required = [
            name for name in tool_names if approval_map.get(name, False)
        ]

        logger.info(
            f"[approval_wait] interrupt: tools={tool_names} "
            f"approval_required={approval_required}"
        )

        response = interrupt({
            "tools": tool_names,
            "message": f"다음 도구를 실행합니다: {', '.join(tool_names)}",
            "approval_required": approval_required,
        })

        approved = (
            response.get("approved", False)
            if isinstance(response, dict)
            else bool(response)
        )

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

    승인이면 tools로, 거부이면 agent로 돌려보내 대안을 제시하게 한다.
    """
    if state.get("approval_status") == ApprovalStatus.APPROVED.value:
        return "tools"
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

        # 최종 텍스트: 마지막 AIMessage 중 tool_calls가 없는 것
        final_text = ""
        for msg in reversed(messages):
            if (
                hasattr(msg, "type")
                and msg.type == "ai"
                and not getattr(msg, "tool_calls", None)
                and hasattr(msg, "content")
            ):
                final_text = msg.content
                break

        # evidence: ToolMessage들에서 수집
        evidence_items: List[Dict[str, Any]] = []
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                try:
                    data = (
                        json.loads(msg.content)
                        if isinstance(msg.content, str)
                        else msg.content
                    )
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
