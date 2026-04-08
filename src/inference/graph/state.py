"""GovOn LangGraph state schema.

v4 아키텍처: ReAct 기반 messages-only state.
모든 도구 결과는 messages(ToolMessage)로 동적 누적되며,
특정 도구명을 state에 하드코딩하지 않는다.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """두 dict를 병합하는 reducer.

    LangGraph state에서 여러 노드가 동일 키에 값을 반환할 때 사용한다.
    후행 dict(b)의 값이 선행 dict(a)를 덮어쓴다.
    """
    merged = dict(a) if a else {}
    if b:
        merged.update(b)
    return merged


def _append_list(a: List, b: List) -> List:
    """리스트를 append하는 reducer (기존 항목 보존)."""
    return (a or []) + (b or [])


class ApprovalStatus(str, Enum):
    """human-in-the-loop 승인 상태."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class GovOnGraphState(TypedDict, total=False):
    """GovOn LangGraph graph state (v4).

    ReAct 루프 기반. 모든 도구 호출/결과는 messages에 누적된다.
    """

    # --- 세션 식별 ---
    session_id: str
    request_id: str

    # --- LangGraph 핵심: messages 누적 (ReAct 루프) ---
    messages: Annotated[Sequence[AnyMessage], add_messages]

    # --- Human-in-the-Loop ---
    approval_status: str  # ApprovalStatus.value

    # --- 최종 출력 ---
    final_text: str
    evidence_items: List[Dict[str, Any]]

    # --- 에러/인터럽트 ---
    error: Optional[str]
    interrupt_reason: Optional[str]

    # --- 레이턴시 계측 ---
    node_latencies: Annotated[Dict[str, float], _merge_dicts]

    # --- ReAct v3 루프 제어 ---
    iteration_count: int  # 현재 agent→tools 사이클 횟수 (0부터 시작)
    max_iterations: int  # 최대 허용 iteration 수 (기본 10)

    # --- v3 도구 호출 메타데이터 ---
    tool_call_history: Annotated[List[Dict[str, Any]], _append_list]
    # 각 항목: {"iteration": int, "tool": str, "latency_ms": float, "success": bool, "timestamp": str}

    # --- v3 위험도 기반 승인 정책 ---
    pending_tool_calls: List[Dict[str, Any]]  # agent가 생성한 tool_calls (OpenAI format)
