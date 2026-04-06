"""GovOn LangGraph state schema.

Issue #415: LangGraph runtime 기반 및 planner/executor adapter 구성.

이 모듈은 StateGraph의 모든 노드가 공유하는 state 타입을 정의한다.
`GovOnGraphState`는 TypedDict로, LangGraph가 state 병합에 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


class ApprovalStatus(str, Enum):
    """human-in-the-loop 승인 상태."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TaskType(str, Enum):
    """planner가 분류하는 작업 유형."""

    DRAFT_RESPONSE = "draft_response"  # 민원 답변 초안 작성
    REVISE_RESPONSE = "revise_response"  # 답변 수정
    APPEND_EVIDENCE = "append_evidence"  # 근거 보강
    LOOKUP_STATS = "lookup_stats"  # 통계/사례 조회
    ISSUE_DETECTION = "issue_detection"  # 이슈 탐지
    STATS_QUERY = "stats_query"  # 통계 조회
    KEYWORD_ANALYSIS = "keyword_analysis"  # 키워드 분석
    DEMOGRAPHICS_QUERY = "demographics_query"  # 인구통계 조회


@dataclass
class ToolPlan:
    """planner가 생성하는 구조화된 실행 계획.

    Parameters
    ----------
    task_type : TaskType
        작업 분류.
    goal : str
        사용자에게 보여줄 작업 설명 (한국어, 1-2문장).
    reason : str
        이 작업이 필요한 이유 (한국어, 1문장).
    tools : List[str]
        실행할 tool 이름 목록 (순서대로).
        예: ["rag_search", "api_lookup", "draft_civil_response"]
    tool_summaries : List[str]
        각 tool의 human-readable approval_summary.
        CLI 렌더링과 approval prompt에 사용된다.
        기본값은 빈 리스트이며, planner adapter가 registry metadata로 채운다.
    """

    task_type: TaskType
    goal: str
    reason: str
    tools: List[str]
    tool_summaries: Optional[List[str]] = field(default=None)
    adapter_mode: str = "llm"  # "llm" | "regex" (CI fallback)

    def __post_init__(self) -> None:
        if self.tool_summaries is None:
            self.tool_summaries = list(self.tools)


class GovOnGraphState(TypedDict, total=False):
    """GovOn LangGraph graph state.

    모든 노드가 공유하는 state object.
    planner와 executor가 동일한 state를 읽고 쓴다.

    `messages` 필드는 `add_messages` reducer를 사용하여
    LangGraph가 메시지를 자동 병합한다.
    """

    # --- 세션 식별 ---
    session_id: str
    request_id: str

    # --- 메시지 히스토리 (LangGraph add_messages reducer 사용) ---
    messages: Annotated[Sequence[AnyMessage], add_messages]

    # --- planner 출력 ---
    task_type: str  # TaskType.value
    goal: str  # 승인 프롬프트에 표시할 작업 설명
    reason: str  # 작업 이유
    planned_tools: List[str]  # 실행 예정 tool 이름 리스트
    tool_summaries: List[str]  # 각 planned_tool의 human-readable approval_summary
    adapter_mode: str  # planner adapter 모드 ("regex" | "llm")

    # --- approval gate ---
    approval_status: str  # ApprovalStatus.value

    # --- executor 출력 ---
    tool_results: Dict[str, Any]  # {tool_name: result_dict, ...}
    accumulated_context: Dict[str, Any]  # tool 간 전달되는 누적 컨텍스트

    # --- synthesis 출력 ---
    final_text: str  # 최종 사용자 응답 텍스트

    # --- 메타데이터 ---
    error: Optional[str]
    interrupt_reason: Optional[str]  # "user_cancel" | "timeout" | None
    total_latency_ms: float

    # --- 레이턴시 계측 ---
    # 각 노드가 {"<node_name>": latency_ms} 형태로 반환하며,
    # _merge_dicts reducer가 모든 노드의 값을 누적 병합한다.
    # 예: {"session_load": 2.1, "planner": 45.3, "tool_execute": 312.5, ...}
    node_latencies: Annotated[Dict[str, float], _merge_dicts]
