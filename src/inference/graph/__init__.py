"""GovOn LangGraph runtime 패키지.

v4 아키텍처: ReAct + ToolNode 기반.

주요 public API:
- `build_govon_graph`: StateGraph 빌더 함수
- `GovOnGraphState`: graph state TypedDict
- `ApprovalStatus`: 승인 상태 enum
"""

from .state import ApprovalStatus, GovOnGraphState

try:
    from .builder import build_govon_graph
except Exception:
    import logging

    logging.getLogger(__name__).warning(
        "builder 모듈 로드 실패, build_govon_graph를 사용할 수 없습니다."
    )
    build_govon_graph = None  # type: ignore[assignment]

__all__ = [
    "build_govon_graph",
    "GovOnGraphState",
    "ApprovalStatus",
]
