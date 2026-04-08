"""LangGraph ToolNode용 도구 팩토리 패키지.

기존 capability 클래스를 StructuredTool로 래핑하여
LangGraph ToolNode에서 사용할 수 있도록 한다.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .analysis_tools import build_analysis_tools
from .search_tools import build_search_tools


def build_all_tools(
    *,
    rag_search_fn: Callable[..., Any],
    api_lookup_action: Optional[Any] = None,
    draft_response_fn: Optional[Callable[..., Any]] = None,
) -> list:
    """전체 도구 목록을 동적으로 생성한다.

    Parameters
    ----------
    rag_search_fn : Callable
        RAG 검색 실행 클로저.
    api_lookup_action : Optional[MinwonAnalysisAction]
        공공데이터포털 API Action 인스턴스.
    draft_response_fn : Optional[Callable]
        답변 초안 생성 클로저. 있으면 adapter_tools를 추가.

    Returns
    -------
    list[StructuredTool]
        전체 도구 목록.
    """
    tools: list = []
    tools.extend(build_search_tools(rag_search_fn, api_lookup_action))
    tools.extend(build_analysis_tools(api_lookup_action))
    if draft_response_fn:
        try:
            from .adapter_tools import build_adapter_tools

            tools.extend(build_adapter_tools(draft_response_fn))
        except ImportError:
            import logging

            logging.getLogger(__name__).warning("adapter_tools 모듈 로드 실패, 어댑터 도구 생략")
    return tools


def get_tool_approval_map(tools: list) -> dict[str, bool]:
    """각 도구의 requires_approval 메타데이터를 dict로 반환한다.

    approval_wait_node에서 동적 조회에 사용한다.

    Parameters
    ----------
    tools : list[StructuredTool]
        도구 목록.

    Returns
    -------
    dict[str, bool]
        {tool_name: requires_approval} 매핑.
    """
    result: dict[str, bool] = {}
    for tool in tools:
        meta = getattr(tool, "metadata", None) or {}
        result[tool.name] = meta.get("requires_approval", False)
    return result


__all__ = [
    "build_all_tools",
    "build_analysis_tools",
    "build_search_tools",
    "get_tool_approval_map",
]
