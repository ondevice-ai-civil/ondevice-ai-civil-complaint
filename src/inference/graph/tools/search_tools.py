"""LangGraph ToolNode용 검색 도구 팩토리.

기존 ApiLookupCapability에 위임하여
StructuredTool 인스턴스를 동적 생성한다.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic 스키마 — LLM이 생성하는 JSON 인자
# ---------------------------------------------------------------------------


class ApiLookupInput(BaseModel):
    """api_lookup tool input schema."""

    query: str = Field(..., description="Search query for similar civil complaints")
    ret_count: int = Field(5, description="Number of results to return (1-20)", ge=1, le=20)


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------


def build_search_tools(
    api_lookup_action: Optional[Any] = None,
) -> list:
    """검색 관련 StructuredTool 목록을 생성한다.

    Parameters
    ----------
    api_lookup_action : Optional[MinwonAnalysisAction]
        공공데이터포털 API Action 인스턴스. None이면 빈 결과 반환.

    Returns
    -------
    list[StructuredTool]
        [api_lookup_tool]
    """
    from src.inference.graph.capabilities.api_lookup import ApiLookupCapability

    _api_cap = ApiLookupCapability(action=api_lookup_action)

    async def _api_lookup(
        query: str,
        ret_count: int = 5,
    ) -> str:
        context: dict[str, Any] = {"ret_count": ret_count}
        try:
            result = await _api_cap.execute(query=query, context=context, session=None)
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)

    api_lookup_tool = StructuredTool.from_function(
        coroutine=_api_lookup,
        name="api_lookup",
        description=(
            "Search the Korea Public Data Portal (data.go.kr) for civil complaint analysis reports, "
            "statistics, and policy data. USE THIS TOOL when you need official government data, "
            "recent policy information, or external references to support your answer. "
            "Returns: title, summary, and source URL for each result."
        ),
        args_schema=ApiLookupInput,
        metadata={"requires_approval": False},
    )

    return [api_lookup_tool]
