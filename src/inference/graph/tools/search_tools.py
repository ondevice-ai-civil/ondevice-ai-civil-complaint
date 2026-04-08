"""LangGraph ToolNode용 검색 도구 팩토리.

기존 RagSearchCapability, ApiLookupCapability에 위임하여
StructuredTool 인스턴스를 동적 생성한다.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic 스키마 — LLM이 생성하는 JSON 인자
# ---------------------------------------------------------------------------


class RagSearchInput(BaseModel):
    """rag_search 도구 입력 스키마."""

    query: str = Field(..., description="검색할 민원 관련 키워드 또는 질의문")
    top_k: int = Field(5, description="반환할 최대 결과 수 (1~50)", ge=1, le=50)
    source_types: Optional[List[str]] = Field(
        None,
        description="검색 대상 문서 유형 (case, law, manual, notice). 미지정 시 전체 검색",
    )


class ApiLookupInput(BaseModel):
    """api_lookup 도구 입력 스키마."""

    query: str = Field(..., description="유사 민원 검색 질의문")
    ret_count: int = Field(5, description="반환할 유사 민원 수 (1~20)", ge=1, le=20)


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------


def build_search_tools(
    rag_search_fn: Callable[..., Any],
    api_lookup_action: Optional[Any] = None,
) -> list:
    """검색 관련 StructuredTool 목록을 생성한다.

    Parameters
    ----------
    rag_search_fn : Callable
        RAG 검색 실행 클로저 (async (query, context, session) -> dict).
    api_lookup_action : Optional[MinwonAnalysisAction]
        공공데이터포털 API Action 인스턴스. None이면 빈 결과 반환.

    Returns
    -------
    list[StructuredTool]
        [rag_search_tool, api_lookup_tool]
    """
    from src.inference.graph.capabilities.rag_search import RagSearchCapability
    from src.inference.graph.capabilities.api_lookup import ApiLookupCapability

    # -- rag_search 클로저 캡처 --
    _rag_cap = RagSearchCapability(execute_fn=rag_search_fn)

    async def _rag_search(
        query: str,
        top_k: int = 5,
        source_types: Optional[List[str]] = None,
    ) -> str:
        context: dict[str, Any] = {"top_k": top_k}
        if source_types is not None:
            context["source_types"] = source_types
        result = await _rag_cap.execute(query=query, context=context, session=None)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    rag_search_tool = StructuredTool.from_function(
        coroutine=_rag_search,
        name="rag_search",
        description=(
            "민원 관련 문서, 매뉴얼, 판례, 법령을 로컬 벡터 DB에서 검색합니다. "
            "민원 답변의 근거 자료를 찾을 때 사용하세요. "
            "반환값: 관련 문서 목록 (제목, 내용 발췌, 유사도 점수)"
        ),
        args_schema=RagSearchInput,
        metadata={"requires_approval": False},
    )

    # -- api_lookup 클로저 캡처 --
    _api_cap = ApiLookupCapability(action=api_lookup_action)

    async def _api_lookup(
        query: str,
        ret_count: int = 5,
    ) -> str:
        context: dict[str, Any] = {"ret_count": ret_count}
        result = await _api_cap.execute(query=query, context=context, session=None)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    api_lookup_tool = StructuredTool.from_function(
        coroutine=_api_lookup,
        name="api_lookup",
        description=(
            "공공데이터포털(data.go.kr) API로 민원 분석 보고서, 통계, 정책 데이터를 조회합니다. "
            "최신 정책 정보나 외부 데이터가 필요할 때 사용하세요. "
            "반환값: API 조회 결과 (제목, 요약, 출처 링크)"
        ),
        args_schema=ApiLookupInput,
        metadata={"requires_approval": False},
    )

    return [rag_search_tool, api_lookup_tool]
