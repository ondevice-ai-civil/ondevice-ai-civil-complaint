"""어댑터별 독립 도구를 동적으로 생성하는 팩토리 모듈.

AdapterRegistry에 등록된 모든 어댑터를 순회하여
각각에 대응하는 StructuredTool을 생성한다.
adapters.yaml에 어댑터를 추가하면 자동으로 새 도구가 등록된다.
"""

from __future__ import annotations

import json
from typing import Callable, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.inference.adapter_registry import AdapterRegistry


class AdapterToolInput(BaseModel):
    """어댑터 도구 입력 스키마."""

    query: str = Field(description="민원 또는 법률 관련 질의 내용")


def build_adapter_tools(draft_response_fn: Callable) -> List[StructuredTool]:
    """AdapterRegistry의 모든 어댑터에 대해 StructuredTool을 동적 생성한다.

    Parameters
    ----------
    draft_response_fn : Callable
        ``async (query, context, session) -> dict`` 시그니처의 답변 생성 함수.

    Returns
    -------
    List[StructuredTool]
        어댑터 수만큼의 도구 리스트.
    """
    registry = AdapterRegistry.get_instance()
    tools: List[StructuredTool] = []

    for adapter_name in registry.list_available():
        description = registry.get_tool_description(adapter_name)
        meta = registry.get_meta(adapter_name)
        tool_name = f"{adapter_name}_adapter"

        # 클로저로 adapter_name을 캡처
        def _make_execute(name: str) -> Callable:
            async def _adapter_execute(query: str) -> str:
                try:
                    result = await draft_response_fn(
                        query=query,
                        context={"adapter": name},
                        session=None,
                    )
                    return json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    return json.dumps(
                        {"error": str(e), "success": False},
                        ensure_ascii=False,
                    )

            return _adapter_execute

        requires_approval = meta.requires_approval if meta else True
        tool = StructuredTool.from_function(
            coroutine=_make_execute(adapter_name),
            name=tool_name,
            description=description,
            args_schema=AdapterToolInput,
            metadata={"requires_approval": requires_approval},
        )
        tools.append(tool)

    return tools
