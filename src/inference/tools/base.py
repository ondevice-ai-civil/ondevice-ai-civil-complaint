"""
GovOn Tool 기본 인터페이스.

모든 tool action은 BaseTool을 상속하며 동일한 입출력 계약을 따른다.
ToolRegistry를 통해 이름 기반으로 tool을 조회·실행할 수 있다.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 공용 입출력 스키마
# ---------------------------------------------------------------------------


class ToolInput(BaseModel):
    """모든 tool 입력의 기본 모델. 각 tool이 필드를 확장한다."""

    pass


class ToolOutput(BaseModel):
    """모든 tool 출력의 표준 모델.

    Attributes
    ----------
    success : bool
        실행 성공 여부.
    data : Any
        tool이 반환하는 결과 데이터.
    error : Optional[str]
        실패 시 에러 메시지.
    elapsed_ms : Optional[float]
        실행 소요 시간 (밀리초).
    tool_name : str
        실행된 tool 이름.
    """

    success: bool = True
    data: Any = None
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None
    tool_name: str = ""


# ---------------------------------------------------------------------------
# 추상 기본 클래스
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """GovOn shell agent tool의 추상 기본 클래스.

    모든 tool은 이 클래스를 상속하고 다음을 구현해야 한다:
    - ``name``: tool 식별자
    - ``description``: 용도 설명 (agent가 tool 선택 시 참조)
    - ``execute()``: 실제 로직
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @abstractmethod
    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        """tool 로직을 실행하고 ToolOutput을 반환한다."""
        ...

    async def run(self, tool_input: ToolInput) -> ToolOutput:
        """execute()를 래핑하여 소요 시간 측정 및 에러 처리를 수행한다."""
        start = time.monotonic()
        try:
            output = await self.execute(tool_input)
            output.elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            output.tool_name = self.name
            return output
        except Exception as e:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            logger.error(f"Tool '{self.name}' 실행 실패: {e}")
            return ToolOutput(
                success=False,
                error=str(e),
                elapsed_ms=elapsed,
                tool_name=self.name,
            )

    def get_schema(self) -> Dict[str, Any]:
        """tool의 메타데이터와 입력 스키마를 반환한다."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_input_schema(),
        }

    def _get_input_schema(self) -> Dict[str, Any]:
        """서브클래스에서 오버라이드하여 Pydantic 스키마를 반환할 수 있다."""
        return {}


# ---------------------------------------------------------------------------
# Tool 레지스트리
# ---------------------------------------------------------------------------


class ToolRegistry:
    """이름 기반 tool 조회·실행 레지스트리.

    Usage::

        registry = ToolRegistry()
        registry.register(RAGSearchTool(hybrid_engine=engine, retriever=retriever))
        output = await registry.execute("rag_search", RAGSearchInput(query="도로 파손"))
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """tool을 레지스트리에 등록한다."""
        if not tool.name:
            raise ValueError("tool.name이 비어 있습니다.")
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' 중복 등록 — 기존 인스턴스를 덮어씁니다.")
        self._tools[tool.name] = tool
        logger.info(f"Tool 등록: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """이름으로 tool을 조회한다."""
        return self._tools.get(name)

    async def execute(self, name: str, tool_input: ToolInput) -> ToolOutput:
        """이름으로 tool을 찾아 실행한다."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutput(
                success=False,
                error=f"등록되지 않은 tool입니다: {name}",
                tool_name=name,
            )
        return await tool.run(tool_input)

    def list_tools(self) -> List[Dict[str, Any]]:
        """등록된 모든 tool의 메타데이터를 반환한다."""
        return [tool.get_schema() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
