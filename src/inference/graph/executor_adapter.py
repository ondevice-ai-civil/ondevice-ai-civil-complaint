"""Executor adapter: tool registry에서 tool을 조회하고 실행.

Issue #415: LangGraph runtime 기반 및 planner/executor adapter 구성.
Issue #416: tool metadata registry 및 LangGraph executor binding 정리.

두 가지 구현체를 제공한다:
- `ExecutorAdapter` (ABC): 추상 인터페이스
- `RegistryExecutorAdapter`: CapabilityBase 기반 registry를 사용하는 구현체
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class ExecutorAdapter(ABC):
    """Tool executor 추상 인터페이스.

    LangGraph graph의 `tool_execute` 노드에서 호출된다.
    """

    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """tool을 실행하고 결과를 반환한다.

        Parameters
        ----------
        tool_name : str
            실행할 tool 이름.
        query : str
            사용자 요청 텍스트.
        context : Dict[str, Any]
            누적 컨텍스트 (이전 tool 결과 포함).

        Returns
        -------
        Dict[str, Any]
            tool 실행 결과. 최소 {"success": bool, ...} 형태.
            실패 시 {"success": False, "error": str}.
        """
        ...

    @abstractmethod
    def list_tools(self) -> list[str]:
        """등록된 tool 이름 목록을 반환한다."""
        ...


class RegistryExecutorAdapter(ExecutorAdapter):
    """기존 tool_registry를 재사용하는 executor.

    `tool_registry`는 `Dict[str, Callable]` 형태로 주입받는다.
    각 callable은 `async (query, context, session) -> dict` 시그니처여야 한다.
    기존 `AgentLoop._execute_tool()` 로직을 계승한다.

    Parameters
    ----------
    tool_registry : Dict[str, Callable]
        tool 이름 -> async callable 매핑.
    session_store : SessionStore
        GovOn 세션 저장소. executor가 tool 호출 시 세션을 주입한다.
    default_timeout : float
        tool 실행 제한 시간 (초). 기본값 30.0.
    """

    def __init__(
        self,
        tool_registry: Dict[str, Callable],
        session_store: Any,  # SessionStore (순환 import 방지를 위해 Any 사용)
        default_timeout: float = 30.0,
    ) -> None:
        self._tools = tool_registry
        self._session_store = session_store
        self._default_timeout = default_timeout

    async def execute(
        self,
        tool_name: str,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """tool을 조회하고 타임아웃 포함하여 실행한다.

        registry에 등록되지 않은 tool은 비MVP capability로 차단한다.
        """
        from src.inference.graph.capabilities.registry import is_mvp_capability

        # 비MVP capability 차단
        if not is_mvp_capability(tool_name):
            logger.warning(f"[RegistryExecutorAdapter] 비MVP capability 차단: {tool_name}")
            return {"success": False, "error": f"비MVP capability: {tool_name}"}

        tool_fn = self._tools.get(tool_name)
        if tool_fn is None:
            return {"success": False, "error": f"등록되지 않은 tool: {tool_name}"}

        session = self._session_store.get_or_create(context.get("session_id"))
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool_fn(query=query, context=context, session=session),
                timeout=self._default_timeout,
            )
            latency = (time.monotonic() - start) * 1000
            if isinstance(result, dict):
                if "latency_ms" not in result:
                    result["latency_ms"] = latency
                result.setdefault("success", True)
                return result
            return {"success": True, "result": result, "latency_ms": latency}
        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            return {
                "success": False,
                "error": f"tool {tool_name} 타임아웃 ({self._default_timeout}초)",
                "latency_ms": latency,
            }
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"[RegistryExecutorAdapter] tool {tool_name} 오류: {exc}", exc_info=True)
            return {"success": False, "error": str(exc), "latency_ms": latency}

    def list_tools(self) -> list[str]:
        """등록된 tool 이름 목록을 반환한다."""
        return list(self._tools.keys())

    def get_tool_metadata(self, tool_name: str) -> Optional[dict]:
        """capability의 planner metadata를 반환한다.

        CapabilityBase 인스턴스가 등록된 경우 metadata 프로퍼티에서 정보를 추출하고,
        일반 callable인 경우 이름만 포함된 기본 dict를 반환한다.
        등록되지 않은 tool이면 None을 반환한다.

        Parameters
        ----------
        tool_name : str
            조회할 tool 이름.

        Returns
        -------
        Optional[dict]
            tool metadata dict 또는 None.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return None
        # CapabilityBase 인터페이스 지원
        if hasattr(tool, "metadata"):
            meta = tool.metadata
            return {
                "name": meta.name,
                "description": meta.description,
                "approval_summary": meta.approval_summary,
                "provider": getattr(meta, "provider", ""),
            }
        return {
            "name": tool_name,
            "description": "",
            "approval_summary": "",
            "provider": "",
        }

    def get_tool_descriptions_for_planner(self) -> List[dict]:
        """planner가 읽을 tool 목록을 단일 메서드로 노출한다.

        등록된 모든 tool의 metadata를 dict 목록으로 반환한다.
        CapabilityBase 인스턴스는 풍부한 metadata를, 일반 callable은
        이름만 포함된 기본 dict를 반환한다.

        Returns
        -------
        List[dict]
            각 tool의 metadata dict 목록.
        """
        descriptions: List[dict] = []
        for name in self._tools:
            meta = self.get_tool_metadata(name)
            if meta is not None:
                descriptions.append(meta)
        return descriptions
