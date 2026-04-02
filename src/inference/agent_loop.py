"""세션 기반 에이전트 루프 모듈.

하나의 세션 안에서 요청 계획, tool 실행, 최종 응답 합성을 처리한다.
classify -> search_similar -> generate_civil_response 흐름을 기본으로 통합하며,
각 단계의 trace와 latency를 기록한다.

Issue: #393
"""

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from loguru import logger

from .session_context import SessionContext
from .tool_router import ExecutionPlan, ToolName, ToolRouter, ToolStep, ToolType, tool_name

# ---------------------------------------------------------------------------
# Tool 실행 결과 / 트레이스
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """단일 tool 실행 결과."""

    tool: ToolName
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": tool_name(self.tool),
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class AgentTrace:
    """에이전트 루프 전체 실행 트레이스."""

    request_id: str
    session_id: str
    plan: Optional[ExecutionPlan] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    final_text: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "plan": self.plan.tool_names if self.plan else [],
            "plan_reason": self.plan.reason if self.plan else "",
            "tool_results": [r.to_dict() for r in self.tool_results],
            "total_latency_ms": round(self.total_latency_ms, 2),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Tool 어댑터 프로토콜
# ---------------------------------------------------------------------------

# Tool 함수 시그니처: (query, context, previous_results) -> dict
ToolFunction = Callable[..., Any]

# Tool timeout (초)
DEFAULT_TOOL_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """세션 기반 에이전트 루프.

    하나의 요청에 대해:
    1. 세션 컨텍스트를 로드한다.
    2. ToolRouter로 실행 계획을 수립한다.
    3. 계획에 따라 tool을 순차 실행하고 결과를 누적한다.
    4. 최종 응답을 합성한다.
    5. 모든 단계의 trace와 latency를 기록한다.

    Parameters
    ----------
        tool_registry : Dict[ToolName, ToolFunction]
        tool 타입별 실행 함수. 각 함수는 비동기여야 한다.
    router : Optional[ToolRouter]
        커스텀 라우터. None이면 기본 ToolRouter를 사용한다.
    tool_timeout : float
        개별 tool 실행 타임아웃(초).
    """

    def __init__(
        self,
        tool_registry: Dict[ToolName, ToolFunction],
        router: Optional[ToolRouter] = None,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
    ) -> None:
        self._tools = {tool_name(name): runner for name, runner in tool_registry.items()}
        self._router = router or ToolRouter()
        self._tool_timeout = tool_timeout

    async def run(
        self,
        query: str,
        session: SessionContext,
        request_id: Optional[str] = None,
        force_tools: Optional[List[ToolName]] = None,
    ) -> AgentTrace:
        """에이전트 루프를 실행한다 (non-streaming).

        Parameters
        ----------
        query : str
            사용자 요청 텍스트.
        session : SessionContext
            현재 세션 컨텍스트.
        request_id : Optional[str]
            요청 ID. 미지정 시 자동 생성.
        force_tools : Optional[List[ToolName]]
            강제 실행할 tool 목록.

        Returns
        -------
        AgentTrace
            전체 실행 트레이스.
        """
        rid = request_id or str(uuid.uuid4())
        trace = AgentTrace(request_id=rid, session_id=session.session_id)
        loop_start = time.monotonic()

        try:
            # 1. 세션 컨텍스트에 현재 요청 기록
            session.add_turn("user", query)

            # 2. 실행 계획 수립
            has_context = bool(session.selected_evidences or session.draft_versions)
            plan = self._router.plan(query, has_context=has_context, force_tools=force_tools)
            trace.plan = plan
            logger.info(
                f"[AgentLoop] request_id={rid} plan={plan.tool_names} reason={plan.reason!r}"
            )

            # 3. tool 순차 실행
            accumulated: Dict[str, Any] = {
                "query": query,
                "session_context": session.build_context_summary(),
            }

            for step in plan.steps:
                result = await self._execute_tool(step, accumulated, session)
                trace.tool_results.append(result)
                step_name = step.step_id

                if result.success:
                    accumulated[step_name] = result.data
                    # 검색 결과를 세션에 저장
                    if step_name == ToolType.SEARCH_SIMILAR.value and "results" in result.data:
                        session.set_evidences(result.data["results"])
                else:
                    # 실패 시 fallback: 다음 단계는 빈 데이터로 진행
                    logger.warning(
                        f"[AgentLoop] tool {step_name} 실패: {result.error}. "
                        f"빈 결과로 계속 진행합니다."
                    )
                    accumulated[step_name] = {}

            # 4. 최종 텍스트 추출
            final_text = self._extract_final_text(accumulated, plan)
            trace.final_text = final_text

            # 5. 세션에 응답 기록
            session.add_turn("assistant", final_text)
            if final_text:
                tool_names = plan.tool_names
                session.add_draft(final_text, tool_trace=tool_names)

        except Exception as e:
            trace.error = str(e)
            logger.error(f"[AgentLoop] request_id={rid} 루프 실행 오류: {e}", exc_info=True)
        finally:
            trace.total_latency_ms = (time.monotonic() - loop_start) * 1000
            logger.info(
                f"[AgentLoop] request_id={rid} 완료 "
                f"total={trace.total_latency_ms:.1f}ms "
                f"tools={[r.to_dict() for r in trace.tool_results]}"
            )

        return trace

    async def run_stream(
        self,
        query: str,
        session: SessionContext,
        request_id: Optional[str] = None,
        force_tools: Optional[List[ToolName]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """에이전트 루프를 스트리밍으로 실행한다.

        각 단계의 진행 상황과 최종 결과를 SSE 이벤트로 전달한다.

        Yields
        ------
        Dict[str, Any]
            SSE 이벤트 데이터. type 필드로 이벤트 종류를 구분:
            - "plan": 실행 계획
            - "tool_start": tool 실행 시작
            - "tool_result": tool 실행 결과
            - "generate_chunk": 생성 텍스트 청크
            - "final": 최종 결과
            - "error": 오류
        """
        rid = request_id or str(uuid.uuid4())
        loop_start = time.monotonic()
        trace = AgentTrace(request_id=rid, session_id=session.session_id)

        try:
            session.add_turn("user", query)

            # 실행 계획
            has_context = bool(session.selected_evidences or session.draft_versions)
            plan = self._router.plan(query, has_context=has_context, force_tools=force_tools)
            trace.plan = plan

            yield {
                "type": "plan",
                "request_id": rid,
                "plan": plan.tool_names,
                "reason": plan.reason,
            }

            accumulated: Dict[str, Any] = {
                "query": query,
                "session_context": session.build_context_summary(),
            }

            for step in plan.steps:
                yield {
                    "type": "tool_start",
                    "request_id": rid,
                    "tool": step.step_id,
                }

                result = await self._execute_tool(step, accumulated, session)
                trace.tool_results.append(result)

                yield {
                    "type": "tool_result",
                    "request_id": rid,
                    "tool": step.step_id,
                    "success": result.success,
                    "latency_ms": round(result.latency_ms, 2),
                    "error": result.error,
                }

                if result.success:
                    accumulated[step.step_id] = result.data
                    if step.step_id == ToolType.SEARCH_SIMILAR.value and "results" in result.data:
                        session.set_evidences(result.data["results"])
                else:
                    accumulated[step.step_id] = {}

            # 최종 텍스트
            final_text = self._extract_final_text(accumulated, plan)
            trace.final_text = final_text
            session.add_turn("assistant", final_text)
            if final_text:
                session.add_draft(final_text, tool_trace=plan.tool_names)

            trace.total_latency_ms = (time.monotonic() - loop_start) * 1000

            yield {
                "type": "final",
                "request_id": rid,
                "text": final_text,
                "trace": trace.to_dict(),
                "finished": True,
            }

        except Exception as e:
            trace.error = str(e)
            trace.total_latency_ms = (time.monotonic() - loop_start) * 1000
            logger.error(f"[AgentLoop] stream request_id={rid} 오류: {e}", exc_info=True)
            yield {
                "type": "error",
                "request_id": rid,
                "error": "에이전트 처리 중 내부 오류가 발생했습니다.",
                "finished": True,
            }

    async def _execute_tool(
        self,
        step: ToolStep,
        accumulated: Dict[str, Any],
        session: SessionContext,
    ) -> ToolResult:
        """단일 tool을 타임아웃과 함께 실행한다."""
        step_name = step.step_id
        tool_fn = self._tools.get(step_name)
        if tool_fn is None:
            return ToolResult(
                tool=step.tool,
                success=False,
                error=f"등록되지 않은 tool: {step_name}",
            )

        start = time.monotonic()
        try:
            result_data = await asyncio.wait_for(
                tool_fn(
                    query=accumulated.get("query", ""),
                    context=accumulated,
                    session=session,
                ),
                timeout=self._tool_timeout,
            )
            elapsed = (time.monotonic() - start) * 1000

            return ToolResult(
                tool=step.tool,
                success=True,
                data=result_data if isinstance(result_data, dict) else {"result": result_data},
                latency_ms=elapsed,
            )

        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            error_msg = f"tool {step_name} 타임아웃 ({self._tool_timeout}초)"
            logger.warning(f"[AgentLoop] {error_msg}")
            return ToolResult(
                tool=step.tool,
                success=False,
                error=error_msg,
                latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                f"[AgentLoop] tool {step_name} 실행 오류: {e}",
                exc_info=True,
            )
            return ToolResult(
                tool=step.tool,
                success=False,
                error=str(e),
                latency_ms=elapsed,
            )

    @staticmethod
    def _extract_final_text(accumulated: Dict[str, Any], plan: ExecutionPlan) -> str:
        """누적된 결과에서 최종 응답 텍스트를 추출한다.

        생성 결과가 있으면 그것을 사용하고,
        없으면 classify/search_similar/api_lookup 결과를 요약한다.
        """
        for tool_type in (
            ToolType.GENERATE_PUBLIC_DOC,
            ToolType.GENERATE_CIVIL_RESPONSE,
        ):
            gen_data = accumulated.get(tool_type.value, {})
            if gen_data.get("text"):
                return gen_data["text"]

        for step in plan.steps:
            step_data = accumulated.get(step.step_id, {})
            if isinstance(step_data, dict) and step_data.get("text"):
                return step_data["text"]

        # 생성 결과가 없으면 다른 결과를 요약
        parts: List[str] = []

        classify_data = accumulated.get(ToolType.CLASSIFY.value, {})
        if classify_data.get("classification"):
            cls = classify_data["classification"]
            category = cls.get("category", "")
            confidence = cls.get("confidence", 0)
            reason = cls.get("reason", "")
            parts.append(
                f"[분류 결과] 카테고리: {category} (신뢰도: {confidence:.0%})\n사유: {reason}"
            )

        search_data = accumulated.get(ToolType.SEARCH_SIMILAR.value, {})
        if search_data.get("results"):
            results = search_data["results"][:3]
            search_lines = ["[유사 민원 사례]"]
            for i, r in enumerate(results, 1):
                title = r.get("title", r.get("qnaTitle", ""))
                content = r.get("content", r.get("qnaContent", r.get("question", "")))[:100]
                search_lines.append(f"{i}. {title}: {content}")
            parts.append("\n".join(search_lines))

        # API 조회 결과 처리
        api_data = accumulated.get(ToolType.API_LOOKUP.value, {})
        if api_data:
            # context_text가 있으면 그대로 추가
            ctx_text = api_data.get("context_text", "")
            if ctx_text:
                parts.append(ctx_text)
            else:
                # context_text 없으면 data.results에서 직접 구성
                api_results = api_data.get("data", {}).get("results", [])
                if api_results:
                    api_lines = ["[공공데이터 유사 민원 사례]"]
                    for i, r in enumerate(api_results[:3], 1):
                        title = r.get("title", r.get("qnaTitle", ""))
                        content = r.get("content", r.get("qnaContent", ""))[:100]
                        api_lines.append(f"{i}. {title}: {content}")
                    parts.append("\n".join(api_lines))

        return "\n\n".join(parts) if parts else "요청을 처리할 수 없습니다."
