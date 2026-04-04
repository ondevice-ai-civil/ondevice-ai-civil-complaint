"""세션 기반 task loop."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from loguru import logger

from .query_builder import build_query_variants, build_runtime_query_context, resolve_tool_query
from .session_context import SessionContext
from .tool_router import ExecutionPlan, ToolName, ToolRouter, ToolStep, ToolType, tool_name


@dataclass
class ToolResult:
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
            "tool_results": [result.to_dict() for result in self.tool_results],
            "total_latency_ms": round(self.total_latency_ms, 2),
            "error": self.error,
        }


ToolFunction = Callable[..., Any]
DEFAULT_TOOL_TIMEOUT = 30.0


class AgentLoop:
    """GovOn MVP capability loop."""

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
        rid = request_id or str(uuid.uuid4())
        trace = AgentTrace(request_id=rid, session_id=session.session_id)
        loop_start = time.monotonic()
        started_at = time.time()

        try:
            session.add_turn("user", query)

            has_context = bool(session.tool_runs or session.conversations)
            plan = self._router.plan(query, has_context=has_context, force_tools=force_tools)
            trace.plan = plan

            accumulated: Dict[str, Any] = build_runtime_query_context(session, query)
            accumulated["conversation"] = [
                {"role": turn.role, "content": turn.content} for turn in session.recent_history[-5:]
            ]
            accumulated["query_variants"] = build_query_variants(
                query,
                tool_names=plan.tool_names,
                context=accumulated,
            )

            for step in plan.steps:
                result = await self._execute_tool(step, accumulated, session)
                trace.tool_results.append(result)
                accumulated[step.step_id] = result.data if result.success else {}
                session.add_tool_run(
                    tool=step.step_id,
                    graph_run_request_id=rid,
                    success=result.success,
                    latency_ms=result.latency_ms,
                    error=result.error,
                    metadata=self._build_tool_log_metadata(result.data),
                )

            trace.final_text = self._extract_final_text(accumulated, plan)
            session.add_turn("assistant", trace.final_text)

        except Exception as exc:
            trace.error = str(exc)
            logger.error(f"[AgentLoop] request_id={rid} 오류: {exc}", exc_info=True)
        finally:
            trace.total_latency_ms = (time.monotonic() - loop_start) * 1000
            self._record_graph_run(
                session=session,
                trace=trace,
                started_at=started_at,
                completed_at=time.time(),
            )

        return trace

    async def run_stream(
        self,
        query: str,
        session: SessionContext,
        request_id: Optional[str] = None,
        force_tools: Optional[List[ToolName]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        rid = request_id or str(uuid.uuid4())
        loop_start = time.monotonic()
        started_at = time.time()
        trace = AgentTrace(request_id=rid, session_id=session.session_id)

        try:
            session.add_turn("user", query)
            has_context = bool(session.tool_runs or session.conversations)
            plan = self._router.plan(query, has_context=has_context, force_tools=force_tools)
            trace.plan = plan

            yield {
                "type": "plan",
                "request_id": rid,
                "plan": plan.tool_names,
                "reason": plan.reason,
            }

            accumulated: Dict[str, Any] = build_runtime_query_context(session, query)
            accumulated["query_variants"] = build_query_variants(
                query,
                tool_names=plan.tool_names,
                context=accumulated,
            )

            for step in plan.steps:
                yield {"type": "tool_start", "request_id": rid, "tool": step.step_id}
                result = await self._execute_tool(step, accumulated, session)
                trace.tool_results.append(result)
                accumulated[step.step_id] = result.data if result.success else {}
                session.add_tool_run(
                    tool=step.step_id,
                    graph_run_request_id=rid,
                    success=result.success,
                    latency_ms=result.latency_ms,
                    error=result.error,
                    metadata=self._build_tool_log_metadata(result.data),
                )
                yield {
                    "type": "tool_result",
                    "request_id": rid,
                    "tool": step.step_id,
                    "success": result.success,
                    "latency_ms": round(result.latency_ms, 2),
                    "error": result.error,
                }

            trace.final_text = self._extract_final_text(accumulated, plan)
            session.add_turn("assistant", trace.final_text)
            trace.total_latency_ms = (time.monotonic() - loop_start) * 1000
            yield {
                "type": "final",
                "request_id": rid,
                "text": trace.final_text,
                "trace": trace.to_dict(),
                "finished": True,
            }

        except Exception as exc:
            trace.error = str(exc)
            trace.total_latency_ms = (time.monotonic() - loop_start) * 1000
            logger.error(f"[AgentLoop] stream request_id={rid} 오류: {exc}", exc_info=True)
            yield {
                "type": "error",
                "request_id": rid,
                "error": "에이전트 처리 중 내부 오류가 발생했습니다.",
                "finished": True,
            }
        finally:
            if trace.total_latency_ms == 0.0:
                trace.total_latency_ms = (time.monotonic() - loop_start) * 1000
            self._record_graph_run(
                session=session,
                trace=trace,
                started_at=started_at,
                completed_at=time.time(),
            )

    async def _execute_tool(
        self,
        step: ToolStep,
        accumulated: Dict[str, Any],
        session: SessionContext,
    ) -> ToolResult:
        step_name = step.step_id
        tool_fn = self._tools.get(step_name)
        if tool_fn is None:
            return ToolResult(
                tool=step.tool, success=False, error=f"등록되지 않은 tool: {step_name}"
            )

        start = time.monotonic()
        try:
            execution_query = resolve_tool_query(step_name, accumulated)
            result_data = await asyncio.wait_for(
                tool_fn(
                    query=execution_query,
                    context=accumulated,
                    session=session,
                ),
                timeout=self._tool_timeout,
            )
            return ToolResult(
                tool=step.tool,
                success=True,
                data=result_data if isinstance(result_data, dict) else {"result": result_data},
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool=step.tool,
                success=False,
                error=f"tool {step_name} 타임아웃 ({self._tool_timeout}초)",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            logger.error(f"[AgentLoop] tool {step_name} 실행 오류: {exc}", exc_info=True)
            return ToolResult(
                tool=step.tool,
                success=False,
                error=str(exc),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    @staticmethod
    def _build_tool_log_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
        """tool log에 남길 작은 preview만 보관한다."""
        metadata: Dict[str, Any] = {}
        if "count" in data:
            metadata["count"] = data["count"]
        if "query" in data:
            metadata["query"] = data["query"]
        if "results" in data and isinstance(data["results"], list):
            metadata["result_count"] = len(data["results"])
        if "text" in data:
            metadata["text_preview"] = str(data["text"])[:200]
        return metadata

    @staticmethod
    def _build_plan_summary(plan: Optional[ExecutionPlan]) -> str:
        if not plan:
            return ""

        tools = " -> ".join(step.step_id for step in plan.steps)
        if plan.reason:
            return f"{plan.reason} | tools: {tools}"
        return tools

    @staticmethod
    def _graph_run_status(trace: AgentTrace) -> str:
        if trace.error:
            return "failed"
        if any(not result.success for result in trace.tool_results):
            return "completed_with_errors"
        return "completed"

    @classmethod
    def _record_graph_run(
        cls,
        session: SessionContext,
        trace: AgentTrace,
        started_at: float,
        completed_at: float,
    ) -> None:
        success_count = sum(1 for result in trace.tool_results if result.success)
        failure_count = len(trace.tool_results) - success_count
        session.add_graph_run(
            request_id=trace.request_id,
            plan_summary=cls._build_plan_summary(trace.plan),
            approval_status="not_requested",
            executed_capabilities=[tool_name(result.tool) for result in trace.tool_results],
            status=cls._graph_run_status(trace),
            error=trace.error,
            total_latency_ms=trace.total_latency_ms,
            metadata={
                "plan_reason": trace.plan.reason if trace.plan else "",
                "tool_result_count": len(trace.tool_results),
                "success_count": success_count,
                "failure_count": failure_count,
                "final_text_preview": trace.final_text[:200],
            },
            started_at=started_at,
            completed_at=completed_at,
        )

    @staticmethod
    def _extract_final_text(accumulated: Dict[str, Any], plan: ExecutionPlan) -> str:
        for tool_type in (ToolType.APPEND_EVIDENCE, ToolType.DRAFT_CIVIL_RESPONSE):
            payload = accumulated.get(tool_type.value, {})
            if isinstance(payload, dict) and payload.get("text"):
                return str(payload["text"])

        for step in plan.steps:
            payload = accumulated.get(step.step_id, {})
            if isinstance(payload, dict) and payload.get("text"):
                return str(payload["text"])

        parts: List[str] = []

        rag_data = accumulated.get(ToolType.RAG_SEARCH.value, {})
        if rag_data.get("results"):
            lines = ["[로컬 문서 근거]"]
            for item in rag_data["results"][:3]:
                title = item.get("title", "")
                content = item.get("content", "")[:120]
                lines.append(f"- {title}: {content}")
            parts.append("\n".join(lines))

        api_data = accumulated.get(ToolType.API_LOOKUP.value, {})
        if api_data.get("context_text"):
            parts.append(api_data["context_text"])
        elif api_data.get("results"):
            lines = ["[외부 조회 결과]"]
            for item in api_data["results"][:3]:
                title = item.get("title", item.get("qnaTitle", ""))
                content = item.get("content", item.get("qnaContent", ""))[:120]
                lines.append(f"- {title}: {content}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else "요청을 처리할 수 없습니다."
