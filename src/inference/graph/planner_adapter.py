"""Planner adapter: 사용자 요청을 구조화된 실행 계획으로 변환.

Issue #415: LangGraph runtime 기반 및 planner/executor adapter 구성.
Issue #416: AVAILABLE_TOOLS를 registry 단일 소스에서 가져온다.

두 가지 구현체를 제공한다:
- `LLMPlannerAdapter`: LLM(ChatOpenAI 또는 호환 모델) 기반 planner
- `RegexPlannerAdapter`: 기존 `ToolRouter` 정규식 로직을 래핑한 fallback planner
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import AnyMessage
from loguru import logger

from .capabilities.registry import get_mvp_capability_ids
from .state import TaskType, ToolPlan

# ---------------------------------------------------------------------------
# 내부 헬퍼: registry에서 tool_summaries를 조회한다
# ---------------------------------------------------------------------------


def _build_tool_summaries(
    tool_names: list[str],
    registry: "dict[str, Any] | None",
) -> list[str]:
    """registry에서 각 tool의 approval_summary를 조회하여 반환한다.

    registry가 없거나 tool이 registry에 없으면 tool 이름 그대로 반환한다.

    Parameters
    ----------
    tool_names : list[str]
        planned tool 이름 목록.
    registry : dict | None
        CapabilityBase 인스턴스가 담긴 registry. None이면 이름 그대로 반환.

    Returns
    -------
    list[str]
        각 tool의 human-readable approval_summary 목록.
    """
    summaries: list[str] = []
    for name in tool_names:
        if registry and name in registry:
            cap = registry[name]
            summaries.append(cap.metadata.approval_summary)
        else:
            summaries.append(name)
    return summaries


class PlannerAdapter(ABC):
    """Planner 추상 인터페이스.

    모든 planner 구현체는 이 인터페이스를 따른다.
    LangGraph graph의 `planner` 노드에서 호출된다.
    """

    @abstractmethod
    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        """사용자 메시지와 컨텍스트를 받아 실행 계획을 반환한다.

        Parameters
        ----------
        messages : Sequence[AnyMessage]
            LangGraph state의 message history.
        context : Dict[str, Any]
            accumulated_context (세션 요약, 이전 tool 결과 등).

        Returns
        -------
        ToolPlan
            task_type, goal, reason, tools를 포함한 구조화된 계획.
        """
        ...


class LLMPlannerAdapter(PlannerAdapter):
    """LLM 기반 planner.

    langchain-openai ChatOpenAI (또는 호환 모델)를 사용하여
    사용자 요청을 분석하고 ToolPlan을 생성한다.
    로컬 vLLM을 OpenAI-compatible endpoint로 연결 가능.

    Parameters
    ----------
    llm : BaseChatModel
        langchain-openai ChatOpenAI 또는 호환 LLM.
    """

    @staticmethod
    def _build_system_prompt() -> str:
        """사용 가능한 도구 목록을 동적으로 반영한 system prompt를 생성한다."""
        tools = ", ".join(sorted(get_mvp_capability_ids()))
        return (
            "당신은 GovOn 민원 답변 보조 시스템의 작업 계획기입니다.\n"
            "사용자의 요청을 분석하여 다음 JSON 형식으로 실행 계획을 출력하세요:\n\n"
            '{"task_type": "<draft_response|revise_response|append_evidence|lookup_stats>",\n'
            ' "goal": "<사용자에게 보여줄 작업 설명 (한국어, 1-2문장)>",\n'
            ' "reason": "<이 작업이 필요한 이유 (한국어, 1문장)>",\n'
            ' "tools": ["<tool1>", "<tool2>", ...]}\n\n'
            f"사용 가능한 도구: {tools}\n"
            "규칙:\n"
            "- draft_response: rag_search, api_lookup, draft_civil_response 순서\n"
            "- revise_response: rag_search, api_lookup, draft_civil_response 순서\n"
            "- append_evidence: rag_search, api_lookup, append_evidence 순서\n"
            "- lookup_stats: api_lookup 단독\n"
            "- issue_detection: issue_detector 단독\n"
            "- stats_query: stats_lookup 단독 또는 stats_lookup, issue_detector 조합\n"
            "- keyword_analysis: keyword_analyzer 단독\n"
            "- demographics_query: demographics_lookup 단독\n"
            "- JSON만 출력하세요. 다른 텍스트 없이.\n"
        )

    def __init__(self, llm: Any, registry: Optional[Dict[str, Any]] = None) -> None:
        self._llm = llm
        self._registry = registry

    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        """LLM을 호출하여 실행 계획을 생성한다.

        LLM 호출 실패 또는 JSON 파싱 실패 시 PlanValidationError를 raise하여
        planner_node의 fallback 핸들러가 처리하도록 한다.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        plan_messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=self._build_user_prompt(messages, context)),
        ]

        try:
            response = await self._llm.ainvoke(plan_messages)
            content = str(response.content or "")
            parsed = json.loads(content)
            tools: list[str] = parsed["tools"]
            return ToolPlan(
                task_type=TaskType(parsed["task_type"]),
                goal=parsed["goal"],
                reason=parsed["reason"],
                tools=tools,
                tool_summaries=_build_tool_summaries(tools, self._registry),
                adapter_mode="llm",
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            from .plan_validator import PlanValidationError

            logger.warning(f"[LLMPlanner] LLM 응답 파싱 실패: {exc}")
            raise PlanValidationError(f"LLM planner 응답 파싱 실패: {exc}") from exc
        except Exception as exc:
            from .plan_validator import PlanValidationError

            logger.warning(f"[LLMPlanner] LLM 호출 실패: {exc}")
            raise PlanValidationError(f"LLM planner 호출 실패: {exc}") from exc

    @staticmethod
    def _build_user_prompt(
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> str:
        """LLM에 전달할 사용자 프롬프트를 구성한다.

        이전 답변(previous_assistant_response)이 있으면 포함하여
        "이 답변에 근거를 붙여줘" 같은 follow-up intent를 LLM이 정확히 분류하도록 한다.
        """
        parts = []
        if context.get("session_context"):
            parts.append(f"[세션 맥락]\n{context['session_context']}")
        if context.get("previous_assistant_response"):
            # planner는 intent 분류만 하므로 앞 400자로 충분하다.
            prev = str(context["previous_assistant_response"])[:400]
            if len(str(context["previous_assistant_response"])) > 400:
                prev += "… (생략)"
            parts.append(f"[이전 답변]\n{prev}")
        user_query = messages[-1].content if messages else ""
        parts.append(f"[사용자 요청]\n{user_query}")
        return "\n\n".join(parts)


class RegexPlannerAdapter(PlannerAdapter):
    """기존 정규식 ToolRouter를 PlannerAdapter 인터페이스로 래핑.

    LLM planner가 실패하거나 사용 불가할 때 fallback으로 동작한다.
    smoke test에서도 LLM 없이 사용한다.
    기존 `src.inference.tool_router.ToolRouter`의 로직을 그대로 재사용한다.
    """

    def __init__(self, registry: Optional[Dict[str, Any]] = None) -> None:
        from src.inference.tool_router import ToolRouter

        self._router = ToolRouter()
        self._registry = registry

    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        """ToolRouter.plan()으로 실행 계획을 생성하고 ToolPlan으로 변환한다."""
        query = messages[-1].content if messages else ""
        has_context = bool(context.get("session_context"))

        execution_plan = self._router.plan(query, has_context=has_context)

        task_type = self._infer_task_type(execution_plan.tool_names)
        tools: list[str] = execution_plan.tool_names

        return ToolPlan(
            task_type=task_type,
            goal=f"요청 처리: {execution_plan.reason}",
            reason=execution_plan.reason,
            tools=tools,
            tool_summaries=_build_tool_summaries(tools, self._registry),
            adapter_mode="regex",
        )

    @staticmethod
    def _infer_task_type(tool_names: list[str]) -> TaskType:
        """tool 이름 목록에서 TaskType을 추론한다."""
        if "issue_detector" in tool_names:
            return TaskType.ISSUE_DETECTION
        if "stats_lookup" in tool_names:
            return TaskType.STATS_QUERY
        if "keyword_analyzer" in tool_names:
            return TaskType.KEYWORD_ANALYSIS
        if "demographics_lookup" in tool_names:
            return TaskType.DEMOGRAPHICS_QUERY
        if "append_evidence" in tool_names:
            return TaskType.APPEND_EVIDENCE
        if "draft_civil_response" in tool_names:
            return TaskType.DRAFT_RESPONSE
        if tool_names == ["api_lookup"]:
            return TaskType.LOOKUP_STATS
        return TaskType.DRAFT_RESPONSE
