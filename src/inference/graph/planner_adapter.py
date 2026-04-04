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
from typing import Any, Dict, Sequence

from langchain_core.messages import AnyMessage

from .capabilities.registry import get_mvp_capability_ids
from .state import TaskType, ToolPlan


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

    AVAILABLE_TOOLS = sorted(get_mvp_capability_ids())

    SYSTEM_PROMPT = (
        "당신은 GovOn 민원 답변 보조 시스템의 작업 계획기입니다.\n"
        "사용자의 요청을 분석하여 다음 JSON 형식으로 실행 계획을 출력하세요:\n\n"
        '{"task_type": "<draft_response|revise_response|append_evidence|lookup_stats>",\n'
        ' "goal": "<사용자에게 보여줄 작업 설명 (한국어, 1-2문장)>",\n'
        ' "reason": "<이 작업이 필요한 이유 (한국어, 1문장)>",\n'
        ' "tools": ["<tool1>", "<tool2>", ...]}\n\n'
        "사용 가능한 도구: rag_search, api_lookup, draft_civil_response, append_evidence\n"
        "규칙:\n"
        "- draft_response: rag_search, api_lookup, draft_civil_response 순서\n"
        "- revise_response: rag_search, api_lookup, draft_civil_response 순서\n"
        "- append_evidence: rag_search, api_lookup, append_evidence 순서\n"
        "- lookup_stats: api_lookup 단독\n"
        "- JSON만 출력하세요. 다른 텍스트 없이.\n"
    )

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        """LLM을 호출하여 실행 계획을 생성한다."""
        from langchain_core.messages import HumanMessage, SystemMessage

        plan_messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_prompt(messages, context)),
        ]

        response = await self._llm.ainvoke(plan_messages)
        parsed = json.loads(response.content)

        return ToolPlan(
            task_type=TaskType(parsed["task_type"]),
            goal=parsed["goal"],
            reason=parsed["reason"],
            tools=parsed["tools"],
        )

    @staticmethod
    def _build_user_prompt(
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> str:
        """LLM에 전달할 사용자 프롬프트를 구성한다."""
        parts = []
        if context.get("session_context"):
            parts.append(f"[세션 맥락]\n{context['session_context']}")
        user_query = messages[-1].content if messages else ""
        parts.append(f"[사용자 요청]\n{user_query}")
        return "\n\n".join(parts)


class RegexPlannerAdapter(PlannerAdapter):
    """기존 정규식 ToolRouter를 PlannerAdapter 인터페이스로 래핑.

    LLM planner가 실패하거나 사용 불가할 때 fallback으로 동작한다.
    smoke test에서도 LLM 없이 사용한다.
    기존 `src.inference.tool_router.ToolRouter`의 로직을 그대로 재사용한다.
    """

    def __init__(self) -> None:
        from src.inference.tool_router import ToolRouter

        self._router = ToolRouter()

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

        return ToolPlan(
            task_type=task_type,
            goal=f"요청 처리: {execution_plan.reason}",
            reason=execution_plan.reason,
            tools=execution_plan.tool_names,
        )

    @staticmethod
    def _infer_task_type(tool_names: list[str]) -> TaskType:
        """tool 이름 목록에서 TaskType을 추론한다."""
        if "append_evidence" in tool_names:
            return TaskType.APPEND_EVIDENCE
        if "draft_civil_response" in tool_names:
            return TaskType.DRAFT_RESPONSE
        if tool_names == ["api_lookup"]:
            return TaskType.LOOKUP_STATS
        return TaskType.DRAFT_RESPONSE
