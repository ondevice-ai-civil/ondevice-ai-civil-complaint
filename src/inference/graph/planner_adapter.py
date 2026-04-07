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

from .capabilities.registry import build_tool_definitions, get_mvp_capability_ids
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

    네이티브 tool calling을 1차로 시도하고, tool_calls가 없으면
    텍스트 JSON 파싱으로 fallback한다.

    Parameters
    ----------
    llm : BaseChatModel
        langchain-openai ChatOpenAI 또는 호환 LLM.
    """

    @staticmethod
    def _build_tool_definitions(registry: "dict[str, Any] | None") -> list[dict]:
        """registry에서 OpenAI-compatible tool definitions를 생성한다."""
        if registry:
            return build_tool_definitions(registry)
        # registry 없으면 이름만으로 minimal definition 생성
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
            for name in sorted(get_mvp_capability_ids())
        ]

    @staticmethod
    def _build_system_prompt() -> str:
        """네이티브 tool calling용 system prompt를 생성한다."""
        return (
            "당신은 GovOn 민원 답변 보조 시스템의 작업 계획기입니다.\n"
            "사용자의 요청을 분석하여 적절한 도구를 호출하세요.\n\n"
            "규칙:\n"
            "- 민원 답변 작성: draft_response + rag_search + api_lookup 동시 호출\n"
            "- 답변 수정: draft_response + rag_search + api_lookup 동시 호출\n"
            "- 통계 조회: api_lookup 단독 호출\n"
            "- 이슈 탐지: issue_detector 단독 호출\n"
            "- 통계 분석: stats_lookup 단독 또는 stats_lookup + issue_detector 조합\n"
            "- 키워드 분석: keyword_analyzer 단독 호출\n"
            "- 인구통계 조회: demographics_lookup 단독 호출\n"
            "- draft_response는 답변 생성의 핵심 도구입니다. 민원/법률 관련 요청에는 반드시 포함하세요.\n"
            "- 필요한 도구를 모두 호출하세요.\n"
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

        1차: bind_tools()로 네이티브 tool calling 시도.
        2차: tool_calls가 없으면 텍스트 JSON 파싱으로 fallback.
        모든 실패 시 PlanValidationError를 raise한다.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        tool_definitions = self._build_tool_definitions(self._registry)

        plan_messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=self._build_user_prompt(messages, context)),
        ]

        try:
            # 1차: 네이티브 tool calling 시도
            llm_with_tools = self._llm.bind_tools(tool_definitions)
            response = await llm_with_tools.ainvoke(plan_messages)

            # tool_calls 파싱
            tool_calls = getattr(response, "tool_calls", None) or []
            if tool_calls:
                tools: list[str] = []
                tool_args: Dict[str, Dict[str, Any]] = {}
                for tc in tool_calls:
                    name = tc["name"] if isinstance(tc, dict) else tc.name
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    # args가 dict가 아니면 빈 dict로 정규화
                    if not isinstance(args, dict):
                        logger.warning(
                            f"[LLMPlanner] tool_call args가 dict가 아님: {name}, type={type(args).__name__}"
                        )
                        args = {}
                    # 중복 tool name은 건너뜀 (첫 번째 호출만 사용)
                    if name in tool_args:
                        logger.warning(f"[LLMPlanner] 중복 tool call 무시: {name}")
                        continue
                    tools.append(name)
                    if args:
                        tool_args[name] = args

                task_type = RegexPlannerAdapter._infer_task_type(tools)
                goal_text = tool_args.get(tools[0], {}).get("query", "") if tools else ""

                # Safety net: draft_response/revise_response task에 핵심 도구 자동 보충
                if task_type in (TaskType.DRAFT_RESPONSE, TaskType.REVISE_RESPONSE):
                    if "draft_response" not in tools:
                        tools.append("draft_response")
                        logger.info("[Planner] draft_response 자동 보충 (핵심 도구)")

                return ToolPlan(
                    task_type=task_type,
                    goal=f"요청 처리: {goal_text[:100]}" if goal_text else "요청을 처리합니다.",
                    reason="LLM이 네이티브 tool calling으로 도구를 선택했습니다.",
                    tools=tools,
                    tool_args=tool_args,
                    tool_summaries=_build_tool_summaries(tools, self._registry),
                    adapter_mode="llm_tool_calling",
                )

            # 2차: fallback — 텍스트 JSON 파싱
            content = str(response.content or "")
            parsed = json.loads(content)
            tools_list: list[str] = parsed["tools"]
            return ToolPlan(
                task_type=TaskType(parsed["task_type"]),
                goal=parsed["goal"],
                reason=parsed["reason"],
                tools=tools_list,
                tool_args={},
                tool_summaries=_build_tool_summaries(tools_list, self._registry),
                adapter_mode="llm",
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            from .plan_validator import PlanValidationError

            logger.warning(f"[LLMPlanner] 응답 파싱 실패: {exc}")
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


class DirectEnginePlannerAdapter(PlannerAdapter):
    """vLLM engine을 직접 호출하는 planner.

    self-call HTTP 오버헤드를 제거하고 engine 인스턴스를 직접 참조한다.
    운영 환경 기본 planner로 사용된다.

    Hermes tool calling 포맷(<tool_call> 태그)을 1차로 파싱하고,
    태그가 없으면 텍스트 JSON 파싱으로 fallback한다.
    """

    def __init__(self, engine_manager: Any, registry: Optional[Dict[str, Any]] = None) -> None:
        self._engine_manager = engine_manager
        self._registry = registry

    @staticmethod
    def _parse_hermes_tool_calls(text: str) -> list[dict]:
        """Hermes 포맷의 <tool_call> 태그를 파싱한다.

        형식: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        """
        import re

        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)
        results: list[dict] = []
        for match in matches:
            try:
                parsed = json.loads(match)
                if "name" in parsed:
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
        return results

    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        from .plan_validator import PlanValidationError

        tool_definitions = LLMPlannerAdapter._build_tool_definitions(self._registry)
        system_prompt = LLMPlannerAdapter._build_system_prompt()
        user_prompt = LLMPlannerAdapter._build_user_prompt(messages, context)

        # OpenAI function calling 형식의 tools (EXAONE chat template 호환)
        tools_for_template = tool_definitions

        # EXAONE chat template으로 프롬프트 생성
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # vLLM engine에서 토크나이저 획득
            if self._engine_manager.engine is not None:
                tokenizer = self._engine_manager.engine.get_tokenizer()
            else:
                raise RuntimeError("engine is None (SKIP_MODEL_LOAD=true?)")
            prompt = tokenizer.apply_chat_template(
                chat_messages,
                tools=tools_for_template,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.info("[DirectEnginePlanner] EXAONE chat template으로 프롬프트 생성 완료")
        except Exception as exc:
            logger.warning(
                f"[DirectEnginePlanner] tokenizer chat template 실패, 텍스트 JSON fallback: {exc}"
            )
            # fallback: 기존 텍스트 JSON 방식
            prompt = (
                f"[|system|]{system_prompt}[|endofturn|]\n"
                f"[|user|]{user_prompt}[|endofturn|]\n"
                "[|assistant|]"
            )

        try:
            from vllm import SamplingParams as _SamplingParams
        except ImportError:
            raise PlanValidationError("vLLM이 설치되어 있지 않습니다.")

        sampling_params = _SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop=["[|endofturn|]"],
        )

        import uuid as _uuid

        request_id = str(_uuid.uuid4())
        try:
            output = await self._engine_manager._run_engine(prompt, sampling_params, request_id)
        except Exception as exc:
            raise PlanValidationError(f"Engine 호출 실패: {exc}") from exc

        if output is None or not output.outputs:
            raise PlanValidationError("Engine 출력이 비어 있음")

        # raw output 로깅 (디버깅용)
        raw_text = output.outputs[0].text
        logger.info(f"[DirectEnginePlanner] raw output ({len(raw_text)} chars): {raw_text[:300]}")
        content = self._engine_manager._strip_thought_blocks(raw_text)
        logger.info(
            f"[DirectEnginePlanner] stripped content ({len(content)} chars): {content[:300]}"
        )

        # 1차: <tool_call> 태그 파싱
        tool_calls = self._parse_hermes_tool_calls(content)
        if tool_calls:
            tools: list[str] = []
            tool_args: Dict[str, Dict[str, Any]] = {}
            for tc in tool_calls:
                tc_name = tc["name"]
                args = tc.get("arguments", {})
                # args가 dict가 아니면 빈 dict로 정규화
                if not isinstance(args, dict):
                    logger.warning(
                        f"[DirectEnginePlanner] tool_call args가 dict가 아님: {tc_name}, type={type(args).__name__}"
                    )
                    args = {}
                # 중복 tool name은 건너뜀 (첫 번째 호출만 사용)
                if tc_name in tool_args:
                    logger.warning(f"[DirectEnginePlanner] 중복 tool call 무시: {tc_name}")
                    continue
                tools.append(tc_name)
                if args:
                    tool_args[tc_name] = args
            task_type = RegexPlannerAdapter._infer_task_type(tools)
            goal_text = tool_args.get(tools[0], {}).get("query", "") if tools else ""

            # Safety net: draft_response/revise_response task에 핵심 도구 자동 보충
            if task_type in (TaskType.DRAFT_RESPONSE, TaskType.REVISE_RESPONSE):
                if "draft_response" not in tools:
                    tools.append("draft_response")
                    logger.info("[Planner] draft_response 자동 보충 (핵심 도구)")

            return ToolPlan(
                task_type=task_type,
                goal=f"요청 처리: {goal_text[:100]}" if goal_text else "요청을 처리합니다.",
                reason="LLM이 네이티브 tool calling으로 도구를 선택했습니다.",
                tools=tools,
                tool_args=tool_args,
                tool_summaries=_build_tool_summaries(tools, self._registry),
                adapter_mode="direct_engine_tool_calling",
            )

        # 2차: fallback — 텍스트 JSON 파싱
        try:
            parsed = json.loads(content)
            tools_list: list[str] = parsed["tools"]
            return ToolPlan(
                task_type=TaskType(parsed["task_type"]),
                goal=parsed["goal"],
                reason=parsed["reason"],
                tools=tools_list,
                tool_args={},
                tool_summaries=_build_tool_summaries(tools_list, self._registry),
                adapter_mode="direct_engine",
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            raise PlanValidationError(f"응답 파싱 실패: {exc}") from exc


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
        if "draft_response" in tool_names:
            return TaskType.DRAFT_RESPONSE
        if tool_names == ["api_lookup"]:
            return TaskType.LOOKUP_STATS
        return TaskType.DRAFT_RESPONSE
