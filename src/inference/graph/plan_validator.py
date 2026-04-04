"""ToolPlan 검증기.

Issue #410: planner schema 검증 및 LangGraph 상태 전이 규칙 구현.

planner_node가 반환한 ToolPlan을 graph 실행 전에 검증한다.
MVP에서 허용되는 capability 목록과 필수 필드를 확인하며,
검증 실패 시 PlanValidationError를 raise한다.
"""

from __future__ import annotations

from .state import TaskType, ToolPlan

MVP_CAPABILITIES: frozenset[str] = frozenset(
    [
        "rag_search",
        "api_lookup",
        "draft_civil_response",
        "append_evidence",
    ]
)


class PlanValidationError(ValueError):
    """planner 출력 검증 실패."""

    pass


class ToolPlanValidator:
    """ToolPlan 검증기.

    planner_node가 반환한 ToolPlan을 graph 실행 전에 검증한다.
    validation 실패 시 PlanValidationError를 raise한다.
    """

    def validate(self, plan: ToolPlan) -> ToolPlan:
        """plan을 검증하고 유효한 경우 그대로 반환한다.

        Parameters
        ----------
        plan : ToolPlan
            planner adapter가 생성한 실행 계획.

        Returns
        -------
        ToolPlan
            검증 통과한 plan (그대로 반환).

        Raises
        ------
        PlanValidationError
            검증 실패 시 구체적인 사유와 함께 raise.
        """
        # 1. tools가 비어있으면 실행할 것이 없다
        if not plan.tools:
            raise PlanValidationError("planned_tools가 비어있습니다")

        # 2. MVP capability 화이트리스트 검사
        unknown_tools = set(plan.tools) - MVP_CAPABILITIES
        if unknown_tools:
            raise PlanValidationError(f"허용되지 않은 capability: {unknown_tools}")

        # 3. goal 필수
        if not plan.goal.strip():
            raise PlanValidationError("goal이 비어있습니다")

        # 4. reason 필수
        if not plan.reason.strip():
            raise PlanValidationError("reason이 비어있습니다")

        # 5. task_type이 유효한 TaskType 값인지 확인
        if not isinstance(plan.task_type, TaskType):
            try:
                TaskType(plan.task_type)
            except ValueError:
                raise PlanValidationError(f"유효하지 않은 task_type: {plan.task_type}")

        return plan

    def make_fallback_plan(self, error: PlanValidationError) -> dict:
        """validation 실패 시 graph state에 설정할 error dict를 반환한다.

        Parameters
        ----------
        error : PlanValidationError
            검증 실패 사유.

        Returns
        -------
        dict
            graph state에 병합할 fallback dict.
        """
        return {
            "error": f"planner validation 실패: {error}",
            "planned_tools": [],
            "goal": "",
            "reason": "",
        }
