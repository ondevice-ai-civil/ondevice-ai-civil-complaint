"""ToolPlanValidator 단위 테스트.

Issue #410: planner schema 검증 및 LangGraph 상태 전이 규칙 구현.
"""

from __future__ import annotations

import pytest

from src.inference.graph.plan_validator import (
    MVP_CAPABILITIES,
    PlanValidationError,
    ToolPlanValidator,
)
from src.inference.graph.state import TaskType, ToolPlan


@pytest.fixture
def validator() -> ToolPlanValidator:
    return ToolPlanValidator()


@pytest.fixture
def valid_plan() -> ToolPlan:
    return ToolPlan(
        task_type=TaskType.DRAFT_RESPONSE,
        goal="민원 답변 초안을 작성합니다.",
        reason="사용자가 민원 답변을 요청했습니다.",
        tools=["rag_search", "draft_civil_response"],
    )


class TestToolPlanValidator:
    """ToolPlanValidator 검증 규칙 테스트."""

    def test_valid_plan_passes(self, validator: ToolPlanValidator, valid_plan: ToolPlan) -> None:
        """정상 plan이 그대로 반환된다."""
        result = validator.validate(valid_plan)
        assert result is valid_plan

    def test_empty_tools_raises(self, validator: ToolPlanValidator) -> None:
        """tools=[] 시 PlanValidationError가 발생한다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="테스트 목표",
            reason="테스트 이유",
            tools=[],
        )
        with pytest.raises(PlanValidationError, match="planned_tools가 비어있습니다"):
            validator.validate(plan)

    def test_unknown_tool_raises(self, validator: ToolPlanValidator) -> None:
        """MVP에 없는 tool이 포함되면 PlanValidationError가 발생한다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="테스트 목표",
            reason="테스트 이유",
            tools=["rag_search", "unknown_tool"],
        )
        with pytest.raises(PlanValidationError, match="허용되지 않은 capability"):
            validator.validate(plan)

    def test_empty_goal_raises(self, validator: ToolPlanValidator) -> None:
        """goal이 빈 문자열이면 PlanValidationError가 발생한다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="",
            reason="테스트 이유",
            tools=["rag_search"],
        )
        with pytest.raises(PlanValidationError, match="goal이 비어있습니다"):
            validator.validate(plan)

    def test_empty_reason_raises(self, validator: ToolPlanValidator) -> None:
        """reason이 빈 문자열이면 PlanValidationError가 발생한다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="테스트 목표",
            reason="",
            tools=["rag_search"],
        )
        with pytest.raises(PlanValidationError, match="reason이 비어있습니다"):
            validator.validate(plan)

    def test_whitespace_goal_raises(self, validator: ToolPlanValidator) -> None:
        """공백만 있는 goal도 PlanValidationError가 발생한다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="   ",
            reason="테스트 이유",
            tools=["rag_search"],
        )
        with pytest.raises(PlanValidationError, match="goal이 비어있습니다"):
            validator.validate(plan)

    def test_whitespace_reason_raises(self, validator: ToolPlanValidator) -> None:
        """공백만 있는 reason도 PlanValidationError가 발생한다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="테스트 목표",
            reason="   ",
            tools=["rag_search"],
        )
        with pytest.raises(PlanValidationError, match="reason이 비어있습니다"):
            validator.validate(plan)

    def test_fallback_plan_structure(self, validator: ToolPlanValidator) -> None:
        """make_fallback_plan이 올바른 dict 구조를 반환한다."""
        error = PlanValidationError("테스트 에러")
        fallback = validator.make_fallback_plan(error)

        assert "error" in fallback
        assert "planner validation 실패" in fallback["error"]
        assert "테스트 에러" in fallback["error"]
        assert fallback["planned_tools"] == []
        assert fallback["goal"] == ""
        assert fallback["reason"] == ""

    def test_all_mvp_capabilities_allowed(self, validator: ToolPlanValidator) -> None:
        """4개 MVP capability가 모두 통과한다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="모든 MVP capability 테스트",
            reason="전체 capability 검증",
            tools=list(MVP_CAPABILITIES),
        )
        result = validator.validate(plan)
        assert result is plan

    def test_adapter_mode_default(self) -> None:
        """ToolPlan의 adapter_mode 기본값이 'regex'이다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="테스트",
            reason="테스트",
            tools=["rag_search"],
        )
        assert plan.adapter_mode == "regex"

    def test_adapter_mode_custom(self) -> None:
        """ToolPlan의 adapter_mode를 'llm'으로 설정할 수 있다."""
        plan = ToolPlan(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="테스트",
            reason="테스트",
            tools=["rag_search"],
            adapter_mode="llm",
        )
        assert plan.adapter_mode == "llm"
