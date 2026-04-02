"""ToolRouter 단위 테스트.

Issue: #393
"""

import pytest

from src.inference.tool_router import ExecutionPlan, ToolRouter, ToolStep, ToolType


class TestToolRouter:
    """ToolRouter 단위 테스트."""

    def setup_method(self):
        self.router = ToolRouter()

    def test_full_pipeline_no_keywords(self):
        """키워드 없는 민원 텍스트는 전체 파이프라인 실행."""
        plan = self.router.plan("도로 포장이 파손되어 보행자 안전에 위험합니다")
        assert plan.tool_names == ["classify", "search_similar", "generate_civil_response"]
        assert "전체 파이프라인" in plan.reason

    def test_classify_keyword(self):
        """분류 키워드만 매칭."""
        plan = self.router.plan("이 민원을 분류해주세요")
        assert ToolType.CLASSIFY.value in plan.tool_names

    def test_search_keyword(self):
        """검색 키워드만 매칭."""
        plan = self.router.plan("유사 민원 사례를 검색해주세요")
        assert ToolType.SEARCH_SIMILAR.value in plan.tool_names

    def test_generate_civil_response_keyword(self):
        """민원 답변 생성 키워드만 매칭."""
        plan = self.router.plan("답변을 작성해주세요")
        assert ToolType.GENERATE_CIVIL_RESPONSE.value in plan.tool_names

    def test_generate_public_doc_keyword(self):
        """공문서 생성 키워드만 매칭."""
        plan = self.router.plan("보도자료 초안을 작성해주세요")
        assert ToolType.GENERATE_PUBLIC_DOC.value in plan.tool_names

    def test_multiple_keywords(self):
        """여러 키워드가 매칭되면 해당 tool들만 실행."""
        plan = self.router.plan("이 민원을 분류하고 유사 사례를 검색해주세요")
        assert "classify" in plan.tool_names
        assert "search_similar" in plan.tool_names

    def test_force_tools(self):
        """force_tools 지정 시 패턴 매칭을 건너뜀."""
        plan = self.router.plan(
            "아무 텍스트",
            force_tools=[ToolType.SEARCH_SIMILAR],
        )
        assert plan.tool_names == ["search_similar"]
        assert "강제 지정" in plan.reason

    def test_force_tools_multiple(self):
        """여러 tool 강제 지정."""
        plan = self.router.plan(
            "텍스트",
            force_tools=[ToolType.CLASSIFY, ToolType.GENERATE_PUBLIC_DOC],
        )
        assert plan.tool_names == ["classify", "generate_public_doc"]

    def test_force_tools_accepts_custom_tool_name(self):
        """custom registry tool 이름도 force_tools로 전달할 수 있다."""
        plan = self.router.plan("텍스트", force_tools=["custom_lookup"])
        assert plan.tool_names == ["custom_lookup"]

    def test_search_only_does_not_add_generate(self):
        """검색만 요청 시 생성 도구는 추가하지 않음."""
        plan = self.router.plan("관련 사례를 찾아주세요")
        assert "search_similar" in plan.tool_names
        assert "generate_public_doc" not in plan.tool_names
        assert "generate_civil_response" not in plan.tool_names

    def test_plan_has_reason(self):
        """모든 계획에는 reason이 있어야 함."""
        plan = self.router.plan("아무 텍스트")
        assert plan.reason

    def test_execution_plan_repr(self):
        """ExecutionPlan repr 동작 확인."""
        plan = ExecutionPlan(
            steps=[ToolStep(tool=ToolType.CLASSIFY)],
            reason="test",
        )
        assert "classify" in repr(plan)


class TestToolRouterApiLookup:
    """API_LOOKUP 관련 ToolRouter 테스트."""

    def setup_method(self):
        self.router = ToolRouter()

    def test_api_lookup_minwon_analysis(self):
        """'민원 분석' 키워드 → api_lookup 포함."""
        plan = self.router.plan("민원 분석해줘")
        assert ToolType.API_LOOKUP.value in plan.tool_names

    def test_api_lookup_national_sinmungo(self):
        """'국민신문고' 키워드 → api_lookup 포함."""
        plan = self.router.plan("국민신문고 유사 사례 보여줘")
        assert ToolType.API_LOOKUP.value in plan.tool_names

    def test_api_lookup_statistics(self):
        """'민원 통계' 키워드 → api_lookup 포함."""
        plan = self.router.plan("민원 통계 조회해줘")
        assert ToolType.API_LOOKUP.value in plan.tool_names

    def test_api_lookup_does_not_break_full_pipeline(self):
        """일반 민원 텍스트는 여전히 기본 3단계 파이프라인으로 실행."""
        plan = self.router.plan("보도블록이 파손되어 위험합니다")
        assert plan.tool_names == ["classify", "search_similar", "generate_civil_response"]
        assert ToolType.API_LOOKUP.value not in plan.tool_names

    def test_api_lookup_with_classify_ordering(self):
        """classify + api_lookup 동시 매칭 시 classify가 먼저 배치."""
        plan = self.router.plan("이 민원을 분류하고 민원 현황도 조회해줘")
        names = plan.tool_names
        assert "classify" in names
        assert "api_lookup" in names
        assert names.index("classify") < names.index("api_lookup")

    def test_api_lookup_depends_on_classify(self):
        """classify + api_lookup 동시 매칭 시 api_lookup의 depends_on이 'classify'."""
        plan = self.router.plan("이 민원을 분류하고 민원 통계 조회해줘")
        api_step = next(s for s in plan.steps if s.tool == ToolType.API_LOOKUP)
        assert api_step.depends_on == "classify"


class TestToolStep:
    """ToolStep 단위 테스트."""

    def test_step_id(self):
        step = ToolStep(tool=ToolType.CLASSIFY)
        assert step.step_id == "classify"

    def test_step_with_depends(self):
        step = ToolStep(tool=ToolType.SEARCH_SIMILAR, depends_on="classify")
        assert step.depends_on == "classify"

    def test_api_lookup_step_id(self):
        """API_LOOKUP step_id 확인."""
        step = ToolStep(tool=ToolType.API_LOOKUP)
        assert step.step_id == "api_lookup"
