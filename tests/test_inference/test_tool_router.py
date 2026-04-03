"""GovOn MVP ToolRouter 단위 테스트."""

from src.inference.tool_router import ExecutionPlan, ToolRouter, ToolStep, ToolType


class TestToolRouter:
    def setup_method(self):
        self.router = ToolRouter()

    def test_default_request_routes_to_drafting_loop(self):
        plan = self.router.plan("도로 포장이 파손되어 위험합니다")
        assert plan.tool_names == ["rag_search", "api_lookup", "draft_civil_response"]
        assert "drafting loop" in plan.reason or "답변 작성" in plan.reason

    def test_evidence_request_routes_to_append_flow(self):
        plan = self.router.plan("이 답변의 근거를 붙여줘")
        assert plan.tool_names == ["rag_search", "api_lookup", "append_evidence"]
        assert "근거" in plan.reason

    def test_lookup_only_request_routes_to_api_lookup(self):
        plan = self.router.plan("민원 통계와 최근 이슈를 조회해줘")
        assert plan.tool_names == ["api_lookup"]

    def test_revision_request_keeps_drafting_loop(self):
        plan = self.router.plan("조금 더 정중하게 다시 써줘", has_context=True)
        assert plan.tool_names == ["rag_search", "api_lookup", "draft_civil_response"]

    def test_force_tools_overrides_pattern_matching(self):
        plan = self.router.plan("무시", force_tools=[ToolType.API_LOOKUP])
        assert plan.tool_names == ["api_lookup"]
        assert "강제 지정" in plan.reason

    def test_force_tools_accepts_custom_tool_name(self):
        plan = self.router.plan("무시", force_tools=["custom_lookup"])
        assert plan.tool_names == ["custom_lookup"]

    def test_execution_plan_repr_contains_step_names(self):
        plan = ExecutionPlan(
            steps=[ToolStep(tool=ToolType.RAG_SEARCH), ToolStep(tool=ToolType.API_LOOKUP)],
            reason="테스트",
        )
        assert "rag_search" in repr(plan)


class TestToolStep:
    def test_step_id_for_enum_tool(self):
        step = ToolStep(tool=ToolType.RAG_SEARCH)
        assert step.step_id == "rag_search"

    def test_step_id_for_custom_tool(self):
        step = ToolStep(tool="custom_tool")
        assert step.step_id == "custom_tool"

    def test_depends_on_is_preserved(self):
        step = ToolStep(tool=ToolType.APPEND_EVIDENCE, depends_on="api_lookup")
        assert step.depends_on == "api_lookup"
