"""context-aware query builder 단위 테스트."""

from src.inference.query_builder import build_runtime_query_context
from src.inference.session_context import SessionContext


class TestBuildRuntimeQueryContext:
    def test_extracts_previous_turns_and_recent_tool_summary(self):
        session = SessionContext()
        session.add_turn("user", "원래 민원 요청")
        session.add_turn("assistant", "이전 초안 답변")
        session.add_tool_run(
            "api_lookup",
            success=True,
            metadata={"query": "원래 민원 요청", "count": 3},
        )
        session.add_turn("user", "근거를 더 붙여줘")

        context = build_runtime_query_context(session, "근거를 더 붙여줘")

        assert context["previous_user_query"] == "원래 민원 요청"
        assert context["previous_assistant_response"] == "이전 초안 답변"
        assert "api_lookup" in context["recent_tool_summary"]
        assert "count 3" in context["recent_tool_summary"]
