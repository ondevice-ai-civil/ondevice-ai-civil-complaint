"""context-aware query builder 단위 테스트."""

from src.inference.query_builder import (
    build_query_variants,
    build_runtime_query_context,
    resolve_tool_query,
)
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


class TestBuildQueryVariants:
    def test_follow_up_queries_include_original_question_and_existing_draft(self):
        context = {
            "previous_user_query": "도로 포장이 파손되어 위험합니다",
            "previous_assistant_response": "도로 보수 접수를 진행하겠습니다.",
            "recent_tool_summary": "api_lookup 도로 포장 파손 count 2",
        }

        variants = build_query_variants(
            "이 답변의 근거를 붙여줘",
            tool_names=["rag_search", "api_lookup", "append_evidence"],
            context=context,
        )

        assert "도로 포장이 파손되어 위험합니다" in variants["rag_search"]
        assert "도로 보수 접수를 진행하겠습니다." in variants["rag_search"]
        assert "이 답변의 근거를 붙여줘" in variants["rag_search"]
        assert "관련 법령 지침 매뉴얼 공지 내부 문서" in variants["rag_search"]
        assert "유사 민원 사례 통계 최근 이슈" in variants["api_lookup"]

    def test_query_variants_clip_long_history(self):
        long_assistant = "기존 초안 " + ("상세 설명 " * 80)
        context = {
            "previous_user_query": "원래 민원 요청",
            "previous_assistant_response": long_assistant,
            "recent_tool_summary": "",
        }

        variants = build_query_variants(
            "좀 더 정중하게 수정해줘",
            tool_names=["rag_search", "api_lookup", "draft_civil_response"],
            context=context,
        )

        assert len(variants["rag_search"]) <= 480
        assert len(variants["api_lookup"]) <= 480
        assert long_assistant not in variants["rag_search"]
        assert long_assistant not in variants["api_lookup"]

    def test_resolve_tool_query_uses_variant_only_for_registered_tool(self):
        context = {
            "query": "원본 요청",
            "query_variants": {
                "rag_search": "원본 요청 관련 법령",
                "api_lookup": "원본 요청 유사 민원 사례",
            },
        }

        assert resolve_tool_query("rag_search", context) == "원본 요청 관련 법령"
        assert resolve_tool_query("draft_civil_response", context) == "원본 요청"
