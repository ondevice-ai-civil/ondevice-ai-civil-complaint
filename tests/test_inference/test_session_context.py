"""GovOn MVP SessionContext / SessionStore 테스트."""

from pathlib import Path

from src.inference.session_context import SessionContext, SessionStore


class TestSessionContext:
    def test_create_default(self):
        ctx = SessionContext()
        assert ctx.session_id
        assert ctx.max_history == 20
        assert ctx.conversations == []
        assert ctx.tool_runs == []
        assert ctx.graph_runs == []

    def test_add_turn(self):
        ctx = SessionContext()
        ctx.add_turn("user", "안녕하세요")
        ctx.add_turn("assistant", "무엇을 도와드릴까요?")

        assert len(ctx.conversations) == 2
        assert ctx.conversations[0].role == "user"
        assert ctx.conversations[1].role == "assistant"

    def test_max_history_trim(self):
        ctx = SessionContext(max_history=3)
        for index in range(5):
            ctx.add_turn("user", f"메시지 {index}")

        assert len(ctx.conversations) == 3
        assert ctx.conversations[0].content == "메시지 2"

    def test_add_tool_run(self):
        ctx = SessionContext()
        ctx.add_tool_run(
            "rag_search",
            graph_run_request_id="req-001",
            success=True,
            latency_ms=12.5,
            metadata={"count": 2},
        )

        assert len(ctx.tool_runs) == 1
        assert ctx.tool_runs[0].tool == "rag_search"
        assert ctx.tool_runs[0].graph_run_request_id == "req-001"
        assert ctx.tool_runs[0].metadata["count"] == 2

    def test_build_context_summary_empty(self):
        ctx = SessionContext()
        assert ctx.build_context_summary() == ""

    def test_build_context_summary_with_dialogue_and_tool_log(self):
        ctx = SessionContext()
        ctx.add_turn("user", "민원 답변 작성해줘")
        ctx.add_turn("assistant", "초안을 작성했습니다.")
        ctx.add_tool_run("api_lookup", success=True, metadata={"count": 3})
        ctx.add_graph_run(
            request_id="run-001",
            approval_status="approved",
            executed_capabilities=["api_lookup", "draft_civil_response"],
        )

        summary = ctx.build_context_summary()
        assert "최근 대화" in summary
        assert "최근 도구 실행" in summary
        assert "최근 작업 실행" in summary
        assert "api_lookup" in summary

    def test_add_graph_run(self):
        ctx = SessionContext()
        ctx.add_graph_run(
            request_id="run-001",
            plan_summary="민원 답변 초안 작성",
            approval_status="approved",
            executed_capabilities=["rag_search", "draft_civil_response"],
            status="completed",
        )

        assert len(ctx.graph_runs) == 1
        assert ctx.graph_runs[0].request_id == "run-001"
        assert ctx.graph_runs[0].approval_status == "approved"
        assert ctx.graph_runs[0].executed_capabilities == [
            "rag_search",
            "draft_civil_response",
        ]


class TestSessionStore:
    def test_get_or_create_new(self, tmp_path: Path):
        store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
        ctx = store.get_or_create()

        assert ctx.session_id
        assert store.count == 1

    def test_get_or_create_existing(self, tmp_path: Path):
        store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
        ctx1 = store.get_or_create(session_id="test-session")
        ctx2 = store.get_or_create(session_id="test-session")

        assert ctx1.session_id == ctx2.session_id == "test-session"
        assert store.count == 1

    def test_persists_turns_tool_runs_and_graph_runs(self, tmp_path: Path):
        store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
        ctx = store.get_or_create(session_id="persist-session")
        ctx.add_turn("user", "질문")
        ctx.add_turn("assistant", "답변")
        ctx.add_tool_run(
            "rag_search",
            graph_run_request_id="req-001",
            success=True,
            latency_ms=8.0,
            metadata={"count": 1},
        )
        ctx.add_graph_run(
            request_id="req-001",
            approval_status="approved",
            executed_capabilities=["rag_search"],
            status="completed",
            total_latency_ms=8.0,
        )

        loaded = store.get("persist-session")

        assert loaded is not None
        assert len(loaded.conversations) == 2
        assert loaded.conversations[1].content == "답변"
        assert len(loaded.tool_runs) == 1
        assert loaded.tool_runs[0].tool == "rag_search"
        assert loaded.tool_runs[0].graph_run_request_id == "req-001"
        assert len(loaded.graph_runs) == 1
        assert loaded.graph_runs[0].request_id == "req-001"
        assert loaded.graph_runs[0].approval_status == "approved"

    def test_restart_safe_lookup_restores_graph_runs(self, tmp_path: Path):
        db_path = tmp_path / "sessions.sqlite3"
        store = SessionStore(db_path=str(db_path))
        ctx = store.get_or_create(session_id="resume-session")
        ctx.add_turn("user", "이전 요청")
        ctx.add_graph_run(
            request_id="req-resume",
            approval_status="approved",
            executed_capabilities=["api_lookup", "draft_civil_response"],
            status="completed_with_errors",
        )

        restarted_store = SessionStore(db_path=str(db_path))
        loaded = restarted_store.get("resume-session")

        assert loaded is not None
        assert loaded.session_id == "resume-session"
        assert loaded.conversations[0].content == "이전 요청"
        assert len(loaded.graph_runs) == 1
        assert loaded.graph_runs[0].executed_capabilities == [
            "api_lookup",
            "draft_civil_response",
        ]
        assert loaded.graph_runs[0].status == "completed_with_errors"

    def test_session_metadata_round_trip(self, tmp_path: Path):
        store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
        ctx = store.get_or_create(session_id="metadata-session")
        ctx.set_metadata("last_request_id", "req-007")
        ctx.set_metadata("approval_mode", "manual")

        restarted_store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
        loaded = restarted_store.get("metadata-session")

        assert loaded is not None
        assert loaded.metadata["last_request_id"] == "req-007"
        assert loaded.metadata["approval_mode"] == "manual"

    def test_delete(self, tmp_path: Path):
        store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
        store.get_or_create(session_id="delete-session")

        assert store.delete("delete-session") is True
        assert store.count == 0
        assert store.delete("delete-session") is False
