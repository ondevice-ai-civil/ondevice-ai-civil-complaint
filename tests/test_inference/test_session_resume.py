"""SqliteSaver restart-safe 및 session resume contract 테스트.

Issue #129: MemorySaver → SqliteSaver 전환 및 session resume contract 구현.

검증 시나리오:
  1. SqliteSaver로 graph 빌드 → interrupt → 새 SqliteSaver(같은 DB) 인스턴스로
     get_state 복원 → resume → 완료 (restart-safe)
  2. SessionStore restart-safe: 재시작 후 graph_run 경로 포함 상태 복원
  3. cleanup_old_sessions: retention policy 메서드 동작 검증
  4. schema_version migration: 기존 DB(v0)에서 v2로 순차 업그레이드 검증
"""

from __future__ import annotations

import os
import sqlite3
import time

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.inference.graph.builder import build_govon_graph
from src.inference.graph.executor_adapter import ExecutorAdapter
from src.inference.graph.planner_adapter import RegexPlannerAdapter
from src.inference.graph.state import ApprovalStatus
from src.inference.session_context import SessionStore

os.environ.setdefault("SKIP_MODEL_LOAD", "true")


# ---------------------------------------------------------------------------
# 공통 fixtures / stubs
# ---------------------------------------------------------------------------


class StubExecutorAdapter(ExecutorAdapter):
    """테스트용 스텁 executor — LLM/외부 API 없이 실행 가능."""

    async def execute(self, tool_name: str, query: str, context: dict) -> dict:
        return {
            "success": True,
            "text": f"[stub] {tool_name} for: {query}",
            "latency_ms": 1.0,
        }

    def list_tools(self) -> list[str]:
        return ["rag_search", "api_lookup", "draft_civil_response"]


@pytest.fixture
def session_store(tmp_path):
    db_path = str(tmp_path / "sessions.sqlite3")
    return SessionStore(db_path=db_path)


def _build_graph_with_sqlite(cp_db_path: str, session_store: SessionStore):
    """SqliteSaver를 사용하는 graph를 빌드한다.

    SqliteSaver import 실패 시 MemorySaver로 fallback하여
    패키지 미설치 환경에서도 테스트가 실행된다.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(cp_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    except ImportError:
        pytest.skip("langgraph-checkpoint-sqlite 미설치 — SqliteSaver 테스트 건너뜀")

    return build_govon_graph(
        planner_adapter=RegexPlannerAdapter(),
        executor_adapter=StubExecutorAdapter(),
        session_store=session_store,
        checkpointer=checkpointer,
    )


# ---------------------------------------------------------------------------
# 1. Restart-safe graph checkpoint 복원 테스트
# ---------------------------------------------------------------------------


class TestRestartSafeGraphCheckpoint:
    """SqliteSaver를 사용한 graph restart-safe 테스트."""

    def test_interrupt_state_survives_new_saver_instance(self, tmp_path, session_store):
        """graph invoke → interrupt → 새 SqliteSaver 인스턴스 → get_state 복원 검증.

        SqliteSaver를 사용하면 프로세스 재시작(= 새 인스턴스) 후에도
        interrupt 상태를 DB에서 복원할 수 있어야 한다.
        """
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            pytest.skip("langgraph-checkpoint-sqlite 미설치")

        cp_db = str(tmp_path / "langgraph_checkpoints.db")
        thread_id = "resume-test-thread-1"
        config = {"configurable": {"thread_id": thread_id}}

        # --- 1단계: 첫 번째 SqliteSaver 인스턴스로 interrupt까지 실행 ---
        conn1 = sqlite3.connect(cp_db, check_same_thread=False)
        saver1 = SqliteSaver(conn1)
        graph1 = build_govon_graph(
            planner_adapter=RegexPlannerAdapter(),
            executor_adapter=StubExecutorAdapter(),
            session_store=session_store,
            checkpointer=saver1,
        )
        initial = {
            "session_id": "resume-session",
            "request_id": "resume-req-1",
            "messages": [HumanMessage(content="답변 초안 작성해줘")],
        }
        graph1.invoke(initial, config=config)
        state_before = graph1.get_state(config)
        assert state_before.next, "interrupt 상태여야 합니다"
        conn1.close()

        # --- 2단계: 새 SqliteSaver 인스턴스(같은 DB 파일)로 상태 복원 ---
        conn2 = sqlite3.connect(cp_db, check_same_thread=False)
        saver2 = SqliteSaver(conn2)
        graph2 = build_govon_graph(
            planner_adapter=RegexPlannerAdapter(),
            executor_adapter=StubExecutorAdapter(),
            session_store=session_store,
            checkpointer=saver2,
        )

        # 새 graph 인스턴스에서 이전 interrupt 상태가 복원되어야 한다
        state_restored = graph2.get_state(config)
        assert (
            state_restored.next
        ), "새 SqliteSaver 인스턴스(같은 DB)에서 interrupt 상태가 복원되어야 합니다"

        # --- 3단계: resume → 완료 검증 ---
        from langgraph.types import Command

        result = graph2.invoke(
            Command(resume={"approved": True}),
            config=config,
        )
        assert result.get("final_text"), "resume 후 final_text가 생성되어야 합니다"
        assert result.get("approval_status") == ApprovalStatus.APPROVED.value

        conn2.close()

    def test_completed_graph_state_persists_across_instances(self, tmp_path, session_store):
        """완료된 graph의 최종 상태가 새 SqliteSaver 인스턴스에서도 조회된다."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            pytest.skip("langgraph-checkpoint-sqlite 미설치")

        from langgraph.types import Command

        cp_db = str(tmp_path / "langgraph_checkpoints.db")
        thread_id = "persist-complete-thread-1"
        config = {"configurable": {"thread_id": thread_id}}

        # 완료까지 실행
        conn1 = sqlite3.connect(cp_db, check_same_thread=False)
        graph1 = build_govon_graph(
            planner_adapter=RegexPlannerAdapter(),
            executor_adapter=StubExecutorAdapter(),
            session_store=session_store,
            checkpointer=SqliteSaver(conn1),
        )
        graph1.invoke(
            {
                "session_id": "s1",
                "request_id": "r1",
                "messages": [HumanMessage(content="답변 작성")],
            },
            config=config,
        )
        graph1.invoke(Command(resume={"approved": True}), config=config)
        conn1.close()

        # 새 인스턴스에서 최종 상태 확인: next가 비어야 한다 (완료)
        conn2 = sqlite3.connect(cp_db, check_same_thread=False)
        graph2 = build_govon_graph(
            planner_adapter=RegexPlannerAdapter(),
            executor_adapter=StubExecutorAdapter(),
            session_store=session_store,
            checkpointer=SqliteSaver(conn2),
        )
        state = graph2.get_state(config)
        assert not state.next, "완료된 graph는 next가 비어야 합니다"
        conn2.close()


# ---------------------------------------------------------------------------
# 2. SessionStore restart-safe with graph_run 경로
# ---------------------------------------------------------------------------


class TestSessionStoreRestartSafe:
    """SessionStore 재시작 안전성 — graph_run 경로 포함."""

    def test_graph_run_path_survives_restart(self, tmp_path):
        """graph_run 기록(plan_summary, approval, capabilities)이 재시작 후 복원된다."""
        db_path = str(tmp_path / "sessions.sqlite3")

        store1 = SessionStore(db_path=db_path)
        ctx = store1.get_or_create(session_id="gr-session")
        ctx.add_graph_run(
            request_id="gr-req-1",
            plan_summary="민원 답변 초안 작성",
            approval_status="approved",
            executed_capabilities=["rag_search", "draft_civil_response"],
            status="completed",
            total_latency_ms=123.4,
        )

        # 재시작 시뮬레이션: 새 SessionStore 인스턴스
        store2 = SessionStore(db_path=db_path)
        loaded = store2.get("gr-session")

        assert loaded is not None
        assert len(loaded.graph_runs) == 1
        run = loaded.graph_runs[0]
        assert run.request_id == "gr-req-1"
        assert run.plan_summary == "민원 답변 초안 작성"
        assert run.approval_status == "approved"
        assert run.executed_capabilities == ["rag_search", "draft_civil_response"]
        assert run.status == "completed"

    def test_tool_run_with_graph_run_id_survives_restart(self, tmp_path):
        """tool_run의 graph_run_request_id 연결이 재시작 후 복원된다."""
        db_path = str(tmp_path / "sessions.sqlite3")

        store1 = SessionStore(db_path=db_path)
        ctx = store1.get_or_create(session_id="tr-session")
        ctx.add_tool_run(
            "api_lookup",
            success=True,
            graph_run_request_id="gr-req-tool",
            latency_ms=55.0,
        )

        store2 = SessionStore(db_path=db_path)
        loaded = store2.get("tr-session")
        assert loaded is not None
        assert loaded.tool_runs[0].graph_run_request_id == "gr-req-tool"


# ---------------------------------------------------------------------------
# 3. Retention Policy
# ---------------------------------------------------------------------------


class TestRetentionPolicy:
    """SessionStore.cleanup_old_sessions 메서드 검증."""

    def test_cleanup_removes_old_sessions(self, tmp_path):
        """max_age_days를 초과한 세션이 삭제된다."""
        store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))

        # 오래된 세션 직접 삽입 (updated_at을 과거로 설정)
        import sqlite3 as _sqlite3
        from contextlib import closing

        old_time = time.time() - 10 * 86400  # 10일 전
        with closing(_sqlite3.connect(store.db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at, metadata_json) "
                "VALUES (?, ?, ?, '{}')",
                ("old-session", old_time, old_time),
            )

        # 최신 세션 생성
        store.get_or_create(session_id="new-session")
        assert store.count == 2

        deleted = store.cleanup_old_sessions(max_age_days=7)
        assert deleted == 1
        assert store.count == 1
        assert store.get("new-session") is not None
        assert store.get("old-session") is None

    def test_cleanup_returns_zero_when_nothing_to_delete(self, tmp_path):
        """삭제할 세션이 없으면 0을 반환한다."""
        store = SessionStore(db_path=str(tmp_path / "sessions.sqlite3"))
        store.get_or_create(session_id="recent-session")

        deleted = store.cleanup_old_sessions(max_age_days=30)
        assert deleted == 0


# ---------------------------------------------------------------------------
# 4. Schema Migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    """schema_version 기반 순차 migration 검증."""

    def test_fresh_db_reaches_latest_schema_version(self, tmp_path):
        """새 DB가 최신 schema_version에 도달한다."""
        from src.inference.session_context import SCHEMA_VERSION

        db_path = str(tmp_path / "sessions.sqlite3")
        SessionStore(db_path=db_path)

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        conn.close()

        assert row[0] == SCHEMA_VERSION

    def test_legacy_db_without_schema_version_migrates_correctly(self, tmp_path):
        """schema_version 테이블 없는 레거시 DB가 올바르게 마이그레이션된다.

        레거시 DB는 이미 기본 테이블(sessions, messages 등)이 있지만
        schema_version이 없는 상태를 시뮬레이션한다.
        이 경우 _migrate_v1의 CREATE TABLE IF NOT EXISTS가 안전하게 동작하고
        _migrate_v2에서 graph_run_request_id 컬럼이 추가되어야 한다.
        """
        from src.inference.session_context import SCHEMA_VERSION

        db_path = str(tmp_path / "legacy.sqlite3")

        # 레거시 DB: graph_run_request_id 없는 tool_runs 테이블
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE tool_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                tool TEXT NOT NULL,
                success INTEGER NOT NULL,
                latency_ms REAL NOT NULL DEFAULT 0,
                error TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                timestamp REAL NOT NULL
            );
            CREATE TABLE graph_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                plan_summary TEXT NOT NULL DEFAULT '',
                approval_status TEXT,
                executed_capabilities_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                error TEXT,
                total_latency_ms REAL NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                started_at REAL NOT NULL,
                completed_at REAL NOT NULL
            );
            CREATE TABLE metadata (
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (owner_type, owner_id, key)
            );
        """)
        conn.close()

        # SessionStore 초기화 → migration 실행
        store = SessionStore(db_path=db_path)

        conn2 = sqlite3.connect(db_path)
        # schema_version이 최신이어야 한다
        row = conn2.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        assert row[0] == SCHEMA_VERSION

        # graph_run_request_id 컬럼이 추가되어야 한다
        cols = {r[1] for r in conn2.execute("PRAGMA table_info(tool_runs)").fetchall()}
        assert (
            "graph_run_request_id" in cols
        ), "migration v2: tool_runs.graph_run_request_id 컬럼이 없습니다"
        conn2.close()

        # 정상 동작 확인
        ctx = store.get_or_create(session_id="legacy-migrated")
        ctx.add_tool_run("rag_search", success=True, graph_run_request_id="req-legacy")
        loaded = store.get("legacy-migrated")
        assert loaded.tool_runs[0].graph_run_request_id == "req-legacy"
