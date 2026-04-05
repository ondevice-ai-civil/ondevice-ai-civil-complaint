"""세션 컨텍스트 및 SQLite 기반 세션 저장소.

GovOn Shell MVP의 세션 모델은 다음을 저장한다.

- 대화 기록
- tool 사용 기록
- task loop 단위 실행 로그

초안 버전, 선택 근거 목록 같은 무거운 상태는 제품 기본 저장 범위에서 제외한다.

Schema versioning
-----------------
_init_db()는 schema_version 테이블을 통해 순차 migration을 관리한다.
현재 최신 버전: SCHEMA_VERSION = 2

Migration history:
  v1 → v2: tool_runs에 graph_run_request_id 컬럼 추가 (이전에는 ad-hoc ALTER TABLE)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

SCHEMA_VERSION = 2
"""현재 SessionStore SQLite 스키마 버전."""


def _default_session_db_path() -> str:
    base_dir = Path(os.getenv("GOVON_HOME", Path.home() / ".govon"))
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / "sessions.sqlite3")


@dataclass
class ConversationTurn:
    """대화 한 턴."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolRunRecord:
    """도구 실행 로그."""

    tool: str
    success: bool
    graph_run_request_id: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphRunRecord:
    """task loop 단위 실행 로그."""

    request_id: str
    plan_summary: str = ""
    approval_status: Optional[str] = None
    executed_capabilities: List[str] = field(default_factory=list)
    status: str = "completed"
    error: Optional[str] = None
    total_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: float = field(default_factory=time.time)


@dataclass
class SessionContext:
    """세션 기반 대화/도구 기록 컨텍스트."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_history: int = 20
    conversations: List[ConversationTurn] = field(default_factory=list)
    tool_runs: List[ToolRunRecord] = field(default_factory=list)
    graph_runs: List[GraphRunRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    _persist_turn: Optional[Callable[[ConversationTurn], None]] = field(default=None, repr=False)
    _persist_tool_run: Optional[Callable[[ToolRunRecord], None]] = field(default=None, repr=False)
    _persist_graph_run: Optional[Callable[[GraphRunRecord], None]] = field(default=None, repr=False)
    _persist_metadata: Optional[Callable[[str, Any], None]] = field(default=None, repr=False)

    def add_turn(self, role: str, content: str, **kwargs: Any) -> None:
        """대화 턴을 추가하고 필요 시 영속화한다."""
        turn = ConversationTurn(role=role, content=content, metadata=kwargs)
        self.conversations.append(turn)
        if len(self.conversations) > self.max_history:
            removed = len(self.conversations) - self.max_history
            self.conversations = self.conversations[removed:]
            logger.debug(f"세션 {self.session_id}: 오래된 대화 {removed}턴 제거")

        if self._persist_turn:
            self._persist_turn(turn)

    def add_tool_run(
        self,
        tool: str,
        success: bool,
        graph_run_request_id: Optional[str] = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """도구 실행 로그를 추가하고 필요 시 영속화한다."""
        record = ToolRunRecord(
            tool=tool,
            graph_run_request_id=graph_run_request_id,
            success=success,
            latency_ms=latency_ms,
            error=error,
            metadata=metadata or {},
        )
        self.tool_runs.append(record)
        if self._persist_tool_run:
            self._persist_tool_run(record)

    @property
    def recent_history(self) -> List[ConversationTurn]:
        return list(self.conversations)

    @property
    def recent_tool_runs(self) -> List[ToolRunRecord]:
        return list(self.tool_runs)

    def add_graph_run(
        self,
        request_id: str,
        plan_summary: str = "",
        approval_status: Optional[str] = None,
        executed_capabilities: Optional[List[str]] = None,
        status: str = "completed",
        error: Optional[str] = None,
        total_latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
    ) -> None:
        """task loop 단위 실행 로그를 추가하고 필요 시 영속화한다."""
        record = GraphRunRecord(
            request_id=request_id,
            plan_summary=plan_summary,
            approval_status=approval_status,
            executed_capabilities=list(executed_capabilities or []),
            status=status,
            error=error,
            total_latency_ms=total_latency_ms,
            metadata=metadata or {},
            started_at=started_at or time.time(),
            completed_at=completed_at or time.time(),
        )
        for index, existing in enumerate(self.graph_runs):
            if existing.request_id == request_id:
                self.graph_runs[index] = record
                break
        else:
            self.graph_runs.append(record)
        if self._persist_graph_run:
            self._persist_graph_run(record)

    @property
    def recent_graph_runs(self) -> List[GraphRunRecord]:
        return list(self.graph_runs)

    def set_metadata(self, key: str, value: Any) -> None:
        """세션 메타데이터를 설정하고 영속화한다."""
        self.metadata[key] = value
        if self._persist_metadata:
            self._persist_metadata(key, value)

    def build_context_summary(self) -> str:
        """최근 대화와 tool 사용 기록을 요약한다."""
        parts: List[str] = []

        if self.conversations:
            history_lines = []
            for turn in self.conversations[-5:]:
                role_label = "사용자" if turn.role == "user" else "시스템"
                history_lines.append(f"[{role_label}] {turn.content}")
            parts.append("### 최근 대화\n" + "\n".join(history_lines))

        if self.tool_runs:
            tool_lines = []
            for record in self.tool_runs[-5:]:
                status = "성공" if record.success else "실패"
                line = f"- {record.tool}: {status}"
                if record.error:
                    line += f" ({record.error})"
                tool_lines.append(line)
            parts.append("### 최근 도구 실행\n" + "\n".join(tool_lines))

        if self.graph_runs:
            run_lines = []
            for record in self.graph_runs[-3:]:
                approval = record.approval_status or "미기록"
                tools = ", ".join(record.executed_capabilities) or "도구 없음"
                line = f"- {record.status} / 승인={approval} / tools={tools}"
                if record.error:
                    line += f" ({record.error})"
                run_lines.append(line)
            parts.append("### 최근 작업 실행\n" + "\n".join(run_lines))

        return "\n\n".join(parts)


class SessionStore:
    """SQLite 기반 세션 저장소."""

    def __init__(self, db_path: Optional[str] = None, max_history: int = 20) -> None:
        self._db_path = db_path or os.getenv("GOVON_SESSION_DB") or _default_session_db_path()
        self._max_history = max_history
        self._init_db()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """데이터베이스를 초기화하고 순차 schema migration을 실행한다.

        schema_version 테이블을 통해 현재 버전을 추적하며,
        버전 번호 순서대로 migration 함수를 적용한다.
        각 migration은 원자적(atomic)으로 실행된다.
        """
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            # schema_version 테이블: 버전 추적의 단일 소스
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current_version = row["v"] if row and row["v"] is not None else 0

        # 버전 1: 기본 스키마 생성
        if current_version < 1:
            self._migrate_v1()

        # 버전 2: tool_runs.graph_run_request_id 컬럼 추가
        if current_version < 2:
            self._migrate_v2()

    def _migrate_v1(self) -> None:
        """v1: 기본 스키마(sessions, messages, tool_runs, graph_runs, metadata) 생성."""
        with closing(self._connect()) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    timestamp REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_runs_session_id
                ON tool_runs(session_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_runs (
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
                    completed_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_graph_runs_session_id
                ON graph_runs(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_graph_runs_session_request
                ON graph_runs(session_id, request_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (owner_type, owner_id, key)
                )
            """)
            conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (1)")
            logger.debug("SessionStore schema migration v1 완료")

    def _migrate_v2(self) -> None:
        """v2: tool_runs에 graph_run_request_id 컬럼 및 복합 인덱스 추가."""
        with closing(self._connect()) as conn, conn:
            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(tool_runs)").fetchall()
            }
            if "graph_run_request_id" not in existing_columns:
                conn.execute("ALTER TABLE tool_runs ADD COLUMN graph_run_request_id TEXT")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_runs_session_graph_run
                ON tool_runs(session_id, graph_run_request_id)
            """)
            conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (2)")
            logger.debug("SessionStore schema migration v2 완료")

    def _ensure_session(self, session_id: str, created_at: Optional[float] = None) -> None:
        now = time.time()
        created = created_at or now
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, '{}')
                ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (session_id, created, now),
            )

    def _load_session_metadata_json(self, session_id: str) -> Dict[str, Any]:
        row = self._load_session_metadata(session_id)
        if row is None:
            return {}
        return json.loads(row["metadata_json"] or "{}")

    def _upsert_session_metadata_json(self, session_id: str, metadata: Dict[str, Any]) -> None:
        self._ensure_session(session_id)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE sessions SET metadata_json=?, updated_at=? WHERE session_id=?",
                (json.dumps(metadata, ensure_ascii=False), time.time(), session_id),
            )

    def _append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        self._ensure_session(session_id)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO messages(session_id, role, content, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn.role,
                    turn.content,
                    turn.timestamp,
                    json.dumps(turn.metadata, ensure_ascii=False),
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (time.time(), session_id),
            )

    def _append_tool_run(self, session_id: str, record: ToolRunRecord) -> None:
        self._ensure_session(session_id)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO tool_runs(
                    session_id,
                    graph_run_request_id,
                    tool,
                    success,
                    latency_ms,
                    error,
                    metadata_json,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    record.graph_run_request_id,
                    record.tool,
                    1 if record.success else 0,
                    record.latency_ms,
                    record.error,
                    json.dumps(record.metadata, ensure_ascii=False),
                    record.timestamp,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (time.time(), session_id),
            )

    def _append_graph_run(self, session_id: str, record: GraphRunRecord) -> None:
        self._ensure_session(session_id)
        with closing(self._connect()) as conn, conn:
            existing = conn.execute(
                """
                SELECT id
                FROM graph_runs
                WHERE session_id=? AND request_id=?
                """,
                (session_id, record.request_id),
            ).fetchone()
            payload = (
                record.plan_summary,
                record.approval_status,
                json.dumps(record.executed_capabilities, ensure_ascii=False),
                record.status,
                record.error,
                record.total_latency_ms,
                json.dumps(record.metadata, ensure_ascii=False),
                record.started_at,
                record.completed_at,
                session_id,
                record.request_id,
            )
            if existing:
                conn.execute(
                    """
                    UPDATE graph_runs
                    SET
                        plan_summary=?,
                        approval_status=?,
                        executed_capabilities_json=?,
                        status=?,
                        error=?,
                        total_latency_ms=?,
                        metadata_json=?,
                        started_at=?,
                        completed_at=?
                    WHERE session_id=? AND request_id=?
                    """,
                    payload,
                )
            else:
                conn.execute(
                    """
                    INSERT INTO graph_runs(
                        session_id,
                        request_id,
                        plan_summary,
                        approval_status,
                        executed_capabilities_json,
                        status,
                        error,
                        total_latency_ms,
                        metadata_json,
                        started_at,
                        completed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        record.request_id,
                        record.plan_summary,
                        record.approval_status,
                        json.dumps(record.executed_capabilities, ensure_ascii=False),
                        record.status,
                        record.error,
                        record.total_latency_ms,
                        json.dumps(record.metadata, ensure_ascii=False),
                        record.started_at,
                        record.completed_at,
                    ),
                )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (time.time(), session_id),
            )

    def _upsert_metadata(self, session_id: str, key: str, value: Any) -> None:
        metadata = self._load_session_metadata_json(session_id)
        metadata[key] = value
        self._upsert_session_metadata_json(session_id, metadata)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO metadata(owner_type, owner_id, key, value_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(owner_type, owner_id, key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (
                    "session",
                    session_id,
                    key,
                    json.dumps(value, ensure_ascii=False),
                    time.time(),
                ),
            )

    def _load_messages(self, session_id: str, max_history: int) -> List[ConversationTurn]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT role, content, timestamp, metadata_json
                FROM messages
                WHERE session_id=?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        turns = [
            ConversationTurn(
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]
        return turns[-max_history:]

    def _load_tool_runs(self, session_id: str) -> List[ToolRunRecord]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT tool, success, latency_ms, error, metadata_json, timestamp
                     , graph_run_request_id
                FROM tool_runs
                WHERE session_id=?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ToolRunRecord(
                tool=row["tool"],
                graph_run_request_id=row["graph_run_request_id"],
                success=bool(row["success"]),
                latency_ms=row["latency_ms"],
                error=row["error"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def _load_graph_runs(self, session_id: str) -> List[GraphRunRecord]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT
                    request_id,
                    plan_summary,
                    approval_status,
                    executed_capabilities_json,
                    status,
                    error,
                    total_latency_ms,
                    metadata_json,
                    started_at,
                    completed_at
                FROM graph_runs
                WHERE session_id=?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            GraphRunRecord(
                request_id=row["request_id"],
                plan_summary=row["plan_summary"],
                approval_status=row["approval_status"],
                executed_capabilities=json.loads(row["executed_capabilities_json"] or "[]"),
                status=row["status"],
                error=row["error"],
                total_latency_ms=row["total_latency_ms"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                started_at=row["started_at"],
                completed_at=row["completed_at"],
            )
            for row in rows
        ]

    def _load_session_metadata(self, session_id: str) -> Optional[sqlite3.Row]:
        with closing(self._connect()) as conn, conn:
            return conn.execute(
                "SELECT session_id, created_at, metadata_json FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()

    def _load_metadata_entries(self, owner_type: str, owner_id: str) -> Dict[str, Any]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT key, value_json
                FROM metadata
                WHERE owner_type=? AND owner_id=?
                ORDER BY key ASC
                """,
                (owner_type, owner_id),
            ).fetchall()
        return {row["key"]: json.loads(row["value_json"] or "null") for row in rows}

    def _build_context(self, session_id: str, max_history: int) -> Optional[SessionContext]:
        row = self._load_session_metadata(session_id)
        if row is None:
            return None
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata.update(self._load_metadata_entries("session", session_id))
        return SessionContext(
            session_id=session_id,
            max_history=max_history,
            conversations=self._load_messages(session_id, max_history),
            tool_runs=self._load_tool_runs(session_id),
            graph_runs=self._load_graph_runs(session_id),
            metadata=metadata,
            created_at=row["created_at"],
            _persist_turn=lambda turn: self._append_turn(session_id, turn),
            _persist_tool_run=lambda record: self._append_tool_run(session_id, record),
            _persist_graph_run=lambda record: self._append_graph_run(session_id, record),
            _persist_metadata=lambda key, value: self._upsert_metadata(session_id, key, value),
        )

    def get_or_create(
        self,
        session_id: Optional[str] = None,
        max_history: Optional[int] = None,
    ) -> SessionContext:
        history_limit = max_history or self._max_history
        if session_id:
            existing = self._build_context(session_id, history_limit)
            if existing is not None:
                return existing

        sid = session_id or str(uuid.uuid4())
        created_at = time.time()
        self._ensure_session(sid, created_at=created_at)
        logger.info(f"새 세션 생성: {sid}")
        return SessionContext(
            session_id=sid,
            max_history=history_limit,
            created_at=created_at,
            _persist_turn=lambda turn: self._append_turn(sid, turn),
            _persist_tool_run=lambda record: self._append_tool_run(sid, record),
            _persist_graph_run=lambda record: self._append_graph_run(sid, record),
            _persist_metadata=lambda key, value: self._upsert_metadata(sid, key, value),
        )

    def get(self, session_id: str) -> Optional[SessionContext]:
        return self._build_context(session_id, self._max_history)

    def delete(self, session_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            deleted = conn.execute(
                "DELETE FROM sessions WHERE session_id=?", (session_id,)
            ).rowcount
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM tool_runs WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM graph_runs WHERE session_id=?", (session_id,))
            conn.execute(
                "DELETE FROM metadata WHERE owner_type='session' AND owner_id=?",
                (session_id,),
            )
        return bool(deleted)

    @property
    def count(self) -> int:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()
        return int(row["cnt"] if row else 0)

    def cleanup_old_sessions(self, max_age_days: int) -> int:
        """sessions.updated_at 기준으로 오래된 세션을 삭제한다.

        자동 호출되지 않으며, 운영자가 명시적으로 호출해야 한다.
        ON DELETE CASCADE가 설정된 테이블(messages, tool_runs, graph_runs, metadata)은
        세션 삭제 시 함께 정리된다.

        Parameters
        ----------
        max_age_days : int
            이 일수보다 오래된 세션(updated_at 기준)을 삭제한다.

        Returns
        -------
        int
            삭제된 세션 수.
        """
        cutoff = time.time() - max_age_days * 86400
        with closing(self._connect()) as conn, conn:
            deleted = conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,)).rowcount
        if deleted:
            logger.info(f"SessionStore: {deleted}개 세션 정리 (max_age_days={max_age_days})")
        return deleted
