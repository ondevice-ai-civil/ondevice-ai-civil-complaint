"""세션 컨텍스트 및 SQLite 기반 세션 저장소.

GovOn Shell MVP의 세션 모델은 다음만 저장한다.

- 대화 기록
- tool 사용 기록

초안 버전, 선택 근거 목록 같은 무거운 상태는 제품 기본 저장 범위에서 제외한다.
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
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionContext:
    """세션 기반 대화/도구 기록 컨텍스트."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_history: int = 20
    conversations: List[ConversationTurn] = field(default_factory=list)
    tool_runs: List[ToolRunRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    _persist_turn: Optional[Callable[[ConversationTurn], None]] = field(default=None, repr=False)
    _persist_tool_run: Optional[Callable[[ToolRunRecord], None]] = field(default=None, repr=False)

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
        latency_ms: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """도구 실행 로그를 추가하고 필요 시 영속화한다."""
        record = ToolRunRecord(
            tool=tool,
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
        return conn

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

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
                );
                """
            )

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
                INSERT INTO tool_runs(session_id, tool, success, latency_ms, error, metadata_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
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
                FROM tool_runs
                WHERE session_id=?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ToolRunRecord(
                tool=row["tool"],
                success=bool(row["success"]),
                latency_ms=row["latency_ms"],
                error=row["error"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def _load_session_metadata(self, session_id: str) -> Optional[sqlite3.Row]:
        with closing(self._connect()) as conn, conn:
            return conn.execute(
                "SELECT session_id, created_at, metadata_json FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()

    def _build_context(self, session_id: str, max_history: int) -> Optional[SessionContext]:
        row = self._load_session_metadata(session_id)
        if row is None:
            return None
        return SessionContext(
            session_id=session_id,
            max_history=max_history,
            conversations=self._load_messages(session_id, max_history),
            tool_runs=self._load_tool_runs(session_id),
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"],
            _persist_turn=lambda turn: self._append_turn(session_id, turn),
            _persist_tool_run=lambda record: self._append_tool_run(session_id, record),
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
        )

    def get(self, session_id: str) -> Optional[SessionContext]:
        return self._build_context(session_id, self._max_history)

    def delete(self, session_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            deleted = conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,)).rowcount
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM tool_runs WHERE session_id=?", (session_id,))
        return bool(deleted)

    @property
    def count(self) -> int:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()
        return int(row["cnt"] if row else 0)
