"""구조화 로그 시스템 — JSON Lines + Rich 터미널 동시 출력.

로그 레벨:
  FLOW   — 파이프라인 노드 전이 (session_load -> planner -> ...)
  METRIC — 레이턴시, 토큰 수, 메모리 사용량
  ASSERT — 시나리오 검증 결과 (PASS/FAIL/WARN)
  DEBUG  — HTTP 요청/응답 상세
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class LogEntry:
    """단일 로그 엔트리."""

    timestamp: str
    level: str  # FLOW | METRIC | ASSERT | INFO | WARN | ERROR | DEBUG
    phase: int
    scenario_id: int
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class E2ELogger:
    """듀얼 출력 로거: JSON Lines 파일 + 터미널."""

    def __init__(self, log_path: str, verbose: bool = True) -> None:
        self._log_path = Path(log_path)
        self._verbose = verbose
        self._entries: List[LogEntry] = []
        self._file = open(self._log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._console = Console(stderr=True) if RICH_AVAILABLE else None
        self._phase = 0
        self._scenario_id = 0

    def set_context(self, phase: int = 0, scenario_id: int = 0) -> None:
        self._phase = phase
        self._scenario_id = scenario_id

    def _write(self, entry: LogEntry) -> None:
        self._entries.append(entry)
        line = entry.to_json()
        self._file.write(line + "\n")
        self._file.flush()

        if self._verbose:
            self._print_terminal(entry)

    def _print_terminal(self, entry: LogEntry) -> None:
        level = entry.level
        tag_map = {
            "FLOW": "[cyan][FLOW][/cyan]" if RICH_AVAILABLE else "[FLOW]",
            "METRIC": "[blue][METRIC][/blue]" if RICH_AVAILABLE else "[METRIC]",
            "ASSERT": "[green][ASSERT][/green]" if RICH_AVAILABLE else "[ASSERT]",
            "INFO": "[white][INFO][/white]" if RICH_AVAILABLE else "[INFO]",
            "WARN": "[yellow][WARN][/yellow]" if RICH_AVAILABLE else "[WARN]",
            "ERROR": "[red][ERROR][/red]" if RICH_AVAILABLE else "[ERROR]",
            "DEBUG": "[dim][DEBUG][/dim]" if RICH_AVAILABLE else "[DEBUG]",
        }
        tag = tag_map.get(level, f"[{level}]")

        if self._console and RICH_AVAILABLE:
            self._console.print(f"{entry.timestamp} {tag} {entry.message}")
        else:
            plain_tag = f"[{level}]"
            print(f"{entry.timestamp} {plain_tag} {entry.message}", file=sys.stderr)

    def _now(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def flow(self, message: str, **data: Any) -> None:
        self._write(LogEntry(self._now(), "FLOW", self._phase, self._scenario_id, message, data))

    def metric(self, message: str, **data: Any) -> None:
        self._write(LogEntry(self._now(), "METRIC", self._phase, self._scenario_id, message, data))

    def assertion(self, message: str, **data: Any) -> None:
        self._write(LogEntry(self._now(), "ASSERT", self._phase, self._scenario_id, message, data))

    def info(self, message: str, **data: Any) -> None:
        self._write(LogEntry(self._now(), "INFO", self._phase, self._scenario_id, message, data))

    def warn(self, message: str, **data: Any) -> None:
        self._write(LogEntry(self._now(), "WARN", self._phase, self._scenario_id, message, data))

    def error(self, message: str, **data: Any) -> None:
        self._write(LogEntry(self._now(), "ERROR", self._phase, self._scenario_id, message, data))

    def debug(self, message: str, **data: Any) -> None:
        if self._verbose:
            self._write(
                LogEntry(self._now(), "DEBUG", self._phase, self._scenario_id, message, data)
            )

    def scenario_result(
        self,
        scenario_id: int,
        name: str,
        phase: int,
        status: str,
        elapsed: float,
        attempts: int = 1,
        assertions: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        error: Optional[str] = None,
        detail: Optional[Any] = None,
    ) -> dict:
        """시나리오 결과를 로그에 기록하고 결과 dict를 반환한다."""
        tag = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}.get(status, "????")

        msg = f"[{tag}] Scenario {scenario_id}: {name} ({elapsed:.2f}s)"
        if status == "passed":
            self.assertion(msg)
        elif status == "skipped":
            self.warn(f"{msg} -- {error or 'skipped'}")
        else:
            self.error(f"{msg} -- {error}")

        if warnings:
            for w in warnings:
                self.warn(f"  [WARN] {w}")

        entry = {
            "id": scenario_id,
            "name": name,
            "phase": phase,
            "status": status,
            "attempts": attempts,
            "elapsed_s": round(elapsed, 3),
            "assertions": assertions or [],
            "warnings": warnings or [],
            "error": error,
            "detail": detail,
        }
        return entry

    def close(self) -> None:
        self._file.close()

    @property
    def entries(self) -> List[LogEntry]:
        return list(self._entries)
