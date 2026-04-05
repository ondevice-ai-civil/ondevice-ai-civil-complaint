"""GovOn CLI — main REPL loop and entry point.

Entry point registered in pyproject.toml:
  [project.scripts]
  govon = "src.cli.shell:main"
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Optional dependencies — graceful degradation
# ---------------------------------------------------------------------------
_PT_AVAILABLE = False
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    _PT_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------
from src.cli.approval_ui import show_approval_prompt
from src.cli.commands import handle_command, is_command
from src.cli.renderer import render_error, render_result, render_session_info, render_status

# ---------------------------------------------------------------------------
# Stub imports for daemon / http_client (other agents implement these).
# If the real modules exist they are used; otherwise lightweight stubs
# are defined inline so the shell can be imported and tested standalone.
# ---------------------------------------------------------------------------
try:
    from src.cli.daemon import DaemonManager  # type: ignore[import]
except ImportError:  # pragma: no cover

    class DaemonManager:  # type: ignore[no-redef]
        """Stub: real implementation provided by daemon.py agent."""

        def ensure_running(self) -> str:
            raise RuntimeError("DaemonManager not available. Install the full GovOn package.")

        def is_running(self) -> bool:
            return False

        def stop(self) -> None:
            pass


try:
    from src.cli.http_client import GovOnClient  # type: ignore[import]
except ImportError:  # pragma: no cover

    class GovOnClient:  # type: ignore[no-redef]
        """Stub: real implementation provided by http_client.py agent."""

        def __init__(self, base_url: str) -> None:
            self._base_url = base_url

        def run(self, query: str, session_id: str | None = None) -> dict:
            raise RuntimeError("GovOnClient not available. Install the full GovOn package.")

        def approve(self, thread_id: str, approved: bool) -> dict:
            raise RuntimeError("GovOnClient not available. Install the full GovOn package.")

        def cancel(self, thread_id: str) -> dict:
            raise RuntimeError("GovOnClient not available. Install the full GovOn package.")

        def health(self) -> dict:
            raise RuntimeError("GovOnClient not available. Install the full GovOn package.")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_PROMPT_TEXT = "govon> "


def _get_input(session: "PromptSession | None") -> str:  # type: ignore[name-defined]
    """Read one line of user input (prompt_toolkit or plain input())."""
    if _PT_AVAILABLE and session is not None:
        return session.prompt(_PROMPT_TEXT)
    return input(_PROMPT_TEXT)


def _process_query(
    client: "GovOnClient",
    query: str,
    session_id: str | None,
) -> tuple[str | None, bool]:
    """Send *query* to the backend and handle approval flow.

    Returns (new_session_id, should_continue).
    `should_continue` is False only when an unrecoverable error is returned
    that suggests the daemon is down.
    """
    render_status("처리 중…")

    try:
        response = client.run(query, session_id)
    except Exception as exc:  # pragma: no cover
        render_error(f"요청 실패: {exc}")
        return session_id, True

    new_session_id: str | None = response.get("session_id") or response.get("thread_id")
    status: str = response.get("status", "")

    if status == "awaiting_approval":
        approval_request: dict = response.get("approval_request") or {}
        approved = show_approval_prompt(approval_request)

        if not approved:
            # 거절: 서버에 통보 후 프롬프트 복귀
            _thread_id: str = response.get("thread_id") or ""
            try:
                client.approve(_thread_id, approved=False)
            except Exception:  # pragma: no cover
                pass
            return new_session_id or session_id, True

        thread_id: str = response.get("thread_id") or ""
        render_status("승인됨 — 계속 진행 중…")
        try:
            approved_response = client.approve(thread_id, approved=True)
        except Exception as exc:  # pragma: no cover
            render_error(f"승인 요청 실패: {exc}")
            return new_session_id or session_id, True

        render_result(approved_response)
        return (
            approved_response.get("session_id")
            or approved_response.get("thread_id")
            or new_session_id
            or session_id,
            True,
        )

    if status in ("completed", "done", "success") or "text" in response or "response" in response:
        render_result(response)
        return new_session_id or session_id, True

    # Unknown status — render raw
    render_result({"text": str(response)})
    return new_session_id or session_id, True


# ---------------------------------------------------------------------------
# REPL loop
# ---------------------------------------------------------------------------


def _run_repl(client: "GovOnClient", initial_session_id: str | None = None) -> None:
    """Run the interactive REPL until EOF or /exit."""
    session_id: str | None = initial_session_id
    pt_session = PromptSession(history=InMemoryHistory()) if _PT_AVAILABLE else None

    while True:
        try:
            text = _get_input(pt_session).strip()
        except EOFError:
            # Ctrl+D
            break
        except KeyboardInterrupt:
            # Ctrl+C while idle → exit
            print()
            break

        if not text:
            continue

        if is_command(text):
            try:
                result = handle_command(text)
            except SystemExit:
                break
            if result is not None:
                print(result)
            continue

        # Normal query
        try:
            session_id, should_continue = _process_query(client, text, session_id)
        except KeyboardInterrupt:
            # Ctrl+C while processing → cancel and return to prompt
            print("\n요청이 취소되었습니다.")
            continue

        if not should_continue:  # pragma: no cover
            break

    if session_id:
        render_session_info(session_id)


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------


def _run_once(client: "GovOnClient", query: str, session_id: str | None) -> None:
    """Run a single query and exit."""
    new_session_id, _ = _process_query(client, query, session_id)
    if new_session_id:
        render_session_info(new_session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the `govon` command."""
    parser = argparse.ArgumentParser(
        prog="govon",
        description="GovOn — shell-first local agentic runtime",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="단발 실행할 질문 (생략 시 인터랙티브 REPL 모드)",
    )
    parser.add_argument(
        "--session",
        metavar="SESSION_ID",
        default=None,
        help="재개할 기존 세션 ID",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="daemon 상태 확인 후 종료",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="daemon 중지 후 종료",
    )

    args = parser.parse_args()

    daemon = DaemonManager()

    # --status
    if args.status:
        if daemon.is_running():
            print("GovOn daemon: 실행 중")
        else:
            print("GovOn daemon: 중지됨")
        sys.exit(0)

    # --stop
    if args.stop:
        daemon.stop()
        print("GovOn daemon이 중지되었습니다.")
        sys.exit(0)

    # Ensure daemon is up and get base URL
    try:
        base_url = daemon.ensure_running()
    except Exception as exc:
        print(f"오류: daemon을 시작할 수 없습니다 — {exc}", file=sys.stderr)
        sys.exit(1)

    client = GovOnClient(base_url)

    if args.query:
        # Single-shot mode
        _run_once(client, args.query, args.session)
    else:
        # Interactive REPL mode
        print("GovOn CLI  (종료: Ctrl+D 또는 /exit)")
        _run_repl(client, initial_session_id=args.session)


if __name__ == "__main__":
    main()
