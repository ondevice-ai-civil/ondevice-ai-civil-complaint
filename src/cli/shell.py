"""GovOn CLI — main REPL loop and entry point.

Entry point registered in pyproject.toml:
  [project.scripts]
  govon = "src.cli.shell:main"
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Logging — must be imported before any other internal module so that
# loguru's default stderr handler is removed on import, preventing debug
# output from leaking into the terminal UI.
# ---------------------------------------------------------------------------
from src.cli.log_config import setup_logging  # noqa: E402  (import order intentional)

# ---------------------------------------------------------------------------
# Optional dependencies — graceful degradation
# ---------------------------------------------------------------------------
_PT_AVAILABLE = False
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    _PT_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------
from src.cli.approval_ui import show_approval_prompt  # noqa: E402
from src.cli.banner import render_banner  # noqa: E402
from src.cli.commands import handle_command, is_command  # noqa: E402
from src.cli.renderer import (  # noqa: E402
    render_error,
    render_metadata,
    render_result,
    render_session_info,
    render_status,
    render_thinking,
    render_tool_progress,
)

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

        def stream(self, query: str, session_id: str | None = None):
            raise RuntimeError("GovOnClient not available. Install the full GovOn package.")
            yield  # make it a generator

        def approve(self, thread_id: str, approved: bool) -> dict:
            raise RuntimeError("GovOnClient not available. Install the full GovOn package.")

        def cancel(self, thread_id: str) -> dict:
            raise RuntimeError("GovOnClient not available. Install the full GovOn package.")

        def health(self) -> dict:
            raise RuntimeError("GovOnClient not available. Install the full GovOn package.")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_PROMPT_TEXT = "\u276f "  # ❯ (Claude Code-style prompt)


def _should_use_tui() -> bool:
    """Return True if the Textual TUI should be used instead of the legacy REPL."""
    if os.environ.get("GOVON_PLAIN") == "1":
        return False
    if not sys.stdin.isatty():
        return False
    try:
        from src.cli.tui import HAS_TEXTUAL

        return HAS_TEXTUAL
    except ImportError:
        return False


def _get_input(session: "PromptSession | None") -> str:  # type: ignore[name-defined]
    """Read one line of user input (prompt_toolkit or plain input())."""
    if _PT_AVAILABLE and session is not None:
        return session.prompt(_PROMPT_TEXT)
    return input(_PROMPT_TEXT)


def _process_query(
    client: GovOnClient,
    query: str,
    session_id: str | None,
) -> tuple[str | None, bool]:
    """Send *query* to the backend and handle approval flow.

    Attempts to use the streaming endpoint (/v2/agent/stream) for per-node
    progress display. Falls back to the blocking run() call when the streaming
    endpoint is unavailable.

    Returns (new_session_id, should_continue).
    `should_continue` is False only when an unrecoverable error is returned
    that suggests the daemon is down.
    """
    result = _try_process_query(client, query, session_id)
    if result is not None:
        return result

    # All paths hit ConnectionError — wait for cold start then retry
    # NOTE: use __dict__ MRO traversal instead of hasattr() because
    # MagicMock returns True for any hasattr() check.
    _has_wait = any("wait_for_ready" in cls.__dict__ for cls in type(client).__mro__)
    if _has_wait:
        render_status("⊛ 서버 연결 재시도 중…")
        if client.wait_for_ready():
            result = _try_process_query(client, query, session_id)
            if result is not None:
                return result

    render_error("✘ 서버에 연결할 수 없습니다. govon --status로 상태를 확인하세요.")
    return session_id, False


def _try_process_query(
    client: GovOnClient,
    query: str,
    session_id: str | None,
) -> tuple[str | None, bool] | None:
    """Attempt to execute a query. Returns result on success, None if all paths fail."""
    # --- Try v3 streaming path first ---
    # Exclude MagicMock auto-attributes: only attempt if defined in the actual class.
    # Traverse the full MRO to support subclass inheritance.
    _has_v3 = any("stream_v3" in cls.__dict__ for cls in type(client).__mro__)
    if _has_v3:
        try:
            return _process_query_streaming_v3(client, query, session_id)
        except (
            AttributeError,
            NotImplementedError,
            httpx.HTTPStatusError,
            httpx.StreamError,
            OSError,
            ConnectionError,
        ):
            pass

    # --- Try v2 streaming path ---
    try:
        return _process_query_streaming(client, query, session_id)
    except (AttributeError, NotImplementedError):
        pass
    except (httpx.HTTPStatusError, httpx.StreamError, OSError):
        pass
    except ConnectionError:
        pass

    # --- Fallback: blocking run() ---
    try:
        return _process_query_blocking(client, query, session_id)
    except ConnectionError:
        return None


def _process_query_streaming_v3(
    client: GovOnClient,
    query: str,
    session_id: str | None,
) -> tuple[str | None, bool]:
    """v3 streaming path: calls client.stream_v3() with fine-grained SSE events.

    Uses SpinnerDisplay (Claude Code-style) that disappears on first content.
    """
    from src.cli.spinner import SpinnerDisplay

    new_session_id: str | None = None
    final_response: dict = {}
    spinner = SpinnerDisplay()
    spinner_active = True
    spinner.start()

    try:
        for event in client.stream_v3(query, session_id):
            event_type = event.get("type", "")

            if event_type == "error":
                spinner.stop()
                spinner_active = False
                render_error(event.get("error", "알 수 없는 오류가 발생했습니다."))
                return session_id, True

            if event_type == "thinking_start":
                iteration = event.get("iteration", 0)
                if iteration > 0 and not spinner_active:
                    spinner = SpinnerDisplay()
                    spinner.start()
                    spinner_active = True

            elif event_type == "thinking_delta":
                # First content token → spinner disappears
                if spinner_active:
                    spinner.stop()
                    spinner_active = False
                content = event.get("content", "")
                if content:
                    render_thinking(content)

            elif event_type == "thinking_end":
                if spinner_active:
                    spinner.stop()
                    spinner_active = False
                tool_calls = event.get("tool_calls", [])
                if tool_calls:
                    print()  # newline after thinking_delta
                    for tc in tool_calls:
                        render_tool_progress(tc.get("name", "unknown"), "start")

            elif event_type == "tool_start":
                pass  # already displayed in thinking_end

            elif event_type == "tool_end":
                tool_name = event.get("tool", "")
                render_tool_progress(tool_name, "end")

            elif event_type == "run_complete":
                if spinner_active:
                    spinner.stop()
                    spinner_active = False
                print()  # newline
                new_session_id = event.get("session_id") or event.get("thread_id")
                final_response = event
                metadata = event.get("metadata", {})
                if metadata:
                    render_metadata(metadata)
    finally:
        if spinner_active:
            spinner.stop()

    if final_response:
        _sid = final_response.get("session_id") or final_response.get("thread_id") or new_session_id
        render_result(final_response)
        return _sid or session_id, True

    render_result({"text": ""})
    return new_session_id or session_id, True


def _process_query_streaming(
    client: GovOnClient,
    query: str,
    session_id: str | None,
) -> tuple[str | None, bool]:
    """Streaming path: calls client.stream() and shows per-node progress."""
    final_response: dict = {}
    approval_event: dict | None = None
    new_session_id: str | None = None

    from src.cli.spinner import SpinnerDisplay

    spinner = SpinnerDisplay()
    spinner.start()
    try:
        for event in client.stream(query, session_id):
            node: str = event.get("node", "")
            event_status: str = event.get("status", "")

            if node == "error" or event_status == "error":
                spinner.stop()
                render_error(event.get("error", "알 수 없는 오류가 발생했습니다."))
                return session_id, True

            if event_status == "awaiting_approval":
                spinner.stop()
                approval_event = event
                break

            # Collect session/thread id from any event
            if not new_session_id:
                new_session_id = event.get("session_id") or event.get("thread_id")

            # Collect final result if present
            if event_status == "completed" or event.get("final_text") or event.get("text"):
                final_response = event
    finally:
        spinner.stop()

    # Handle approval
    if approval_event is not None:
        if not new_session_id:
            new_session_id = approval_event.get("session_id") or approval_event.get("thread_id")
        approval_request: dict = approval_event.get("approval_request") or {}
        approved = show_approval_prompt(approval_request)
        thread_id: str = approval_event.get("thread_id") or ""

        if not approved:
            try:
                client.approve(thread_id, approved=False)
            except Exception:  # pragma: no cover
                pass
            return new_session_id or session_id, True

        render_status("✤ 승인됨 — 계속 진행 중…")
        try:
            approved_response = client.approve(thread_id, approved=True)
        except Exception as exc:  # pragma: no cover
            render_error(f"✘ 승인 요청 실패: {exc}")
            return new_session_id or session_id, True

        render_result(approved_response)
        return (
            approved_response.get("session_id")
            or approved_response.get("thread_id")
            or new_session_id
            or session_id,
            True,
        )

    # Handle completed result from streaming events
    if final_response:
        _sid = final_response.get("session_id") or final_response.get("thread_id") or new_session_id
        render_result(final_response)
        return _sid or session_id, True

    # No useful response received
    render_result({"text": ""})
    return new_session_id or session_id, True


def _process_query_blocking(
    client: GovOnClient,
    query: str,
    session_id: str | None,
) -> tuple[str | None, bool]:
    """Blocking fallback path: calls client.run() with a simple spinner."""
    render_status("⊹ 처리 중…")

    try:
        response = client.run(query, session_id)
    except ConnectionError:
        raise  # propagate to caller for cold start retry
    except Exception as exc:  # pragma: no cover
        render_error(f"✘ 요청 실패: {exc}")
        return session_id, True

    new_session_id: str | None = response.get("session_id") or response.get("thread_id")
    status: str = response.get("status", "")

    if status == "awaiting_approval":
        approval_request: dict = response.get("approval_request") or {}
        approved = show_approval_prompt(approval_request)

        if not approved:
            # Rejected: notify server and return to prompt
            _thread_id: str = response.get("thread_id") or ""
            try:
                client.approve(_thread_id, approved=False)
            except Exception:  # pragma: no cover
                pass
            return new_session_id or session_id, True

        thread_id: str = response.get("thread_id") or ""
        render_status("✤ 승인됨 — 계속 진행 중…")
        try:
            approved_response = client.approve(thread_id, approved=True)
        except Exception as exc:  # pragma: no cover
            render_error(f"✘ 승인 요청 실패: {exc}")
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


def _ensure_server_ready(client: GovOnClient, is_remote: bool) -> bool:
    """Check server readiness, waiting for cold start if needed.

    Returns True if server is ready, False otherwise.
    Shows user-friendly status messages during the wait.
    """
    try:
        client.health()
        return True
    except (ConnectionError, httpx.ConnectError):
        if is_remote:
            render_status("서버에 연결 중… (sleeping 상태에서 깨어나는 중)")
        else:
            render_status("로컬 서버에 연결할 수 없습니다. 서버 상태를 확인해 주세요.")
            return False
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            render_status("서버 시작 중… (모델 로딩 대기)")
        else:
            render_status(f"서버 응답 대기 중… (HTTP {exc.response.status_code})")
    except httpx.TimeoutException:
        render_status("서버 응답 대기 중… (빌드 또는 모델 로딩 중)")
    except Exception:
        render_status("서버 연결 시도 중…")

    # Wait for server to become ready (cold start / build)
    if not client.wait_for_ready():
        render_error("서버에 연결할 수 없습니다. 나중에 다시 시도해 주세요.")
        return False
    return True


def _run_repl(
    client: GovOnClient,
    initial_session_id: str | None = None,
    is_remote: bool = False,
) -> None:
    """Run the interactive REPL until EOF or /exit."""
    session_id: str | None = initial_session_id
    history_path = Path.home() / ".govon" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pt_session = PromptSession(history=FileHistory(str(history_path))) if _PT_AVAILABLE else None
    server_checked = False

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

        # Lazy server readiness check on first query
        if not server_checked:
            if not _ensure_server_ready(client, is_remote):
                continue
            server_checked = True

        # Normal query
        try:
            session_id, should_continue = _process_query(client, text, session_id)
        except KeyboardInterrupt:
            # Ctrl+C while processing → cancel and return to prompt
            print("\n✧ 요청이 취소되었습니다.")
            continue

        if not should_continue:
            break

    if session_id:
        render_session_info(session_id)


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------


def _run_once(client: GovOnClient, query: str, session_id: str | None) -> None:
    """Run a single query and exit."""
    new_session_id, _ = _process_query(client, query, session_id)
    if new_session_id:
        render_session_info(new_session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the `govon` command."""
    # ── resolve package version early (needed for --version flag) ────────
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        ver = _pkg_version("govon")
    except PackageNotFoundError:
        ver = "dev"

    # ── SIGPIPE: restore default so that `govon … | head` works cleanly ──
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except AttributeError:
        pass  # Windows does not have SIGPIPE

    # ── graceful shutdown on SIGTERM / SIGHUP ─────────────────────────────
    def _graceful_exit(signum, frame):  # noqa: ANN001
        sys.exit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)
    try:
        signal.signal(signal.SIGHUP, _graceful_exit)
    except AttributeError:
        pass  # Windows does not have SIGHUP

    # ── early dispatch for server subcommand ──────────────────────────────
    # argparse handles positional + subparser mixing poorly,
    # so intercept 'server' first and delegate to a separate handler.
    raw_args = sys.argv[1:]

    # Bootstrap parse: extract --debug before any subcommand dispatch so
    # setup_logging() runs even for `govon server ...` (which exits early).
    _bootstrap = argparse.ArgumentParser(add_help=False)
    _bootstrap.add_argument("--debug", action="store_true", default=False)
    _bootstrap_args, _ = _bootstrap.parse_known_args(raw_args)
    setup_logging(debug=_bootstrap_args.debug)

    if raw_args and raw_args[0] == "server":
        from src.cli.server import handle_server

        sys.exit(handle_server(raw_args[1:]))

    parser = argparse.ArgumentParser(
        prog="govon",
        description="GovOn — shell-first local agentic runtime",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Subcommands:\n  govon server <command>   Docker backend management (pull/start/stop/status/logs)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {ver}")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Single-shot query (omit for interactive REPL mode)",
    )
    parser.add_argument(
        "--session",
        metavar="SESSION_ID",
        default=None,
        help="Existing session ID to resume",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check daemon status and exit",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop daemon and exit",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        dest="no_banner",
        help="시작 배너를 출력하지 않음",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging to stderr (WARNING+ level)",
    )

    args = parser.parse_args()

    # If GOVON_RUNTIME_URL is set, connect directly to the remote server
    # without managing the daemon.
    runtime_url = os.environ.get("GOVON_RUNTIME_URL")

    if runtime_url:
        if not runtime_url.startswith(("http://", "https://")):
            print(
                f"✘ 오류: GOVON_RUNTIME_URL은 http:// 또는 https://로 시작해야 합니다: {runtime_url}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Remote runtime mode: connect directly to the specified URL without daemon management
        if args.status:
            print(f"✦ GovOn daemon: 원격 모드 (GOVON_RUNTIME_URL={runtime_url})")
            sys.exit(0)
        if args.stop:
            print("✘ 오류: 원격 런타임 모드에서는 --stop을 사용할 수 없습니다.", file=sys.stderr)
            sys.exit(1)
        base_url = runtime_url.rstrip("/")
    else:
        # Local daemon mode
        daemon = DaemonManager()

        # --status
        if args.status:
            if daemon.is_running():
                print("✦ GovOn daemon: 실행 중")
            else:
                print("✧ GovOn daemon: 중지됨")
            sys.exit(0)

        # --stop
        if args.stop:
            daemon.stop()
            print("✧ GovOn daemon이 중지되었습니다.")
            sys.exit(0)

        # Local daemon: try to start but don't block the UI
        try:
            base_url = daemon.ensure_running()
        except Exception:
            base_url = daemon.get_base_url()

    client = GovOnClient(base_url)

    # Show UI immediately — server readiness is checked lazily at query time
    if args.query:
        # Single-shot mode: must ensure server is ready before sending
        _ensure_server_ready(client, is_remote=bool(runtime_url))
        _run_once(client, args.query, args.session)
    elif _should_use_tui():
        # Textual TUI mode — full terminal UI
        from src.cli.tui.app import GovOnApp

        app = GovOnApp(
            client=client,
            session_id=args.session,
            version=ver,
            runtime_url=runtime_url,
        )
        app.run(inline=True)
    else:
        # Legacy REPL mode — rich + prompt_toolkit fallback
        if not args.no_banner:
            mode = "remote" if runtime_url else "local"
            render_banner(version=ver, mode=mode, runtime_url=runtime_url)

            try:
                from rich.console import Console
                from rich.rule import Rule

                Console().print(Rule("종료: Ctrl+D 또는 /exit", style="dim"))
            except ImportError:  # pragma: no cover
                print("─── 종료: Ctrl+D 또는 /exit ───")

        _run_repl(client, initial_session_id=args.session, is_remote=bool(runtime_url))


if __name__ == "__main__":
    main()
