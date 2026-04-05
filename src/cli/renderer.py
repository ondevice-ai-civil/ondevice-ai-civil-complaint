"""Result rendering for GovOn CLI.

Uses `rich` when available; falls back to plain print() otherwise.
"""

from __future__ import annotations

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    _console = Console()
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _console = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False


def render_result(result: dict) -> None:
    """Render the final agent response to the terminal.

    Expected keys (at least one required):
      - result["text"] or result["response"]: main answer text
      - result["citations"] or result["sources"]: list of source strings (optional)
    """
    text_body: str = result.get("text") or result.get("response") or ""
    citations: list = result.get("citations") or result.get("sources") or []

    if _RICH_AVAILABLE:
        content = Text(text_body)
        if citations:
            content.append("\n\n출처\n", style="bold")
            for idx, src in enumerate(citations, 1):
                content.append(f"  {idx}. {src}\n", style="dim")
        _console.print(Panel(content, title="[bold green]GovOn[/bold green]", border_style="green"))
    else:
        print("\n── GovOn ──────────────────────────────────")
        print(text_body)
        if citations:
            print("\n출처")
            for idx, src in enumerate(citations, 1):
                print(f"  {idx}. {src}")
        print("───────────────────────────────────────────\n")


def render_status(message: str) -> None:
    """Render a transient status / progress message."""
    if _RICH_AVAILABLE:
        _console.print(f"[dim]→ {message}[/dim]")
    else:
        print(f"→ {message}")


def render_error(message: str) -> None:
    """Render an error message in red."""
    if _RICH_AVAILABLE:
        _console.print(f"[bold red]오류:[/bold red] {message}")
    else:
        print(f"오류: {message}")


def render_session_info(session_id: str) -> None:
    """Render session resume hint at shell exit."""
    hint = f"[session: {session_id}]  govon --session {session_id} 로 재개 가능"
    if _RICH_AVAILABLE:
        _console.print(f"[dim]{hint}[/dim]")
    else:
        print(hint)
