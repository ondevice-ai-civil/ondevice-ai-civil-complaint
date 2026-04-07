"""Approval / rejection UI for GovOn CLI.

Renders a direction-key–driven prompt using `prompt_toolkit` when available.
Falls back to a plain input() prompt if prompt_toolkit is not installed.
"""

from __future__ import annotations

import io
import unicodedata

from src.cli.terminal import (
    get_approval_box_width,
    get_narrow_terminal_warning,
    get_terminal_columns,
    is_layout_supported,
)

_PT_AVAILABLE = False
try:
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    _PT_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

_RICH_AVAILABLE = False
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

TASK_TYPE_STYLES = {
    "draft_response": "cyan",
    "revise_response": "blue",
    "lookup_stats": "magenta",
    "issue_detection": "yellow",
    "stats_query": "magenta",
    "keyword_analysis": "yellow",
    "demographics_query": "bright_blue",
}

TASK_TYPE_LABELS = {
    "draft_response": "답변 초안 작성",
    "revise_response": "답변 수정",
    "lookup_stats": "통계 조회",
    "issue_detection": "이슈 탐지",
    "stats_query": "통계 조회",
    "keyword_analysis": "키워드 분석",
    "demographics_query": "인구통계 조회",
}

DEFAULT_TASK_TYPE_STYLE = "cyan"
DEFAULT_TASK_TYPE_LABEL = "일반 작업"


def _display_width(s: str) -> int:
    """Return the display width of *s*, counting wide (CJK) chars as 2."""
    w = 0
    for ch in s:
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ("W", "F") else 1
    return w


def _get_task_type_style(task_type: str | None) -> str:
    """Return the accent style used for an approval task type."""
    return TASK_TYPE_STYLES.get(task_type or "", DEFAULT_TASK_TYPE_STYLE)


def _get_task_type_label(task_type: str | None) -> str:
    """Return the human-readable label used for an approval task type."""
    return TASK_TYPE_LABELS.get(task_type or "", DEFAULT_TASK_TYPE_LABEL)


def _get_approval_panel_width(columns: int) -> int:
    """Return a responsive rich panel width for the approval UI."""
    return min(columns - 2, get_approval_box_width(columns) + 4)


def _build_tool_summaries_text(tool_summaries: list[str], accent_style: str):
    """Return styled tool summary lines for the approval panel."""
    text = Text()
    for idx, summary in enumerate(tool_summaries, 1):
        if idx > 1:
            text.append("\n")
        text.append(f"{idx}. ", style=f"bold {accent_style}")
        text.append(summary)
    return text


def _build_choice_text(label: str, *, selected: bool, style: str):
    """Return a styled approval choice row."""
    bullet = "●" if selected else "○"
    text_style = f"bold {style}" if selected else "dim white"
    return Text(f"{bullet} {label}", style=text_style)


def _build_approval_panel(approval_request: dict, selected: int, *, columns: int):
    """Build the rich approval panel shown inside the prompt_toolkit UI."""
    task_type: str | None = approval_request.get("task_type")
    accent_style = _get_task_type_style(task_type)
    goal: str = approval_request.get("goal", "")
    reason: str = approval_request.get("reason", "")
    tool_summaries: list[str] = approval_request.get("tool_summaries") or []

    summary = Table.grid(expand=True, padding=(0, 1))
    summary.add_column(style="bold bright_white", no_wrap=True, width=6)
    summary.add_column(ratio=1)
    summary.add_row("유형", Text(_get_task_type_label(task_type), style=f"bold {accent_style}"))
    if goal:
        summary.add_row("목표", Text(goal))
    if reason:
        summary.add_row("이유", Text(reason, style="dim"))
    if tool_summaries:
        summary.add_row("작업", _build_tool_summaries_text(tool_summaries, accent_style))

    choices = Table.grid(expand=True)
    choices.add_column()
    choices.add_row(_build_choice_text("승인", selected=selected == 0, style="green"))
    choices.add_row(_build_choice_text("거절", selected=selected == 1, style="red"))

    footer = Text("↑↓ 방향키 / j k 선택, Enter 확정, q 취소", style="dim")
    body = Group(summary, Text(""), choices, Text(""), footer)
    return Panel(
        body,
        title=Text("작업 승인 요청", style=f"bold {accent_style}"),
        border_style=accent_style,
        width=_get_approval_panel_width(columns),
        padding=(0, 1),
    )


def _render_approval_panel_ansi(approval_request: dict, selected: int, *, columns: int) -> str:
    """Render the rich approval panel to ANSI text for prompt_toolkit."""
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
        width=columns,
    )
    console.print(_build_approval_panel(approval_request, selected, columns=columns))
    return buffer.getvalue().rstrip()


def show_approval_prompt(approval_request: dict) -> bool:
    """Show an interactive approval / rejection prompt.

    Returns True if approved, False if rejected.
    """
    terminal_columns = get_terminal_columns()
    if not is_layout_supported(terminal_columns):
        print(get_narrow_terminal_warning(terminal_columns))
        return _fallback_prompt(approval_request, columns=terminal_columns)

    if not (_PT_AVAILABLE and _RICH_AVAILABLE):
        return _fallback_prompt(approval_request, columns=terminal_columns)

    return _pt_prompt(approval_request, columns=terminal_columns)


def _pt_prompt(approval_request: dict, *, columns: int) -> bool:
    """prompt_toolkit–based arrow-key selection UI."""
    state = {"selected": 0, "result": None}

    def get_text():
        return ANSI(
            _render_approval_panel_ansi(
                approval_request,
                state["selected"],
                columns=columns,
            )
        )

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        state["selected"] = (state["selected"] - 1) % 2
        _refresh_control()

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        state["selected"] = (state["selected"] + 1) % 2
        _refresh_control()

    @kb.add("enter")
    def _confirm(event):
        state["result"] = state["selected"] == 0
        event.app.exit()

    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event):
        state["result"] = False
        event.app.exit()

    control = FormattedTextControl(text=get_text)
    window = Window(content=control)
    layout = Layout(HSplit([window]))

    def _refresh_control():
        control.text = get_text  # keep as callable
        app.invalidate()

    app: Application = Application(layout=layout, key_bindings=kb, full_screen=False)
    app.run()

    return bool(state["result"])


def _fallback_prompt(approval_request: dict, columns: int | None = None) -> bool:
    """Plain input() fallback when prompt_toolkit is unavailable."""
    task_type: str | None = approval_request.get("task_type")
    goal: str = approval_request.get("goal", "")
    reason: str = approval_request.get("reason", "")
    tool_summaries: list[str] = approval_request.get("tool_summaries") or []
    terminal_columns = get_terminal_columns() if columns is None else columns
    separator = "─" * max(terminal_columns - 2, 12)
    title = " 작업 승인 요청 "
    title_width = _display_width(title)
    if terminal_columns > title_width:
        fill_width = terminal_columns - title_width
        left_fill = fill_width // 2
        right_fill = fill_width - left_fill
        title_line = f"{'─' * left_fill}{title}{'─' * right_fill}"
    else:
        title_line = title

    print(f"\n{title_line}")
    if task_type:
        print(f"  유형: {_get_task_type_label(task_type)}")
    if goal:
        print(f"  목표: {goal}")
    if reason:
        print(f"  이유: {reason}")
    if tool_summaries:
        print("\n  수행할 작업:")
        for idx, s in enumerate(tool_summaries, 1):
            print(f"    {idx}. {s}")
    print(separator)

    try:
        answer = input("승인하시겠습니까? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return answer in ("y", "yes", "예", "네")
