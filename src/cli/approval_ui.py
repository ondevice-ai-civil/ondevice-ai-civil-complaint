"""Approval / rejection UI for GovOn CLI.

Renders a direction-key–driven prompt using `prompt_toolkit` when available.
Falls back to a plain input() prompt if prompt_toolkit is not installed.
"""

from __future__ import annotations

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
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    _PT_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


def _display_width(s: str) -> int:
    """Return the display width of *s*, counting wide (CJK) chars as 2."""
    w = 0
    for ch in s:
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ("W", "F") else 1
    return w


def _box_line(content: str = "", *, width: int) -> str:
    """Return a single box line padded to *width* display columns."""
    pad = width - _display_width(content)
    inner = content + " " * max(pad, 0)
    return f"│ {inner} │"


def _build_box_lines(
    approval_request: dict, selected: int, box_width: int | None = None
) -> list[str]:
    """Build the raw text lines of the approval box (no ANSI needed here)."""
    goal: str = approval_request.get("goal", "")
    reason: str = approval_request.get("reason", "")
    tool_summaries: list[str] = approval_request.get("tool_summaries") or []

    w = get_approval_box_width(get_terminal_columns()) if box_width is None else box_width
    _header = "─ 작업 승인 요청 "
    top = "┌" + _header + "─" * max(w - _display_width(_header) + 2, 0) + "┐"
    bot = "└" + "─" * (w + 2) + "┘"

    lines: list[str] = [top, _box_line(width=w)]

    def _wrap(label: str, value: str) -> None:
        prefix = f"  {label}: "
        available = max(w - _display_width(prefix), 1)
        if _display_width(value) <= available:
            lines.append(_box_line(f"{prefix}{value}", width=w))
        else:
            # Truncate value to fit within available display columns
            chunk: list[str] = []
            used = 0
            for ch in value:
                cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                if used + cw > available:
                    break
                chunk.append(ch)
                used += cw
            first = "".join(chunk)
            lines.append(_box_line(f"{prefix}{first}", width=w))
            rest = value[len(first) :]
            while rest:
                row: list[str] = []
                used = 0
                col_limit = w - 4
                for ch in rest:
                    cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                    if used + cw > col_limit:
                        break
                    row.append(ch)
                    used += cw
                seg = "".join(row)
                lines.append(_box_line(f"    {seg}", width=w))
                rest = rest[len(seg) :]

    _wrap("목표", goal)
    _wrap("이유", reason)

    if tool_summaries:
        lines.append(_box_line(width=w))
        lines.append(_box_line("  수행할 작업:", width=w))
        for idx, summary in enumerate(tool_summaries, 1):
            prefix = f"    {idx}. "
            avail = max(w - _display_width(prefix), 1)
            if _display_width(summary) <= avail:
                lines.append(_box_line(f"{prefix}{summary}", width=w))
            else:
                chunk2: list[str] = []
                used2 = 0
                for ch in summary:
                    cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                    if used2 + cw > avail:
                        break
                    chunk2.append(ch)
                    used2 += cw
                first2 = "".join(chunk2)
                lines.append(_box_line(f"{prefix}{first2}", width=w))
                rest2 = summary[len(first2) :]
                while rest2:
                    row2: list[str] = []
                    used2 = 0
                    col_limit2 = max(w - 7, 1)
                    for ch in rest2:
                        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                        if used2 + cw > col_limit2:
                            break
                        row2.append(ch)
                        used2 += cw
                    seg2 = "".join(row2)
                    lines.append(_box_line(f"       {seg2}", width=w))
                    rest2 = rest2[len(seg2) :]

    lines.append(_box_line(width=w))
    approve_bullet = "●" if selected == 0 else "○"
    reject_bullet = "●" if selected == 1 else "○"
    lines.append(_box_line(f"  {approve_bullet} 승인", width=w))
    lines.append(_box_line(f"  {reject_bullet} 거절", width=w))
    lines.append(bot)
    return lines


def show_approval_prompt(approval_request: dict) -> bool:
    """Show an interactive approval / rejection prompt.

    Returns True if approved, False if rejected.
    """
    terminal_columns = get_terminal_columns()
    if not is_layout_supported(terminal_columns):
        print(get_narrow_terminal_warning(terminal_columns))
        return _fallback_prompt(approval_request, columns=terminal_columns)

    if not _PT_AVAILABLE:
        return _fallback_prompt(approval_request, columns=terminal_columns)

    return _pt_prompt(approval_request, columns=terminal_columns)


def _pt_prompt(approval_request: dict, *, columns: int) -> bool:
    """prompt_toolkit–based arrow-key selection UI."""
    state = {"selected": 0, "result": None}
    box_width = get_approval_box_width(columns)

    def get_text():
        # Keep a stable width for a single prompt interaction.
        lines = _build_box_lines(approval_request, state["selected"], box_width=box_width)
        return "\n".join(lines) + "\n\n↑↓ 방향키로 선택, Enter로 확정"

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
