"""Approval / rejection UI for GovOn CLI.

Renders a direction-key–driven prompt using `prompt_toolkit` when available.
Falls back to a plain input() prompt if prompt_toolkit is not installed.
"""

from __future__ import annotations

import unicodedata

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

_BOX_WIDTH = 55


def _display_width(s: str) -> int:
    """Return the display width of *s*, counting wide (CJK) chars as 2."""
    w = 0
    for ch in s:
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ("W", "F") else 1
    return w


def _box_line(content: str = "", width: int = _BOX_WIDTH) -> str:
    """Return a single box line padded to *width* display columns."""
    pad = width - _display_width(content)
    inner = content + " " * max(pad, 0)
    return f"│ {inner} │"


def _build_box_lines(approval_request: dict, selected: int) -> list[str]:
    """Build the raw text lines of the approval box (no ANSI needed here)."""
    goal: str = approval_request.get("goal", "")
    reason: str = approval_request.get("reason", "")
    tool_summaries: list[str] = approval_request.get("tool_summaries") or []

    w = _BOX_WIDTH
    _header = "─ 작업 승인 요청 "
    top = "┌" + _header + "─" * (w - _display_width(_header) + 2) + "┐"
    bot = "└" + "─" * (w + 2) + "┘"

    lines: list[str] = [top, _box_line()]

    def _wrap(label: str, value: str) -> None:
        available = w - _display_width(label) - 2  # "  label: " prefix overhead
        if _display_width(value) <= available:
            lines.append(_box_line(f"  {label}: {value}"))
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
            lines.append(_box_line(f"  {label}: {first}"))
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
                lines.append(_box_line(f"    {seg}"))
                rest = rest[len(seg) :]

    _wrap("목표", goal)
    _wrap("이유", reason)

    if tool_summaries:
        lines.append(_box_line())
        lines.append(_box_line("  수행할 작업:"))
        for idx, summary in enumerate(tool_summaries, 1):
            prefix = f"    {idx}. "
            avail = w - _display_width(prefix)
            if _display_width(summary) <= avail:
                lines.append(_box_line(f"{prefix}{summary}"))
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
                lines.append(_box_line(f"{prefix}{first2}"))
                rest2 = summary[len(first2) :]
                while rest2:
                    row2: list[str] = []
                    used2 = 0
                    col_limit2 = w - 7
                    for ch in rest2:
                        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                        if used2 + cw > col_limit2:
                            break
                        row2.append(ch)
                        used2 += cw
                    seg2 = "".join(row2)
                    lines.append(_box_line(f"       {seg2}"))
                    rest2 = rest2[len(seg2) :]

    lines.append(_box_line())
    approve_bullet = "●" if selected == 0 else "○"
    reject_bullet = "●" if selected == 1 else "○"
    lines.append(_box_line(f"  {approve_bullet} 승인"))
    lines.append(_box_line(f"  {reject_bullet} 거절"))
    lines.append(bot)
    return lines


def show_approval_prompt(approval_request: dict) -> bool:
    """Show an interactive approval / rejection prompt.

    Returns True if approved, False if rejected.
    """
    if not _PT_AVAILABLE:
        return _fallback_prompt(approval_request)

    return _pt_prompt(approval_request)


def _pt_prompt(approval_request: dict) -> bool:
    """prompt_toolkit–based arrow-key selection UI."""
    state = {"selected": 0, "result": None}

    def get_text():
        lines = _build_box_lines(approval_request, state["selected"])
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


def _fallback_prompt(approval_request: dict) -> bool:
    """Plain input() fallback when prompt_toolkit is unavailable."""
    goal: str = approval_request.get("goal", "")
    reason: str = approval_request.get("reason", "")
    tool_summaries: list[str] = approval_request.get("tool_summaries") or []

    print("\n── 작업 승인 요청 ─────────────────────────────")
    if goal:
        print(f"  목표: {goal}")
    if reason:
        print(f"  이유: {reason}")
    if tool_summaries:
        print("\n  수행할 작업:")
        for idx, s in enumerate(tool_summaries, 1):
            print(f"    {idx}. {s}")
    print("───────────────────────────────────────────────")

    try:
        answer = input("승인하시겠습니까? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return answer in ("y", "yes", "예", "네")
