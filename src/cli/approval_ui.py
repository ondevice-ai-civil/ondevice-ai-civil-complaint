"""Approval / rejection UI for GovOn CLI.

Renders a direction-key–driven prompt using `prompt_toolkit` when available.
Falls back to a plain input() prompt if prompt_toolkit is not installed.
"""

from __future__ import annotations

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


def _box_line(content: str = "", width: int = _BOX_WIDTH) -> str:
    """Return a single box line padded to *width* (inner characters)."""
    inner = content.ljust(width)
    return f"│ {inner} │"


def _build_box_lines(approval_request: dict, selected: int) -> list[str]:
    """Build the raw text lines of the approval box (no ANSI needed here)."""
    goal: str = approval_request.get("goal", "")
    reason: str = approval_request.get("reason", "")
    tool_summaries: list[str] = approval_request.get("tool_summaries") or []

    w = _BOX_WIDTH
    top = "┌─ 작업 승인 요청 " + "─" * (w - len("─ 작업 승인 요청 ") + 2) + "┐"
    bot = "└" + "─" * (w + 2) + "┘"

    lines: list[str] = [top, _box_line()]

    def _wrap(label: str, value: str) -> None:
        available = w - len(label) - 2  # "  label: " prefix overhead
        if len(value) <= available:
            lines.append(_box_line(f"  {label}: {value}"))
        else:
            lines.append(_box_line(f"  {label}: {value[:available]}"))
            rest = value[available:]
            while rest:
                lines.append(_box_line(f"    {rest[:w - 4]}"))
                rest = rest[w - 4 :]

    _wrap("목표", goal)
    _wrap("이유", reason)

    if tool_summaries:
        lines.append(_box_line())
        lines.append(_box_line("  수행할 작업:"))
        for idx, summary in enumerate(tool_summaries, 1):
            prefix = f"    {idx}. "
            avail = w - len(prefix)
            if len(summary) <= avail:
                lines.append(_box_line(f"{prefix}{summary}"))
            else:
                lines.append(_box_line(f"{prefix}{summary[:avail]}"))
                rest = summary[avail:]
                while rest:
                    lines.append(_box_line(f"       {rest[:w - 7]}"))
                    rest = rest[w - 7 :]

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
        control.text = get_text()

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
