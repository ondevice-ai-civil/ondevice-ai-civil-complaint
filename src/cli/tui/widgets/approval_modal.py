"""Textual approval modal for human-in-the-loop tool execution approval.

Reuses constants and normalization logic from the legacy approval_ui module.
"""

from __future__ import annotations

from textual.binding import Binding
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from src.cli.approval_ui import (
    _get_task_type_label,
    _get_task_type_style,
    _normalize_approval_request,
)
from src.cli.tui.messages import ApprovalResult


class ApprovalModal(Widget):
    """Inline approval widget with keyboard navigation.

    Displays the approval request details and lets the user choose
    approve or reject using arrow keys / j/k and Enter.
    """

    DEFAULT_CSS = """
    ApprovalModal {
        width: 100%;
        height: auto;
        margin: 1 0;
        padding: 1 2;
        border: thick $warning;
    }

    ApprovalModal .choice {
        height: 1;
        padding: 0 1;
    }

    ApprovalModal .choice-selected {
        text-style: bold;
    }

    ApprovalModal .hint {
        color: $text-muted;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("up", "select_prev", "위로", show=False),
        Binding("down", "select_next", "아래로", show=False),
        Binding("k", "select_prev", "위로", show=False),
        Binding("j", "select_next", "아래로", show=False),
        Binding("enter", "confirm", "확정", show=False),
        Binding("q", "cancel", "취소", show=False),
    ]

    selected: reactive[int] = reactive(0)

    def __init__(self, approval_request: dict, thread_id: str, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self._request = _normalize_approval_request(approval_request)
        self._thread_id = thread_id
        self.can_focus = True

    def compose(self):  # noqa: ANN201
        """Build the approval UI content."""
        task_type = self._request.get("task_type")
        accent = _get_task_type_style(task_type)
        label = _get_task_type_label(task_type)
        goal = self._request.get("goal", "")
        reason = self._request.get("reason", "")
        tool_summaries = self._request.get("tool_summaries") or []

        yield Static(f"[bold {accent}]\uc791\uc5c5 \uc2b9\uc778 \uc694\uccad[/bold {accent}]")
        yield Static(f"[bold]\uc720\ud615:[/bold] [{accent}]{label}[/{accent}]")
        if goal:
            yield Static(f"[bold]\ubaa9\ud45c:[/bold] {goal}")
        if reason:
            yield Static(f"[dim]\uc774\uc720: {reason}[/dim]")
        if tool_summaries:
            for idx, summary in enumerate(tool_summaries, 1):
                yield Static(f"  [{accent}]{idx}.[/{accent}] {summary}")

        yield Static("")
        yield Static(self._choice_text(0), id="choice-approve", classes="choice")
        yield Static(self._choice_text(1), id="choice-reject", classes="choice")
        yield Static(
            "[dim]\u2191\u2193 \ubc29\ud5a5\ud0a4 / j k \uc120\ud0dd, Enter \ud655\uc815, q \ucde8\uc18c[/dim]",
            classes="hint",
        )

    def on_mount(self) -> None:
        """Focus this widget for keyboard input."""
        self.focus()

    def watch_selected(self, value: int) -> None:
        """Update choice display when selection changes."""
        try:
            self.query_one("#choice-approve", Static).update(self._choice_text(0))
            self.query_one("#choice-reject", Static).update(self._choice_text(1))
        except Exception:
            pass

    def _choice_text(self, index: int) -> str:
        """Render a choice line with selection indicator."""
        if index == 0:
            bullet = "\u25cf" if self.selected == 0 else "\u25cb"
            style = "bold green" if self.selected == 0 else "dim"
            return f"[{style}]{bullet} \uc2b9\uc778[/{style}]"
        bullet = "\u25cf" if self.selected == 1 else "\u25cb"
        style = "bold red" if self.selected == 1 else "dim"
        return f"[{style}]{bullet} \uac70\uc808[/{style}]"

    def action_select_prev(self) -> None:
        """Move selection up."""
        self.selected = max(0, self.selected - 1)

    def action_select_next(self) -> None:
        """Move selection down."""
        self.selected = min(1, self.selected + 1)

    def action_confirm(self) -> None:
        """Confirm the current selection."""
        approved = self.selected == 0
        self.post_message(ApprovalResult(approved=approved, thread_id=self._thread_id))
        self.remove()

    def action_cancel(self) -> None:
        """Cancel (reject) the approval request."""
        self.post_message(ApprovalResult(approved=False, thread_id=self._thread_id))
        self.remove()
