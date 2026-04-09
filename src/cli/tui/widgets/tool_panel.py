"""Collapsible tool call panel for displaying tool execution status."""

from __future__ import annotations

from textual.widgets import Static


class ToolCallPanel(Static):
    """Displays a tool call with bordered header and optional result.

    Shows tool name, execution status, and timing information.
    Collapsed by default — shows only the header line.
    """

    DEFAULT_CSS = """
    ToolCallPanel {
        width: 100%;
        height: auto;
        margin: 0 2;
    }
    """

    def __init__(self, tool_name: str, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self._completed = False
        self._latency_ms: float = 0
        self.update(
            f"[cyan]\u250c\u2500 \u2699 [bold]{tool_name}[/bold] \uc2e4\ud589 \uc911\u2026[/cyan]"
        )

    def complete(self, latency_ms: float = 0) -> None:
        """Mark the tool call as completed with timing info."""
        self._completed = True
        self._latency_ms = latency_ms
        latency_str = f" ({latency_ms:.0f}ms)" if latency_ms else ""
        self.update(
            f"[green]\u2514\u2500 \u2726 [bold]{self.tool_name}[/bold] "
            f"\uc644\ub8cc{latency_str}[/green]"
        )
