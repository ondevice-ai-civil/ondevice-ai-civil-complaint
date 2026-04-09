"""Dim thinking text block for LLM reasoning display."""

from __future__ import annotations

from textual.widgets import Static


class ThinkingBlock(Static):
    """Displays LLM thinking/reasoning text in dim italic style.

    Supports incremental updates as thinking tokens stream in.
    """

    DEFAULT_CSS = """
    ThinkingBlock {
        width: 100%;
        color: $text-muted;
        text-style: italic;
        margin: 0 2;
    }
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__("", **kwargs)
        self._accumulated = ""

    def append(self, text: str) -> None:
        """Append thinking text and re-render."""
        self._accumulated += text
        self.update(self._accumulated)
