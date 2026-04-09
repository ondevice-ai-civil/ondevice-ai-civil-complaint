"""Streaming markdown view widget for LLM response rendering."""

from __future__ import annotations

from textual.widgets import Markdown


class MarkdownView(Markdown):
    """Markdown widget that supports incremental content via append_delta().

    Accumulates text deltas and re-renders the full markdown on each update.
    Uses Textual's built-in Markdown widget which supports syntax highlighting.
    """

    DEFAULT_CSS = """
    MarkdownView {
        width: 100%;
        margin: 0 0;
    }
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__("", **kwargs)
        self._accumulated = ""

    def append_delta(self, text: str) -> None:
        """Append a text chunk and re-render the markdown."""
        self._accumulated += text
        self.update(self._accumulated)

    def set_content(self, text: str) -> None:
        """Replace the entire content at once."""
        self._accumulated = text
        self.update(text)
