"""Input bar widget with history support for the GovOn TUI."""

from __future__ import annotations

from pathlib import Path

from textual.widgets import Input


class InputBar(Input):
    """Input field with ❯ prompt and persistent command history.

    History is stored in ~/.govon/history (same as the legacy REPL).
    """

    DEFAULT_CSS = """
    InputBar {
        width: 100%;
        dock: bottom;
    }
    """

    HISTORY_PATH = Path.home() / ".govon" / "history"

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(placeholder="\u276f ", **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._load_history()

    def _load_history(self) -> None:
        """Load command history from file."""
        try:
            if self.HISTORY_PATH.exists():
                self._history = self.HISTORY_PATH.read_text().strip().splitlines()[-500:]
        except OSError:
            pass

    def _save_entry(self, text: str) -> None:
        """Append a single entry to the history file."""
        try:
            self.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.HISTORY_PATH, "a") as f:
                f.write(f"{text}\n")
            self._history.append(text)
        except OSError:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Save submitted text to history."""
        text = event.value.strip()
        if text:
            self._save_entry(text)
            self._history_index = -1
