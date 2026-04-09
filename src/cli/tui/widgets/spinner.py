"""Textual spinner widget with Korean proverbs and elapsed time.

Uses Textual's built-in timer system instead of a background thread.
Reuses spinner characters and verbs from src.cli.spinner.
"""

from __future__ import annotations

import random
import time

from textual.reactive import reactive
from textual.widget import Widget

from src.cli.spinner import SPINNER_CHARS, STATUS_VERBS, _format_elapsed, _format_tokens


class SpinnerWidget(Widget):
    """Animated spinner showing a Korean proverb, elapsed time, and token count."""

    DEFAULT_CSS = """
    SpinnerWidget {
        height: 1;
        width: 100%;
    }
    """

    tokens: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self._verb = random.choice(STATUS_VERBS)  # noqa: S311
        self._start_time = time.monotonic()
        self._frame = 0
        self._timer = None

    def on_mount(self) -> None:
        """Start the animation timer at ~30fps."""
        self._timer = self.set_interval(1 / 30, self._tick)

    def _tick(self) -> None:
        """Advance one animation frame."""
        self._frame += 1
        # Change verb every ~90 frames (~3 seconds)
        if self._frame % 90 == 0:
            self._verb = random.choice(STATUS_VERBS)  # noqa: S311
        self.refresh()

    def render(self) -> str:
        """Render the current spinner frame."""
        char = SPINNER_CHARS[self._frame % len(SPINNER_CHARS)]
        elapsed = _format_elapsed(time.monotonic() - self._start_time)
        parts = f"{char} {self._verb}\u2026 ({elapsed}"
        if self.tokens > 0:
            parts += f" \u00b7 \u2193 {_format_tokens(self.tokens)} tokens"
        parts += ")"
        return parts

    def stop(self) -> None:
        """Stop the animation timer."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
