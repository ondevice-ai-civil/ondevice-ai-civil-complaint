"""Semantic color system for GovOn CLI.

Centralizes all Rich color/style strings into themed token sets,
replacing hardcoded color literals throughout the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GovOnTheme:
    """Immutable set of Rich-compatible style strings for every semantic token.

    All values are strings that can be passed directly to Rich markup or
    ``rich.style.Style.parse()``.  An empty string means "no styling".
    """

    # Brand colors derived from govon_mark.png
    brand_primary: str  # dark green  #2D5A3D
    brand_accent: str  # cream/ivory #F5E6C8

    # General text
    text_primary: str
    text_secondary: str
    text_link: str

    # Status indicators
    status_success: str
    status_error: str
    status_warning: str
    status_info: str

    # Panel chrome
    panel_border: str
    panel_title: str

    # Tool call lifecycle
    tool_start: str
    tool_end: str

    # Diff output
    diff_added: str
    diff_removed: str

    # Accent colors used for approval-UI task-type badges
    accent_cyan: str
    accent_blue: str
    accent_magenta: str
    accent_yellow: str


# ---------------------------------------------------------------------------
# Built-in themes
# ---------------------------------------------------------------------------

DARK_THEME = GovOnTheme(
    brand_primary="color(#2D5A3D)",
    brand_accent="color(#F5E6C8)",
    text_primary="",
    text_secondary="dim",
    text_link="bright_cyan",
    status_success="bold green",
    status_error="bold red",
    status_warning="yellow",
    status_info="dim",
    panel_border="green",
    panel_title="bold green",
    tool_start="yellow",
    tool_end="green",
    diff_added="green",
    diff_removed="red",
    accent_cyan="cyan",
    accent_blue="bright_blue",
    accent_magenta="magenta",
    accent_yellow="yellow",
)

LIGHT_THEME = GovOnTheme(
    brand_primary="dark_green",
    brand_accent="color(#8B6914)",
    text_primary="black",
    text_secondary="grey50",
    text_link="blue",
    status_success="bold dark_green",
    status_error="bold red3",
    status_warning="dark_orange",
    status_info="grey50",
    panel_border="dark_green",
    panel_title="bold dark_green",
    tool_start="dark_orange",
    tool_end="dark_green",
    diff_added="dark_green",
    diff_removed="red3",
    accent_cyan="dark_cyan",
    accent_blue="blue",
    accent_magenta="dark_magenta",
    accent_yellow="dark_orange",
)

NO_COLOR_THEME = GovOnTheme(
    brand_primary="",
    brand_accent="",
    text_primary="",
    text_secondary="",
    text_link="",
    status_success="",
    status_error="",
    status_warning="",
    status_info="",
    panel_border="",
    panel_title="",
    tool_start="",
    tool_end="",
    diff_added="",
    diff_removed="",
    accent_cyan="",
    accent_blue="",
    accent_magenta="",
    accent_yellow="",
)

# ---------------------------------------------------------------------------
# Theme resolution
# ---------------------------------------------------------------------------

_THEME_MAP: dict[str, GovOnTheme] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "no-color": NO_COLOR_THEME,
}


@lru_cache(maxsize=1)
def get_theme() -> GovOnTheme:
    """Return the active theme, resolved once and cached for the process lifetime.

    Resolution order:
    1. ``NO_COLOR`` env var present (any value) → :data:`NO_COLOR_THEME`
    2. ``GOVON_THEME`` env var set to ``"light"``, ``"dark"``, or ``"no-color"``
    3. Default → :data:`DARK_THEME`
    """
    if os.environ.get("NO_COLOR") is not None:
        return NO_COLOR_THEME

    govon_theme_env = os.environ.get("GOVON_THEME", "").strip().lower()
    if govon_theme_env:
        theme = _THEME_MAP.get(govon_theme_env)
        if theme is None:
            valid = ", ".join(f'"{k}"' for k in _THEME_MAP)
            raise ValueError(
                f"Unknown GOVON_THEME value {govon_theme_env!r}. " f"Valid options: {valid}."
            )
        return theme

    return DARK_THEME


def get_rich_style(token: str) -> str:
    """Return the Rich markup style string for the given semantic token name.

    Args:
        token: Attribute name on :class:`GovOnTheme`, e.g. ``"status_success"``.

    Returns:
        A Rich-compatible style string (may be empty when styling is disabled).

    Raises:
        AttributeError: If *token* is not a valid :class:`GovOnTheme` field.

    Example::

        style = get_rich_style("status_error")
        console.print(f"[{style}]Error: something went wrong[/{style}]")
    """
    theme = get_theme()
    return getattr(theme, token)
