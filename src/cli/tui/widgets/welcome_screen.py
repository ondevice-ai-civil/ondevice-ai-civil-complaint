"""Welcome screen widget matching Claude Code's startup layout.

Displays the GovOn 'G' logo (ASCII art from govon_mark.png colors),
version info, runtime mode, and tips panel side by side.

Brand colors from govon_mark.png:
  - Dark green background: #1B3B2F / #2D5A3D
  - Cream/ivory 'G' text:  #F5E6C8
"""

from __future__ import annotations

import sys
from urllib.parse import urlparse

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static

# Brand color constants (from govon_mark.png)
_GREEN = "#2D5A3D"
_CREAM = "#F5E6C8"
_DIM = "#7A8B7E"

# ASCII art 'G' logo — matches banner.py _LARGE_LOGO_LINES
_LOGO_ART = r"""
  ██████╗
 ██╔════╝
 ██║ ████╗
 ██║ ╚═██║
 ╚██████╔╝
  ╚═════╝
""".strip("\n")


def _python_version() -> str:
    """Short Python version string."""
    return f"Python {sys.version_info.major}.{sys.version_info.minor}"


def _mode_label(runtime_url: str | None) -> str:
    """Human-readable mode label."""
    if runtime_url:
        try:
            host = urlparse(runtime_url).hostname or runtime_url
        except (ValueError, AttributeError):
            host = runtime_url
        return f"\uc6d0\uaca9: {host}"
    return "\ub85c\uceec \ubaa8\ub4dc"


class WelcomeScreen(Widget):
    """Claude Code-style welcome screen with logo, info, and tips."""

    DEFAULT_CSS = f"""
    WelcomeScreen {{
        width: 100%;
        height: auto;
        margin: 1 0;
        padding: 1 2;
        border: round {_GREEN};
    }}

    WelcomeScreen #welcome-layout {{
        width: 100%;
        height: auto;
    }}

    WelcomeScreen #logo-section {{
        width: auto;
        min-width: 30;
        height: auto;
        padding: 0 2;
    }}

    WelcomeScreen #info-section {{
        width: 1fr;
        height: auto;
        padding: 0 2;
        border-left: vkey {_GREEN};
    }}

    WelcomeScreen .logo-art {{
        color: {_CREAM};
    }}

    WelcomeScreen .version-text {{
        color: {_CREAM};
        text-style: bold;
    }}

    WelcomeScreen .mode-text {{
        color: {_DIM};
    }}

    WelcomeScreen .section-title {{
        color: {_CREAM};
        text-style: bold;
        margin: 0 0 1 0;
    }}

    WelcomeScreen .tip-text {{
        color: {_DIM};
    }}

    WelcomeScreen .separator {{
        color: {_GREEN};
        margin: 1 0;
    }}
    """

    def __init__(
        self,
        version: str = "dev",
        runtime_url: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(**kwargs)
        self._version = version
        self._runtime_url = runtime_url

    def compose(self) -> ComposeResult:
        """Build the welcome screen layout."""
        with Horizontal(id="welcome-layout"):
            # Left side: Logo + version info
            with Vertical(id="logo-section"):
                yield Static(_LOGO_ART, classes="logo-art")
                yield Static("")
                yield Static(f"GovOn v{self._version}", classes="version-text")
                yield Static(_mode_label(self._runtime_url), classes="mode-text")
                yield Static(_python_version(), classes="mode-text")

            # Right side: Tips and guide
            with Vertical(id="info-section"):
                yield Static("\uc2dc\uc791 \uac00\uc774\ub4dc", classes="section-title")
                yield Static(
                    "\uc9c8\ubb38\uc744 \uc785\ub825\ud558\uba74 AI \uc5d0\uc774\uc804\ud2b8\uac00 "
                    "\ubd84\uc11d\ud558\uace0 \ub3c4\uad6c\ub97c \uc0ac\uc6a9\ud569\ub2c8\ub2e4",
                    classes="tip-text",
                )
                yield Static(
                    "/help \ub85c \uba85\ub839\uc5b4 \ubaa9\ub85d \ud655\uc778",
                    classes="tip-text",
                )
                yield Static(
                    "/exit \ub610\ub294 Ctrl+D \ub85c \uc885\ub8cc",
                    classes="tip-text",
                )
                yield Static("", classes="separator")
                yield Static(
                    "\uc2b9\uc778 \ud544\uc694 \ub3c4\uad6c\ub294 \uc2e4\ud589 \uc804 "
                    "\ud655\uc778\uc744 \uc694\uccad\ud569\ub2c8\ub2e4",
                    classes="tip-text",
                )
                yield Static(
                    "Esc \ub85c \uc9c4\ud589 \uc911\uc778 \uc791\uc5c5 \ucde8\uc18c",
                    classes="tip-text",
                )
