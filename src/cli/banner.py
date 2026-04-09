"""Startup banner for the GovOn CLI.

Renders an ASCII art 'G' logo (inspired by govon_mark.png: dark green background,
cream bold-italic 'G') alongside build metadata.  Three size variants are chosen
based on the current terminal width:

  LARGE   — terminals ≥ 80 columns
  COMPACT — terminals ≥ 50 columns
  TINY    — terminals < 50 columns (single-line fallback)

Rich is used for colored output when available; plain print() is the fallback.
"""

from __future__ import annotations

import sys
from urllib.parse import urlparse

from src.cli.terminal import get_terminal_columns
from src.cli.theme import get_theme

try:
    from rich.console import Console
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

# ---------------------------------------------------------------------------
# ASCII art definitions
# ---------------------------------------------------------------------------

# LARGE variant — for terminals ≥ 80 columns.
# Each line of the logo is stored separately so metadata can be injected
# beside specific rows at render time.
_LARGE_LOGO_LINES: tuple[str, ...] = (
    "  ██████╗ ",
    " ██╔════╝ ",
    " ██║ ████╗",
    " ██║ ╚═██║",
    " ╚██████╔╝",
    "  ╚═════╝ ",
)

# Indices of logo rows beside which metadata lines are placed (0-based).
_LARGE_META_ROWS: tuple[int, ...] = (2, 3)

# COMPACT variant — for terminals ≥ 50 columns.
_COMPACT_LOGO_LINES: tuple[str, ...] = (
    " ▄██████▄ ",
    " ██╔═══██║",
    " ╚██████╔╝",
)

_COMPACT_META_ROWS: tuple[int, ...] = (0, 1)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mode_label(mode: str, runtime_url: str | None) -> str:
    """Return a human-readable mode string for display next to the logo.

    Args:
        mode: Either ``"local"`` or ``"remote"``.
        runtime_url: The full runtime URL when *mode* is ``"remote"``; ignored
            for local mode.  Only the hostname is shown.

    Returns:
        A short descriptive string, e.g. ``"로컬 모드"`` or ``"원격: api.example.com"``.
    """
    if mode == "remote" and runtime_url:
        try:
            host = urlparse(runtime_url).hostname or runtime_url
        except Exception:
            host = runtime_url
        return f"원격: {host}"
    if mode == "remote":
        return "원격 모드"
    return "로컬 모드"


def _python_version_short() -> str:
    """Return a condensed Python version string, e.g. ``'Python 3.12'``."""
    major, minor = sys.version_info[:2]
    return f"Python {major}.{minor}"


def _inject_meta(
    logo_lines: tuple[str, ...],
    meta_rows: tuple[int, ...],
    meta_lines: list[str],
    separator: str = "  ",
) -> list[str]:
    """Combine logo lines with metadata lines at the designated row indices.

    Args:
        logo_lines: Tuple of pre-formatted logo row strings (equal visual width).
        meta_rows: Indices of logo lines beside which metadata should appear.
        meta_lines: Metadata strings to inject (one per meta row).
        separator: String placed between the logo column and the metadata column.

    Returns:
        List of combined strings, one per logo row.
    """
    result: list[str] = []
    meta_iter = iter(meta_lines)
    for idx, line in enumerate(logo_lines):
        if idx in meta_rows:
            meta = next(meta_iter, "")
            result.append(f"{line}{separator}{meta}")
        else:
            result.append(line)
    return result


# ---------------------------------------------------------------------------
# Rich-based rendering
# ---------------------------------------------------------------------------


def _rich_large_banner(version: str, mode: str, runtime_url: str | None) -> None:
    """Render the LARGE banner using Rich styled text."""
    theme = get_theme()
    console = Console()

    meta_lines = [
        f"GovOn v{version}",
        _mode_label(mode, runtime_url),
        _python_version_short(),
    ]

    # Build a combined text object line-by-line so we can apply per-segment styles.
    combined = Text()
    for idx, logo_line in enumerate(_LARGE_LOGO_LINES):
        combined.append(logo_line, style=theme.brand_primary)
        if idx in _LARGE_META_ROWS:
            meta_index = _LARGE_META_ROWS.index(idx)
            if meta_index < len(meta_lines):
                combined.append("  ")
                combined.append(meta_lines[meta_index], style=theme.brand_accent)
        combined.append("\n")

    # Append any remaining metadata below the logo when there are more meta
    # lines than injection slots (here: Python version on the line after the logo).
    logo_end_padding = " " * len(_LARGE_LOGO_LINES[0])
    if len(meta_lines) > len(_LARGE_META_ROWS):
        for extra_meta in meta_lines[len(_LARGE_META_ROWS) :]:
            combined.append(logo_end_padding)
            combined.append("  ")
            combined.append(extra_meta, style=theme.text_secondary)
            combined.append("\n")

    console.print(combined)


def _rich_compact_banner(version: str, mode: str, runtime_url: str | None) -> None:
    """Render the COMPACT banner using Rich styled text."""
    theme = get_theme()
    console = Console()

    meta_lines = [
        f"GovOn v{version}",
        _mode_label(mode, runtime_url),
    ]

    combined = Text()
    for idx, logo_line in enumerate(_COMPACT_LOGO_LINES):
        combined.append(logo_line, style=theme.brand_primary)
        if idx in _COMPACT_META_ROWS:
            meta_index = _COMPACT_META_ROWS.index(idx)
            if meta_index < len(meta_lines):
                combined.append("  ")
                combined.append(meta_lines[meta_index], style=theme.brand_accent)
        combined.append("\n")

    console.print(combined)


def _rich_tiny_banner(version: str, mode: str, runtime_url: str | None) -> None:
    """Render the TINY single-line banner using Rich styled text."""
    theme = get_theme()
    console = Console()

    line = Text()
    line.append("✦ ", style=theme.brand_primary)
    line.append(f"GovOn v{version}", style=theme.brand_accent)
    line.append(" — ", style=theme.text_secondary)
    line.append(_mode_label(mode, runtime_url), style=theme.text_secondary)

    console.print(line)


# ---------------------------------------------------------------------------
# Plain-text fallback rendering
# ---------------------------------------------------------------------------


def _plain_large_banner(version: str, mode: str, runtime_url: str | None) -> None:
    """Render the LARGE banner without Rich (plain print)."""
    meta_lines = [
        f"GovOn v{version}",
        _mode_label(mode, runtime_url),
        _python_version_short(),
    ]
    lines = _inject_meta(_LARGE_LOGO_LINES, _LARGE_META_ROWS, meta_lines[:2])
    for line in lines:
        print(line)
    # Remaining metadata below logo
    logo_pad = " " * len(_LARGE_LOGO_LINES[0])
    for extra in meta_lines[len(_LARGE_META_ROWS) :]:
        print(f"{logo_pad}  {extra}")


def _plain_compact_banner(version: str, mode: str, runtime_url: str | None) -> None:
    """Render the COMPACT banner without Rich (plain print)."""
    meta_lines = [
        f"GovOn v{version}",
        _mode_label(mode, runtime_url),
    ]
    lines = _inject_meta(_COMPACT_LOGO_LINES, _COMPACT_META_ROWS, meta_lines)
    for line in lines:
        print(line)


def _plain_tiny_banner(version: str, mode: str, runtime_url: str | None) -> None:
    """Render the TINY single-line banner without Rich (plain print)."""
    print(f"✦ GovOn v{version} — {_mode_label(mode, runtime_url)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_banner(
    version: str,
    mode: str = "local",
    runtime_url: str | None = None,
) -> None:
    """Print the GovOn startup banner to stdout.

    The banner variant (LARGE / COMPACT / TINY) is chosen automatically based
    on the current terminal width.  When Rich is installed the logo is rendered
    in brand colors (``brand_primary`` for the 'G' art, ``brand_accent`` for
    version/mode metadata).  When Rich is unavailable, plain ASCII is printed.

    Args:
        version: Application version string supplied by the caller, e.g.
            ``"1.2.3"``.  Typically read from ``pyproject.toml`` at startup.
        mode: Either ``"local"`` (default) or ``"remote"``.
        runtime_url: Full URL of the remote runtime endpoint.  Only used when
            *mode* is ``"remote"``; the hostname portion is extracted and shown
            to the user.  Pass ``None`` when running in local mode.

    Examples::

        render_banner("1.0.0")
        render_banner("1.0.0", mode="remote", runtime_url="https://api.govon.kr")
    """
    columns = get_terminal_columns()

    if columns >= 80:
        size = "large"
    elif columns >= 50:
        size = "compact"
    else:
        size = "tiny"

    if _RICH_AVAILABLE:
        _dispatch = {
            "large": _rich_large_banner,
            "compact": _rich_compact_banner,
            "tiny": _rich_tiny_banner,
        }
    else:
        _dispatch = {
            "large": _plain_large_banner,
            "compact": _plain_compact_banner,
            "tiny": _plain_tiny_banner,
        }

    _dispatch[size](version, mode, runtime_url)
