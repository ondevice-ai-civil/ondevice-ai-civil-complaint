"""Terminal layout helpers for the GovOn CLI."""

from __future__ import annotations

import shutil

DEFAULT_TERMINAL_COLUMNS = 80
MIN_TERMINAL_COLUMNS = 40
MIN_CONTENT_WIDTH = 20
APPROVAL_BOX_MAX_WIDTH = 55
APPROVAL_BOX_MARGIN = 4
PANEL_MARGIN = 2


def get_terminal_columns(default: int = DEFAULT_TERMINAL_COLUMNS) -> int:
    """Return the current terminal width in columns."""
    return max(shutil.get_terminal_size(fallback=(default, 24)).columns, 1)


def is_layout_supported(columns: int | None = None) -> bool:
    """Return True when the terminal is wide enough for full rich layouts."""
    current_columns = get_terminal_columns() if columns is None else columns
    return current_columns >= MIN_TERMINAL_COLUMNS


def get_approval_box_width(columns: int | None = None) -> int:
    """Return the inner width for the approval box."""
    current_columns = get_terminal_columns() if columns is None else columns
    return max(
        MIN_CONTENT_WIDTH, min(APPROVAL_BOX_MAX_WIDTH, current_columns - APPROVAL_BOX_MARGIN)
    )


def get_panel_width(columns: int | None = None) -> int:
    """Return the rich panel width for result rendering."""
    current_columns = get_terminal_columns() if columns is None else columns
    return max(MIN_CONTENT_WIDTH, current_columns - PANEL_MARGIN)


def get_narrow_terminal_warning(columns: int | None = None) -> str:
    """Return the warning shown when the terminal is too narrow."""
    current_columns = get_terminal_columns() if columns is None else columns
    return (
        f"터미널 너비가 {current_columns}열로 좁아 plain mode로 전환합니다. "
        f"최소 {MIN_TERMINAL_COLUMNS}열 이상에서 전체 레이아웃이 보장됩니다."
    )
