"""Terminal helper tests for CLI responsive layout logic."""

from __future__ import annotations

from src.cli import terminal


def test_is_layout_supported_respects_minimum_columns():
    """40열 이상부터 full layout을 허용한다."""
    assert terminal.is_layout_supported(39) is False
    assert terminal.is_layout_supported(40) is True


def test_get_approval_box_width_clamps_to_minimum_and_maximum():
    """approval box width는 최소/최대 범위를 유지한다."""
    assert terminal.get_approval_box_width(1) == terminal.MIN_CONTENT_WIDTH
    assert terminal.get_approval_box_width(40) == 36
    assert terminal.get_approval_box_width(200) == terminal.APPROVAL_BOX_MAX_WIDTH


def test_get_panel_width_tracks_terminal_minus_margin():
    """result panel width는 terminal margin만 차감한다."""
    assert terminal.get_panel_width(40) == 38
    assert terminal.get_panel_width(200) == 198


def test_get_narrow_terminal_warning_includes_current_width():
    """narrow warning은 현재 width와 최소 요구 width를 함께 안내한다."""
    warning = terminal.get_narrow_terminal_warning(30)

    assert "30열" in warning
    assert str(terminal.MIN_TERMINAL_COLUMNS) in warning
