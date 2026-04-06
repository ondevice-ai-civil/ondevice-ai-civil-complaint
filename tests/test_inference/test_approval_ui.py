"""CLI approval UI terminal layout tests for issue #504."""

from __future__ import annotations

from unittest.mock import patch

from src.cli import approval_ui


def _sample_request() -> dict:
    return {
        "goal": "도로 파손 민원을 담당 부서에 전달하고 처리 상태를 안내합니다",
        "reason": "현장 점검과 조치 일정 확인이 필요합니다",
        "tool_summaries": [
            "도로 파손 관련 민원 처리 근거를 확인합니다",
            "유사 사례와 담당 부서 정보를 함께 조회합니다",
        ],
    }


def test_build_box_lines_fit_within_40_columns():
    """40열 터미널에서도 승인 박스가 줄바꿈되며 폭을 넘지 않는다."""
    with patch("src.cli.approval_ui.get_terminal_columns", return_value=40):
        lines = approval_ui._build_box_lines(_sample_request(), selected=0)

    max_width = max(approval_ui._display_width(line) for line in lines)
    assert max_width <= 40
    assert any("목표" in line for line in lines)
    assert any("승인" in line for line in lines)


def test_build_box_lines_fit_within_80_columns():
    """80열 터미널에서는 승인 박스가 여유 있게 렌더링된다."""
    with patch("src.cli.approval_ui.get_terminal_columns", return_value=80):
        lines = approval_ui._build_box_lines(_sample_request(), selected=0)

    max_width = max(approval_ui._display_width(line) for line in lines)
    assert max_width <= 80


def test_build_box_lines_fit_within_120_columns_and_keep_max_width():
    """120열 터미널에서도 승인 박스는 최대 폭 55를 유지하며 깨지지 않는다."""
    with patch("src.cli.approval_ui.get_terminal_columns", return_value=120):
        lines = approval_ui._build_box_lines(_sample_request(), selected=0)

    max_width = max(approval_ui._display_width(line) for line in lines)
    assert max_width <= 61


def test_show_approval_prompt_warns_and_falls_back_on_narrow_terminal(capsys):
    """40열 미만에서는 경고 후 plain fallback으로 전환한다."""
    with patch("src.cli.approval_ui._PT_AVAILABLE", True):
        with patch("src.cli.approval_ui.get_terminal_columns", return_value=30):
            with patch("src.cli.approval_ui._fallback_prompt", return_value=True) as mock_fallback:
                with patch("src.cli.approval_ui._pt_prompt") as mock_pt_prompt:
                    assert approval_ui.show_approval_prompt(_sample_request()) is True

    captured = capsys.readouterr()
    assert "plain mode" in captured.out
    assert "최소 40열" in captured.out
    mock_fallback.assert_called_once()
    mock_pt_prompt.assert_not_called()


def test_fallback_prompt_uses_terminal_width_for_separator(capsys):
    """plain fallback separator도 현재 터미널 폭을 넘지 않는다."""
    with patch("builtins.input", return_value="n"):
        approval_ui._fallback_prompt(_sample_request(), columns=32)

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    separator_lines = [line for line in lines if set(line) == {"─"}]
    assert separator_lines
    max_width = max(approval_ui._display_width(line) for line in separator_lines)
    assert max_width <= 32
