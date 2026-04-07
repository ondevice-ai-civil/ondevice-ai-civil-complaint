"""CLI approval UI terminal layout tests for issue #504."""

from __future__ import annotations

from unittest.mock import patch

from src.cli import approval_ui


def _sample_request() -> dict:
    return {
        "task_type": "draft_response",
        "goal": "도로 파손 민원을 담당 부서에 전달하고 처리 상태를 안내합니다",
        "reason": "현장 점검과 조치 일정 확인이 필요합니다",
        "tool_summaries": [
            "도로 파손 관련 민원 처리 근거를 확인합니다",
            "유사 사례와 담당 부서 정보를 함께 조회합니다",
        ],
    }


def test_get_approval_panel_width_scales_with_terminal_columns():
    """승인 패널은 좁은/넓은 터미널에서 반응형 폭을 유지한다."""
    assert approval_ui._get_approval_panel_width(40) == 38
    assert approval_ui._get_approval_panel_width(80) == 59
    assert approval_ui._get_approval_panel_width(120) == 59


def test_build_approval_panel_uses_task_type_style_and_selection():
    """Rich Panel 렌더링은 task_type 색상과 선택 상태를 반영한다."""

    class FakeText:
        def __init__(self, text="", style=None):
            self.plain = text
            self.style = style

        def append(self, text, style=None):
            self.plain += text

    class FakeTable:
        def __init__(self, *args, **kwargs):
            self.rows = []
            self.columns = []

        @classmethod
        def grid(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        def add_column(self, *args, **kwargs):
            self.columns.append((args, kwargs))

        def add_row(self, *cells):
            self.rows.append(cells)

    class FakeGroup:
        def __init__(self, *renderables):
            self.renderables = renderables

    class FakePanel:
        def __init__(self, renderable, title=None, border_style=None, width=None, padding=None):
            self.renderable = renderable
            self.title = title
            self.border_style = border_style
            self.width = width
            self.padding = padding

    with patch.object(approval_ui, "Text", FakeText, create=True):
        with patch.object(approval_ui, "Table", FakeTable, create=True):
            with patch.object(approval_ui, "Group", FakeGroup, create=True):
                with patch.object(approval_ui, "Panel", FakePanel, create=True):
                    panel = approval_ui._build_approval_panel(
                        _sample_request(),
                        selected=0,
                        columns=80,
                    )

    summary = panel.renderable.renderables[0]
    choices = panel.renderable.renderables[2]
    assert panel.width == 59
    assert panel.border_style == approval_ui._get_task_type_style("draft_response")
    assert panel.title.plain == "작업 승인 요청"
    assert summary.rows[0][0] == "유형"
    assert summary.rows[0][1].plain == approval_ui._get_task_type_label("draft_response")
    assert choices.rows[0][0].plain == "● 승인"
    assert choices.rows[0][0].style == "bold green"
    assert choices.rows[1][0].plain == "○ 거절"
    assert choices.rows[1][0].style == "dim white"


def test_show_approval_prompt_warns_and_falls_back_on_narrow_terminal(capsys):
    """40열 미만에서는 경고 후 plain fallback으로 전환한다."""
    with patch("src.cli.approval_ui._PT_AVAILABLE", True):
        with patch("src.cli.approval_ui._RICH_AVAILABLE", True):
            with patch("src.cli.approval_ui.get_terminal_columns", return_value=30):
                with patch(
                    "src.cli.approval_ui._fallback_prompt",
                    return_value=True,
                ) as mock_fallback:
                    with patch("src.cli.approval_ui._pt_prompt") as mock_pt_prompt:
                        assert approval_ui.show_approval_prompt(_sample_request()) is True

    captured = capsys.readouterr()
    assert "plain mode" in captured.out
    assert "최소 40열" in captured.out
    mock_fallback.assert_called_once()
    mock_pt_prompt.assert_not_called()


def test_show_approval_prompt_falls_back_when_rich_is_unavailable():
    """prompt_toolkit가 있어도 Rich가 없으면 plain fallback으로 전환한다."""
    with patch("src.cli.approval_ui._PT_AVAILABLE", True):
        with patch("src.cli.approval_ui._RICH_AVAILABLE", False):
            with patch("src.cli.approval_ui.get_terminal_columns", return_value=80):
                with patch(
                    "src.cli.approval_ui._fallback_prompt",
                    return_value=True,
                ) as mock_fallback:
                    with patch("src.cli.approval_ui._pt_prompt") as mock_pt_prompt:
                        assert approval_ui.show_approval_prompt(_sample_request()) is True

    mock_fallback.assert_called_once_with(_sample_request(), columns=80)
    mock_pt_prompt.assert_not_called()


def test_show_approval_prompt_reuses_resolved_columns_for_pt_prompt():
    """prompt_toolkit 경로는 이미 구한 터미널 폭을 다시 전달한다."""
    with patch("src.cli.approval_ui._PT_AVAILABLE", True):
        with patch("src.cli.approval_ui._RICH_AVAILABLE", True):
            with patch("src.cli.approval_ui.get_terminal_columns", return_value=80):
                with patch("src.cli.approval_ui._pt_prompt", return_value=True) as mock_pt_prompt:
                    assert approval_ui.show_approval_prompt(_sample_request()) is True

    mock_pt_prompt.assert_called_once_with(_sample_request(), columns=80)


def test_fallback_prompt_uses_terminal_width_for_separator(capsys):
    """plain fallback separator도 현재 터미널 폭을 넘지 않는다."""
    with patch("builtins.input", return_value="n"):
        approval_ui._fallback_prompt(_sample_request(), columns=32)

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    title_line = lines[0]
    type_line = lines[1]
    separator_lines = [line for line in lines if set(line) == {"─"}]

    assert "작업 승인 요청" in title_line
    assert "답변 초안 작성" in type_line
    assert separator_lines
    max_width = max(approval_ui._display_width(line) for line in [title_line, *separator_lines])
    assert max_width <= 32
