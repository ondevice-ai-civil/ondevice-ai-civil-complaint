"""approval_ui.py 순수 함수 단위 테스트 (#493).

검증 범위:
  1. _display_width          — ASCII / CJK / 혼합 / 빈 문자열 / 전각 문자 폭 계산
  2. _get_task_type_style     — task_type → accent 색상 매핑
  3. _get_task_type_label     — task_type → 한국어 레이블 매핑
  4. _get_approval_panel_width — 터미널 폭별 반응형 패널 폭 계산
  5. _normalize_approval_request — v4 payload 키 정규화
  6. _build_choice_text       — 선택 상태에 따른 bullet/스타일 (Rich 필요)
  7. _fallback_prompt         — y/yes/예/네/n/EOFError/KeyboardInterrupt 입력 처리
  8. show_approval_prompt     — 좁은 터미널 경고·fallback 전환, 컬럼 전달
  9. _build_approval_panel    — task_type 색상·선택 상태 반영 (Rich mock)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.cli import approval_ui
from src.cli.approval_ui import (
    _display_width,
    _fallback_prompt,
    _get_approval_panel_width,
    _get_task_type_label,
    _get_task_type_style,
    _normalize_approval_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _minimal_request() -> dict:
    return {"goal": "test goal", "reason": "test reason"}


# ---------------------------------------------------------------------------
# Group 1: _display_width
# ---------------------------------------------------------------------------


def test_display_width_empty_string():
    """빈 문자열은 폭이 0이다."""
    assert _display_width("") == 0


def test_display_width_ascii_only():
    """순수 ASCII 문자열은 글자 수와 폭이 같다."""
    assert _display_width("hello") == 5, f"'hello' 폭 기대 5, 실제 {_display_width('hello')}"


def test_display_width_cjk_only():
    """한글 각 글자는 폭 2로 계산한다."""
    assert (
        _display_width("안녕하세요") == 10
    ), f"'안녕하세요' 폭 기대 10, 실제 {_display_width('안녕하세요')}"


def test_display_width_mixed_ascii_and_cjk():
    """ASCII와 한글 혼합 — 'a한b글c' = 1+2+1+2+1 = 7."""
    assert _display_width("a한b글c") == 7, f"'a한b글c' 폭 기대 7, 실제 {_display_width('a한b글c')}"


def test_display_width_fullwidth_latin():
    """전각 라틴 문자(east_asian_width=F)는 폭 2로 계산한다."""
    assert (
        _display_width("Ａ") == 2
    ), f"'Ａ'(U+FF21) 폭 기대 2, 실제 {_display_width('Ａ')}"  # eaw='F'


# ---------------------------------------------------------------------------
# Group 2: _get_task_type_style
# ---------------------------------------------------------------------------


def test_get_task_type_style_known_type():
    """알려진 task_type은 지정된 색상을 반환한다."""
    assert _get_task_type_style("draft_response") == "cyan", "draft_response 색상 기대 'cyan'"
    assert _get_task_type_style("revise_response") == "blue", "revise_response 색상 기대 'blue'"


def test_get_task_type_style_unknown_type():
    """알 수 없는 task_type은 기본 색상(cyan)을 반환한다."""
    result = _get_task_type_style("unknown_type")
    assert result == "cyan", f"unknown type 기본 색상 기대 'cyan', 실제 {result!r}"


def test_get_task_type_style_none():
    """None 입력 시 기본 색상을 반환한다."""
    result = _get_task_type_style(None)
    assert result == "cyan", f"None 입력 기본 색상 기대 'cyan', 실제 {result!r}"


# ---------------------------------------------------------------------------
# Group 3: _get_task_type_label
# ---------------------------------------------------------------------------


def test_get_task_type_label_known_type():
    """알려진 task_type은 한국어 레이블을 반환한다."""
    assert (
        _get_task_type_label("draft_response") == "답변 초안 작성"
    ), "draft_response 레이블 기대 '답변 초안 작성'"
    assert (
        _get_task_type_label("lookup_stats") == "통계 조회"
    ), "lookup_stats 레이블 기대 '통계 조회'"


def test_get_task_type_label_unknown_type():
    """알 수 없는 task_type은 기본 레이블을 반환한다."""
    result = _get_task_type_label("unknown")
    assert result == "일반 작업", f"unknown type 기본 레이블 기대 '일반 작업', 실제 {result!r}"


def test_get_task_type_label_none():
    """None 입력 시 기본 레이블을 반환한다."""
    result = _get_task_type_label(None)
    assert result == "일반 작업", f"None 입력 기본 레이블 기대 '일반 작업', 실제 {result!r}"


# ---------------------------------------------------------------------------
# Group 4: _get_approval_panel_width
# ---------------------------------------------------------------------------


def test_get_approval_panel_width_narrow_terminal():
    """40열 터미널에서 패널 폭은 38(=40-2)이다."""
    result = _get_approval_panel_width(40)
    assert result == 38, f"40열 터미널 패널 폭 기대 38, 실제 {result}"


def test_get_approval_panel_width_normal_terminal():
    """80열 터미널에서 패널 폭은 59(=max_box 55 + 4)이다."""
    result = _get_approval_panel_width(80)
    assert result == 59, f"80열 터미널 패널 폭 기대 59, 실제 {result}"


def test_get_approval_panel_width_wide_terminal():
    """120열 이상에서도 최대 폭 59를 유지한다."""
    result = _get_approval_panel_width(120)
    assert result == 59, f"120열 터미널 패널 폭 기대 59, 실제 {result}"


# ---------------------------------------------------------------------------
# Group 5: _normalize_approval_request
# ---------------------------------------------------------------------------


def test_normalize_approval_request_v4_tools_to_tool_summaries():
    """v4 'tools' 키가 'tool_summaries'로 정규화된다."""
    req = {"tools": ["도구1", "도구2"]}
    result = _normalize_approval_request(req)
    assert result["tool_summaries"] == ["도구1", "도구2"], f"tool_summaries 정규화 실패: {result}"


def test_normalize_approval_request_v4_message_to_goal():
    """v4 'message' 키가 'goal'로 정규화된다."""
    req = {"message": "목표 내용"}
    result = _normalize_approval_request(req)
    assert result["goal"] == "목표 내용", f"goal 정규화 실패: {result}"


def test_normalize_approval_request_v4_approval_required_to_reason():
    """v4 'approval_required' 키가 'reason'으로 정규화된다."""
    req = {"approval_required": ["file_write", "shell_exec"]}
    result = _normalize_approval_request(req)
    assert "file_write" in result["reason"], f"reason에 'file_write' 없음: {result['reason']!r}"
    assert "shell_exec" in result["reason"], f"reason에 'shell_exec' 없음: {result['reason']!r}"


def test_normalize_approval_request_existing_keys_not_overwritten():
    """기존 키가 있으면 v4 정규화 키로 덮어쓰지 않는다."""
    req = {"goal": "기존 목표", "message": "v4 메시지"}
    result = _normalize_approval_request(req)
    assert result["goal"] == "기존 목표", f"기존 goal이 덮어씌워짐: {result['goal']!r}"


def test_normalize_approval_request_empty_dict():
    """빈 dict도 크래시 없이 처리된다."""
    result = _normalize_approval_request({})
    assert isinstance(result, dict), f"빈 dict 정규화 결과가 dict가 아님: {type(result)}"


# ---------------------------------------------------------------------------
# Group 6: _build_choice_text (Rich 필요)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not approval_ui._RICH_AVAILABLE, reason="rich 미설치")
def test_build_choice_text_selected():
    """selected=True이면 ● bullet과 bold 스타일이 적용된다."""
    text = approval_ui._build_choice_text("승인", selected=True, style="green")
    assert "●" in text.plain, f"selected=True bullet 기대 '●', 실제 {text.plain!r}"
    assert "승인" in text.plain, f"text에 '승인' 없음: {text.plain!r}"
    assert "bold" in text.style, f"selected 스타일에 'bold' 없음: {text.style!r}"


@pytest.mark.skipif(not approval_ui._RICH_AVAILABLE, reason="rich 미설치")
def test_build_choice_text_not_selected():
    """selected=False이면 ○ bullet과 dim 스타일이 적용된다."""
    text = approval_ui._build_choice_text("거절", selected=False, style="red")
    assert "○" in text.plain, f"selected=False bullet 기대 '○', 실제 {text.plain!r}"
    assert "거절" in text.plain, f"text에 '거절' 없음: {text.plain!r}"
    assert text.style == "dim white", f"미선택 스타일 기대 'dim white', 실제 {text.style!r}"


# ---------------------------------------------------------------------------
# Group 7: _fallback_prompt
# ---------------------------------------------------------------------------


def test_fallback_prompt_y_returns_true():
    """'y' 입력 시 True를 반환한다."""
    with patch("builtins.input", return_value="y"):
        assert _fallback_prompt(_minimal_request(), columns=40) is True


def test_fallback_prompt_yes_returns_true():
    """'yes' 입력 시 True를 반환한다."""
    with patch("builtins.input", return_value="yes"):
        assert _fallback_prompt(_minimal_request(), columns=40) is True


def test_fallback_prompt_korean_ye_returns_true():
    """'예' 입력 시 True를 반환한다."""
    with patch("builtins.input", return_value="예"):
        assert _fallback_prompt(_minimal_request(), columns=40) is True


def test_fallback_prompt_korean_ne_returns_true():
    """'네' 입력 시 True를 반환한다."""
    with patch("builtins.input", return_value="네"):
        assert _fallback_prompt(_minimal_request(), columns=40) is True


def test_fallback_prompt_uppercase_y_returns_true():
    """대소문자 무관: 'Y' 입력 시 True를 반환한다."""
    with patch("builtins.input", return_value="Y"):
        assert _fallback_prompt(_minimal_request(), columns=40) is True


def test_fallback_prompt_n_returns_false():
    """'n' 입력 시 False를 반환한다."""
    with patch("builtins.input", return_value="n"):
        assert _fallback_prompt(_minimal_request(), columns=40) is False


def test_fallback_prompt_empty_returns_false():
    """빈 입력(Enter만 누름) 시 False를 반환한다."""
    with patch("builtins.input", return_value=""):
        assert _fallback_prompt(_minimal_request(), columns=40) is False


def test_fallback_prompt_eoferror_returns_false():
    """파이프 입력 종료(EOF) 시 False를 반환한다."""
    with patch("builtins.input", side_effect=EOFError):
        assert _fallback_prompt(_minimal_request(), columns=40) is False


def test_fallback_prompt_keyboard_interrupt_returns_false():
    """Ctrl+C 인터럽트 시 False를 반환한다."""
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert _fallback_prompt(_minimal_request(), columns=40) is False


# ---------------------------------------------------------------------------
# Group 8: show_approval_prompt 통합
# ---------------------------------------------------------------------------


def test_show_approval_prompt_warns_and_falls_back_on_narrow_terminal(capsys):
    """40열 미만 터미널에서 경고 출력 후 fallback으로 전환한다."""
    with (
        patch("src.cli.approval_ui._PT_AVAILABLE", True),
        patch("src.cli.approval_ui._RICH_AVAILABLE", True),
        patch("src.cli.approval_ui.get_terminal_columns", return_value=30),
        patch("src.cli.approval_ui._fallback_prompt", return_value=True) as mock_fallback,
        patch("src.cli.approval_ui._pt_prompt") as mock_pt_prompt,
    ):
        assert approval_ui.show_approval_prompt(_sample_request()) is True

    captured = capsys.readouterr()
    assert "plain mode" in captured.out, f"경고 메시지에 'plain mode' 없음: {captured.out!r}"
    assert "최소 40열" in captured.out, f"경고 메시지에 '최소 40열' 없음: {captured.out!r}"
    mock_fallback.assert_called_once()
    mock_pt_prompt.assert_not_called()


def test_show_approval_prompt_falls_back_when_rich_is_unavailable():
    """prompt_toolkit가 있어도 Rich가 없으면 plain fallback으로 전환한다."""
    with (
        patch("src.cli.approval_ui._PT_AVAILABLE", True),
        patch("src.cli.approval_ui._RICH_AVAILABLE", False),
        patch("src.cli.approval_ui.get_terminal_columns", return_value=80),
        patch("src.cli.approval_ui._fallback_prompt", return_value=True) as mock_fallback,
        patch("src.cli.approval_ui._pt_prompt") as mock_pt_prompt,
    ):
        result = approval_ui.show_approval_prompt(_sample_request())
        assert result is True, f"Rich 없을 때 show_approval_prompt 기대 True, 실제 {result}"

    mock_fallback.assert_called_once_with(_sample_request(), columns=80)
    mock_pt_prompt.assert_not_called()


def test_show_approval_prompt_reuses_resolved_columns_for_pt_prompt():
    """prompt_toolkit 경로는 이미 구한 터미널 폭을 _pt_prompt에 전달한다."""
    with (
        patch("src.cli.approval_ui._PT_AVAILABLE", True),
        patch("src.cli.approval_ui._RICH_AVAILABLE", True),
        patch("src.cli.approval_ui.get_terminal_columns", return_value=80),
        patch("src.cli.approval_ui._pt_prompt", return_value=True) as mock_pt_prompt,
    ):
        assert approval_ui.show_approval_prompt(_sample_request()) is True

    mock_pt_prompt.assert_called_once_with(_sample_request(), columns=80)


def test_fallback_prompt_uses_terminal_width_for_separator(capsys):
    """plain fallback의 separator는 지정한 터미널 폭(32)을 초과하지 않는다."""
    with patch("builtins.input", return_value="n"):
        _fallback_prompt(_sample_request(), columns=32)

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    title_line = lines[0]
    separator_lines = [line for line in lines if set(line) == {"─"}]

    assert "작업 승인 요청" in title_line, f"title_line에 '작업 승인 요청' 없음: {title_line!r}"
    assert separator_lines, "separator 라인이 출력되지 않음"
    max_width = max(_display_width(line) for line in [title_line, *separator_lines])
    assert max_width <= 32, f"separator/title 폭 기대 ≤32, 실제 {max_width}"


# ---------------------------------------------------------------------------
# Group 9: _build_approval_panel (Rich mock)
# ---------------------------------------------------------------------------


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

    with (
        patch.object(approval_ui, "Text", FakeText, create=True),
        patch.object(approval_ui, "Table", FakeTable, create=True),
        patch.object(approval_ui, "Group", FakeGroup, create=True),
        patch.object(approval_ui, "Panel", FakePanel, create=True),
    ):
        panel = approval_ui._build_approval_panel(
            _sample_request(),
            selected=0,
            columns=80,
        )

    summary = panel.renderable.renderables[0]
    choices = panel.renderable.renderables[2]
    assert panel.width == 59, f"패널 폭 기대 59, 실제 {panel.width}"
    expected_style = approval_ui._get_task_type_style("draft_response")
    assert (
        panel.border_style == expected_style
    ), f"border_style 기대 {expected_style!r}, 실제 {panel.border_style!r}"
    assert (
        panel.title.plain == "작업 승인 요청"
    ), f"panel title 기대 '작업 승인 요청', 실제 {panel.title.plain!r}"
    assert (
        summary.rows[0][0] == "유형"
    ), f"summary 첫 행 레이블 기대 '유형', 실제 {summary.rows[0][0]!r}"
    expected_label = approval_ui._get_task_type_label("draft_response")
    assert (
        summary.rows[0][1].plain == expected_label
    ), f"task_type 레이블 기대 {expected_label!r}, 실제 {summary.rows[0][1].plain!r}"
    assert (
        choices.rows[0][0].plain == "● 승인"
    ), f"승인 bullet 기대 '● 승인', 실제 {choices.rows[0][0].plain!r}"
    assert (
        choices.rows[0][0].style == "bold green"
    ), f"승인 스타일 기대 'bold green', 실제 {choices.rows[0][0].style!r}"
    assert (
        choices.rows[1][0].plain == "○ 거절"
    ), f"거절 bullet 기대 '○ 거절', 실제 {choices.rows[1][0].plain!r}"
    assert (
        choices.rows[1][0].style == "dim white"
    ), f"거절 스타일 기대 'dim white', 실제 {choices.rows[1][0].style!r}"
