"""approval_ui.py 순수 함수 단위 테스트.

Issue #493: _display_width, _build_box_lines, _fallback_prompt 함수에 대한
경계값 포함 단위 테스트.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.cli.approval_ui import _BOX_WIDTH, _build_box_lines, _display_width, _fallback_prompt


class TestDisplayWidth:
    """_display_width — ASCII/CJK/혼합 문자열 너비 계산 테스트."""

    def test_empty_string(self):
        assert _display_width("") == 0

    def test_ascii_only(self):
        assert _display_width("hello") == 5
        assert _display_width("A") == 1

    def test_cjk_wide_chars(self):
        # 한글, 한자, 일본어는 east_asian_width W 또는 F → 각 2
        assert _display_width("안녕") == 4  # 2자 × 2 = 4
        assert _display_width("한") == 2
        assert _display_width("漢字") == 4

    def test_mixed_ascii_cjk(self):
        # "a안" = 1 + 2 = 3
        assert _display_width("a안") == 3
        # "ok가나" = 1+1+2+2 = 6
        assert _display_width("ok가나") == 6

    def test_space_is_single_width(self):
        assert _display_width("  ") == 2

    def test_digit_is_single_width(self):
        assert _display_width("123") == 3

    def test_fullwidth_latin(self):
        # 전각 라틴 문자 (Fullwidth, east_asian_width F) → 2
        assert _display_width("\uff21") == 2  # Ａ (FULLWIDTH LATIN CAPITAL LETTER A)


class TestBuildBoxLines:
    """_build_box_lines — approval box 렌더링 테스트."""

    def _base_request(self, **kwargs):
        base = {"goal": "테스트 목표", "reason": "테스트 이유", "tool_summaries": []}
        base.update(kwargs)
        return base

    def test_returns_list_of_strings(self):
        lines = _build_box_lines(self._base_request(), selected=0)
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)

    def test_has_top_and_bottom_border(self):
        lines = _build_box_lines(self._base_request(), selected=0)
        assert lines[0].startswith("┌")
        assert lines[-1].startswith("└")

    def test_minimum_lines_count(self):
        # top, empty, goal, reason, empty, approve, reject, bottom = 최소 8줄
        lines = _build_box_lines(self._base_request(), selected=0)
        assert len(lines) >= 8

    def test_selected_0_shows_approve_filled(self):
        lines = _build_box_lines(self._base_request(), selected=0)
        # 선택 0 → 승인 줄에 ● 표시
        approve_line = next((l for l in lines if "● 승인" in l), None)
        assert approve_line is not None, "승인 줄(●)이 없습니다"

    def test_selected_1_shows_reject_filled(self):
        lines = _build_box_lines(self._base_request(), selected=1)
        # 선택 1 → 거절 줄에 ● 표시
        reject_line = next((l for l in lines if "● 거절" in l), None)
        assert reject_line is not None, "거절 줄(●)이 없습니다"

    def test_empty_goal_and_reason(self):
        req = {"goal": "", "reason": "", "tool_summaries": []}
        lines = _build_box_lines(req, selected=0)
        assert len(lines) >= 6  # 최소 구조 유지

    def test_tool_summaries_appear(self):
        req = self._base_request(tool_summaries=["rag_search 실행", "api_lookup 실행"])
        lines = _build_box_lines(req, selected=0)
        content = "\n".join(lines)
        assert "rag_search 실행" in content
        assert "api_lookup 실행" in content

    def test_empty_tool_summaries(self):
        req = self._base_request(tool_summaries=[])
        lines = _build_box_lines(req, selected=0)
        # "수행할 작업:" 섹션이 없어야 한다
        content = "\n".join(lines)
        assert "수행할 작업" not in content

    def test_long_goal_wraps_into_multiple_lines(self):
        # _BOX_WIDTH보다 훨씬 긴 목표 문자열 → 여러 줄로 분할
        long_goal = "가" * 100  # CJK 100자 = 200 display columns (> _BOX_WIDTH)
        req = self._base_request(goal=long_goal)
        lines = _build_box_lines(req, selected=0)
        # 최소 1줄 이상 추가로 생성돼야 한다
        assert len(lines) > 8

    def test_box_line_width_within_bounds(self):
        # 각 본문 줄(│…│)의 표시 너비는 _BOX_WIDTH + 4 (│ + space + content + space + │) 이내
        req = self._base_request(tool_summaries=["도구 실행"])
        lines = _build_box_lines(req, selected=0)
        _limit = _BOX_WIDTH + 6
        for line in lines:
            if line.startswith("│"):
                assert _display_width(line) <= _limit, (
                    f"줄 너비 초과: {_display_width(line)} > {_limit}: {line!r}"
                )

    def test_none_tool_summaries(self):
        req = {"goal": "목표", "reason": "이유", "tool_summaries": None}
        lines = _build_box_lines(req, selected=0)
        assert len(lines) >= 6


class TestFallbackPrompt:
    """_fallback_prompt — y/n 입력 케이스 테스트."""

    def test_yes_returns_true(self):
        with patch("builtins.input", return_value="y"):
            assert _fallback_prompt({"goal": "g", "reason": "r"}) is True

    def test_no_returns_false(self):
        with patch("builtins.input", return_value="n"):
            assert _fallback_prompt({"goal": "g", "reason": "r"}) is False

    def test_korean_yes_returns_true(self):
        with patch("builtins.input", return_value="예"):
            assert _fallback_prompt({}) is True

    def test_korean_ne_returns_true(self):
        with patch("builtins.input", return_value="네"):
            assert _fallback_prompt({}) is True

    def test_empty_returns_false(self):
        with patch("builtins.input", return_value=""):
            assert _fallback_prompt({}) is False

    def test_eof_returns_false(self):
        with patch("builtins.input", side_effect=EOFError):
            assert _fallback_prompt({}) is False

    def test_keyboard_interrupt_returns_false(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _fallback_prompt({}) is False

    def test_full_word_yes(self):
        with patch("builtins.input", return_value="yes"):
            assert _fallback_prompt({}) is True
