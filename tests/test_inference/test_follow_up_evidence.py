"""#161 후속 근거 보강 요청 UX 및 응답 렌더링 테스트.

검증 범위:
  1. synthesis._extract_final_text()가 append_evidence 타입에서 기존 답변 앞에 근거 추가
  2. _collect_evidence_items()가 RAG+API evidence items를 정확히 수집
  3. LangGraph 2턴 실행 — 초안 후 근거 요청 시 기존 답변이 유지됨
  4. render_result()가 [로컬 문서]/[외부 API] 라벨로 구조화된 출처 표시
  5. evidence_items 없을 때 기존 citations fallback 동작
  6. LLMPlannerAdapter._build_user_prompt()에 [이전 답변] 섹션 포함
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

os.environ.setdefault("SKIP_MODEL_LOAD", "true")

from src.cli.renderer import render_evidence_section, render_result
from src.inference.graph.builder import build_govon_graph
from src.inference.graph.executor_adapter import ExecutorAdapter
from src.inference.graph.nodes import _collect_evidence_items, _extract_final_text
from src.inference.graph.planner_adapter import LLMPlannerAdapter, RegexPlannerAdapter
from src.inference.session_context import SessionStore

# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

_SAMPLE_RAG_EVIDENCE = {
    "source_type": "rag",
    "title": "민원처리법.pdf",
    "excerpt": "민원은 접수일로부터 7일 이내에 처리해야 합니다.",
    "link_or_path": "/data/docs/민원처리법.pdf",
    "page": 23,
    "score": 0.92,
    "provider_meta": {"provider": "local_vectordb"},
}

_SAMPLE_API_EVIDENCE = {
    "source_type": "api",
    "title": "유사 민원 처리 사례",
    "excerpt": "도로 파손 민원은 접수 후 현장 조사를 거쳐 처리됩니다.",
    "link_or_path": "https://data.go.kr/example",
    "page": None,
    "score": 0.75,
    "provider_meta": {"provider": "data.go.kr"},
}


# ---------------------------------------------------------------------------
# Test 1: _extract_final_text() — append_evidence 타입에서 기존 답변 prepend
# ---------------------------------------------------------------------------


def test_synthesis_prepends_draft_for_append_evidence():
    """append_evidence 타입일 때 기존 답변 위에 근거 섹션을 추가한다."""
    previous_draft = "도로 파손 민원을 접수해드리겠습니다. 담당 부서로 전달하겠습니다."
    accumulated = {
        "previous_assistant_response": previous_draft,
        "append_evidence": {
            "success": True,
            "text": "",
            "evidence": {
                "status": "ok",
                "items": [_SAMPLE_RAG_EVIDENCE, _SAMPLE_API_EVIDENCE],
                "summary_text": "",
                "errors": [],
            },
        },
    }

    result = _extract_final_text(accumulated, "append_evidence")

    assert result.startswith(previous_draft), "기존 답변이 맨 앞에 와야 한다"
    assert "[참조 근거]" in result or "[로컬]" in result or "민원처리법" in result


def test_synthesis_append_evidence_no_previous_draft():
    """이전 답변 없이 append_evidence 타입이면 근거 섹션만 반환한다."""
    accumulated = {
        "rag_search": {
            "success": True,
            "evidence": {
                "status": "ok",
                "items": [_SAMPLE_RAG_EVIDENCE],
                "summary_text": "",
                "errors": [],
            },
        },
    }

    result = _extract_final_text(accumulated, "append_evidence")

    assert result  # 빈 문자열이 아니어야 한다
    assert "민원처리법" in result or "[참조 근거]" in result or "[로컬]" in result


def test_synthesis_draft_response_type_unaffected():
    """draft_response 타입은 기존 동작 그대로 — prepend 없이 직접 텍스트 반환."""
    accumulated = {
        "previous_assistant_response": "이전 답변 텍스트",
        "draft_civil_response": {
            "success": True,
            "text": "새로운 초안 답변입니다.",
        },
    }

    result = _extract_final_text(accumulated, "draft_response")

    assert result == "새로운 초안 답변입니다."
    assert "이전 답변 텍스트" not in result


# ---------------------------------------------------------------------------
# Test 2: _collect_evidence_items() — RAG+API items 수집
# ---------------------------------------------------------------------------


def test_collect_evidence_items_from_multiple_tools():
    """여러 tool 결과에서 evidence items를 수집하고 score 내림차순 정렬한다."""
    accumulated = {
        "rag_search": {
            "evidence": {
                "status": "ok",
                "items": [_SAMPLE_RAG_EVIDENCE],
                "errors": [],
            }
        },
        "api_lookup": {
            "evidence": {
                "status": "ok",
                "items": [_SAMPLE_API_EVIDENCE],
                "errors": [],
            }
        },
        "session_context": "세션 요약",  # 스킵되어야 함
        "query": "도로 파손",  # 스킵되어야 함
    }

    items = _collect_evidence_items(accumulated)

    assert len(items) == 2
    # score 내림차순: RAG(0.92) > API(0.75)
    assert items[0]["source_type"] == "rag"
    assert items[1]["source_type"] == "api"


def test_collect_evidence_items_empty():
    """evidence가 없는 accumulated에서는 빈 리스트를 반환한다."""
    accumulated = {
        "session_context": "요약",
        "draft_civil_response": {"success": True, "text": "답변"},
    }
    assert _collect_evidence_items(accumulated) == []


def test_collect_evidence_items_max_10():
    """evidence items는 최대 10개까지 반환한다."""
    many_items = [
        {"source_type": "rag", "title": f"doc{i}", "excerpt": "", "score": float(i)}
        for i in range(15)
    ]
    accumulated = {"rag_search": {"evidence": {"status": "ok", "items": many_items, "errors": []}}}

    items = _collect_evidence_items(accumulated)
    assert len(items) == 10
    # 가장 높은 score 10개 (14, 13, ..., 5)
    assert items[0]["score"] == 14.0


# ---------------------------------------------------------------------------
# Test 3: LangGraph 2턴 — 초안 후 근거 요청 흐름
# ---------------------------------------------------------------------------


class StubEvidenceExecutorAdapter(ExecutorAdapter):
    """테스트용 executor: tool별로 고정된 결과 반환."""

    def list_tools(self) -> list[str]:
        return ["rag_search", "api_lookup", "draft_civil_response", "append_evidence"]

    async def execute(self, tool_name: str, query: str, context: dict) -> dict:
        if tool_name == "draft_civil_response":
            return {
                "success": True,
                "text": "도로 파손 민원을 접수해드리겠습니다.",
                "latency_ms": 1.0,
            }
        if tool_name == "rag_search":
            return {
                "success": True,
                "text": "",
                "evidence": {
                    "status": "ok",
                    "items": [_SAMPLE_RAG_EVIDENCE],
                    "errors": [],
                },
                "latency_ms": 1.0,
            }
        if tool_name == "api_lookup":
            return {
                "success": True,
                "text": "",
                "evidence": {
                    "status": "ok",
                    "items": [_SAMPLE_API_EVIDENCE],
                    "errors": [],
                },
                "latency_ms": 1.0,
            }
        if tool_name == "append_evidence":
            return {
                "success": True,
                "text": "",
                "evidence": {
                    "status": "ok",
                    "items": [_SAMPLE_RAG_EVIDENCE, _SAMPLE_API_EVIDENCE],
                    "errors": [],
                },
                "latency_ms": 1.0,
            }
        return {"success": True, "text": f"[stub] {tool_name}", "latency_ms": 1.0}


@pytest.fixture
def evidence_graph():
    """근거 보강 테스트용 graph fixture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(db_path=os.path.join(tmpdir, "test.db"))
        planner = RegexPlannerAdapter()
        executor = StubEvidenceExecutorAdapter()
        checkpointer = MemorySaver()
        graph = build_govon_graph(
            session_store=store,
            planner_adapter=planner,
            executor_adapter=executor,
            checkpointer=checkpointer,
        )
        yield graph, store


@pytest.mark.asyncio
async def test_graph_follow_up_evidence_preserves_draft(evidence_graph):
    """2턴 그래프: 초안 후 근거 요청 시 Turn 1 초안이 Turn 2 final_text에 포함된다.

    검증:
      - Turn 1 final_text가 session store에 저장된다
      - Turn 2 session_load_node가 동일 session_id로 previous_assistant_response를 복원
      - Turn 2 final_text는 Turn 1 초안 + 근거 섹션으로 구성된다
      - Turn 2 evidence_items가 비어있지 않다
    """
    graph, store = evidence_graph

    # Turn 1과 Turn 2는 같은 session_id를 공유하지만
    # LangGraph checkpointer thread_id는 분리해 독립적인 그래프 실행을 보장한다
    session_id = "test-evidence-flow"
    TURN1_DRAFT = "도로 파손 민원을 접수해드리겠습니다."

    config = {"configurable": {"thread_id": session_id}}

    # Turn 1: 초안 요청
    initial_state = {
        "session_id": session_id,
        "request_id": "req-1",
        "messages": [HumanMessage(content="도로 파손 민원 답변 작성해줘")],
    }
    await graph.ainvoke(initial_state, config)
    # approval interrupt 후 승인
    await graph.ainvoke(Command(resume={"approved": True}), config)

    turn1_state = (await graph.aget_state(config)).values
    assert (
        turn1_state.get("final_text") == TURN1_DRAFT
    ), "Turn 1 final_text가 stub 응답과 일치해야 한다"

    # Turn 2: 근거 보강 요청 — 같은 session_id로 previous_assistant_response 주입 확인
    config2 = {"configurable": {"thread_id": session_id + "-ev"}}
    initial_state2 = {
        "session_id": session_id,  # 같은 세션: persist_node가 Turn 1 응답을 저장했음
        "request_id": "req-2",
        "messages": [HumanMessage(content="근거를 보여줘")],
    }
    await graph.ainvoke(initial_state2, config2)
    await graph.ainvoke(Command(resume={"approved": True}), config2)

    turn2_state = (await graph.aget_state(config2)).values
    evidence_items = turn2_state.get("evidence_items", [])
    assert isinstance(evidence_items, list), "evidence_items는 리스트여야 한다"
    assert len(evidence_items) > 0, "Turn 2에서 근거 보강 후 evidence_items가 비어있으면 안 된다"

    final_text2 = turn2_state.get("final_text", "")
    assert TURN1_DRAFT in final_text2, (
        "Turn 2 final_text는 Turn 1 초안을 포함해야 한다 "
        "(session_load → previous_assistant_response → _extract_final_text prepend 검증)"
    )


# ---------------------------------------------------------------------------
# Test 4: render_result() — 구조화된 근거 표시
# ---------------------------------------------------------------------------


def test_render_result_with_evidence_items(capsys):
    """evidence_items가 있으면 [로컬 문서]/[외부 API] 라벨로 표시한다."""
    result = {
        "text": "도로 파손 민원을 접수해드리겠습니다.",
        "evidence_items": [_SAMPLE_RAG_EVIDENCE, _SAMPLE_API_EVIDENCE],
    }

    with patch("src.cli.renderer._RICH_AVAILABLE", False):
        render_result(result)

    captured = capsys.readouterr()
    assert "[로컬 문서]" in captured.out
    assert "[외부 API]" in captured.out
    assert "민원처리법.pdf" in captured.out
    assert "유사 민원 처리 사례" in captured.out


def test_render_evidence_section_rag_shows_page_and_score():
    """render_evidence_section()이 RAG 항목에 page와 score를 포함한다."""
    text = render_evidence_section([_SAMPLE_RAG_EVIDENCE])

    assert "[로컬 문서]" in text
    assert "민원처리법.pdf" in text
    assert "p.23" in text
    assert "0.92" in text


def test_render_evidence_section_api_shows_url():
    """render_evidence_section()이 API 항목에 URL을 포함한다."""
    text = render_evidence_section([_SAMPLE_API_EVIDENCE])

    assert "[외부 API]" in text
    assert "data.go.kr" in text


def test_render_evidence_section_empty_returns_empty():
    """빈 evidence_items는 빈 문자열을 반환한다."""
    assert render_evidence_section([]) == ""


# ---------------------------------------------------------------------------
# Test 5: render_result() — evidence_items 없을 때 citations fallback
# ---------------------------------------------------------------------------


def test_render_backward_compat_citations(capsys):
    """evidence_items가 없을 때 기존 citations 리스트 방식으로 표시한다."""
    result = {
        "text": "답변 텍스트",
        "citations": ["출처1", "출처2"],
    }

    with patch("src.cli.renderer._RICH_AVAILABLE", False):
        render_result(result)

    captured = capsys.readouterr()
    assert "출처1" in captured.out
    assert "출처2" in captured.out


def test_render_result_no_evidence_no_citations(capsys):
    """evidence도 citations도 없을 때 텍스트만 표시한다."""
    result = {"text": "단순 답변"}

    with patch("src.cli.renderer._RICH_AVAILABLE", False):
        render_result(result)

    captured = capsys.readouterr()
    assert "단순 답변" in captured.out
    assert "[로컬 문서]" not in captured.out


def test_render_result_rich_panel_uses_explicit_terminal_width():
    """Rich Panel 렌더링은 현재 터미널 폭을 기준으로 width를 명시한다."""
    from src.cli import renderer

    mock_console = MagicMock()
    with patch.object(renderer, "_RICH_AVAILABLE", True):
        with patch.object(renderer, "_console", mock_console):
            with patch.object(renderer, "get_terminal_columns", return_value=80):
                renderer.render_result({"text": "답변 텍스트"})

    panel = mock_console.print.call_args.args[0]
    assert panel.width == 78


def test_render_result_rich_panel_uses_markdown_for_answer_body():
    """Rich 렌더링은 답변 본문을 Markdown renderable로 감싼다."""
    from src.cli import renderer

    class FakeMarkdown:
        def __init__(self, markup, code_theme=None):
            self.markup = markup
            self.code_theme = code_theme

    mock_console = MagicMock()
    markdown_text = "**굵게**\n- 목록\n`코드`\n```python\nprint('안내')\n```"
    with patch.object(renderer, "_RICH_AVAILABLE", True):
        with patch.object(renderer, "_console", mock_console):
            with patch.object(renderer, "Markdown", FakeMarkdown):
                with patch.object(renderer, "get_terminal_columns", return_value=80):
                    renderer.render_result({"text": markdown_text})

    panel = mock_console.print.call_args.args[0]
    assert isinstance(panel.renderable, FakeMarkdown)
    assert panel.renderable.markup == markdown_text
    assert panel.renderable.code_theme == renderer.MARKDOWN_CODE_THEME


def test_render_result_rich_panel_groups_markdown_with_evidence():
    """근거가 있으면 Markdown 본문과 근거 블록을 함께 묶어 렌더링한다."""
    from src.cli import renderer

    class FakeMarkdown:
        def __init__(self, markup, code_theme=None):
            self.markup = markup
            self.code_theme = code_theme

    class FakeGroup:
        def __init__(self, *renderables):
            self.renderables = renderables

    mock_console = MagicMock()
    result = {
        "text": "## 답변\n- 항목",
        "evidence_items": [_SAMPLE_RAG_EVIDENCE],
    }
    with patch.object(renderer, "_RICH_AVAILABLE", True):
        with patch.object(renderer, "_console", mock_console):
            with patch.object(renderer, "Markdown", FakeMarkdown):
                with patch.object(renderer, "Group", FakeGroup):
                    with patch.object(renderer, "get_terminal_columns", return_value=80):
                        renderer.render_result(result)

    panel = mock_console.print.call_args.args[0]
    assert isinstance(panel.renderable, FakeGroup)
    assert isinstance(panel.renderable.renderables[0], FakeMarkdown)
    assert panel.renderable.renderables[0].markup == "## 답변\n- 항목"
    assert "민원처리법.pdf" in panel.renderable.renderables[1].plain


def test_render_result_plain_fallback_keeps_raw_markdown_text(capsys):
    """plain fallback에서는 마크다운 원문을 그대로 출력한다."""
    result = {"text": "**굵게**\n- 목록\n`코드`"}

    with patch("src.cli.renderer._RICH_AVAILABLE", False):
        render_result(result)

    captured = capsys.readouterr()
    assert "**굵게**" in captured.out
    assert "- 목록" in captured.out
    assert "`코드`" in captured.out


def test_render_result_rich_panel_scales_at_120_columns():
    """120열 터미널에서는 Rich Panel width가 현재 폭에 맞춰 확장된다."""
    from src.cli import renderer

    mock_console = MagicMock()
    with patch.object(renderer, "_RICH_AVAILABLE", True):
        with patch.object(renderer, "_console", mock_console):
            with patch.object(renderer, "get_terminal_columns", return_value=120):
                renderer.render_result({"text": "넓은 터미널 응답"})

    panel = mock_console.print.call_args.args[0]
    assert panel.width == 118


def test_render_result_rich_panel_scales_at_40_columns():
    """40열 터미널에서도 Rich Panel이 현재 폭 안에서 렌더링된다."""
    from src.cli import renderer

    mock_console = MagicMock()
    with patch.object(renderer, "_RICH_AVAILABLE", True):
        with patch.object(renderer, "_console", mock_console):
            with patch.object(renderer, "get_terminal_columns", return_value=40):
                renderer.render_result({"text": "40열 응답"})

    panel = mock_console.print.call_args.args[0]
    assert panel.width == 38


def test_render_result_falls_back_to_plain_on_narrow_terminal(capsys):
    """40열 미만에서는 경고 후 plain mode로 결과를 출력한다."""
    from src.cli import renderer

    mock_console = MagicMock()
    with patch.object(renderer, "_RICH_AVAILABLE", True):
        with patch.object(renderer, "_console", mock_console):
            with patch.object(renderer, "get_terminal_columns", return_value=30):
                renderer.render_result({"text": "좁은 터미널 응답"})

    captured = capsys.readouterr()
    assert "plain mode" in captured.out
    assert "좁은 터미널 응답" in captured.out
    mock_console.print.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: LLMPlannerAdapter._build_user_prompt() — [이전 답변] 주입
# ---------------------------------------------------------------------------


def test_llm_planner_injects_previous_response():
    """previous_assistant_response가 있으면 [이전 답변] 섹션을 포함한다."""
    from langchain_core.messages import HumanMessage as HM

    messages = [HM(content="이 답변의 근거를 붙여줘")]
    context = {
        "session_context": "세션 요약",
        "previous_assistant_response": "도로 파손 민원을 접수해드리겠습니다.",
    }

    prompt = LLMPlannerAdapter._build_user_prompt(messages, context)

    assert "[이전 답변]" in prompt
    assert "도로 파손 민원을 접수해드리겠습니다." in prompt
    assert "[사용자 요청]" in prompt
    assert "이 답변의 근거를 붙여줘" in prompt


def test_llm_planner_no_previous_response():
    """previous_assistant_response가 없으면 [이전 답변] 섹션을 포함하지 않는다."""
    from langchain_core.messages import HumanMessage as HM

    messages = [HM(content="민원 답변 작성해줘")]
    context = {"session_context": "세션 요약"}

    prompt = LLMPlannerAdapter._build_user_prompt(messages, context)

    assert "[이전 답변]" not in prompt
    assert "[사용자 요청]" in prompt
