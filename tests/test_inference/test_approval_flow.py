"""승인/거절/취소 복귀 흐름 테스트.

Issue #418: 승인, 거절, 인터럽트, resume 상황에서
shell과 LangGraph runtime이 같은 의미로 동작하는지 검증한다.

테스트 시나리오:
  1. approved=False → status=="rejected" 반환
  2. approved=True → status=="completed" 반환
  3. /cancel 엔드포인트 → status=="cancelled" 반환
  4. /run 응답에 session_id 포함
  5. /approve 응답에 session_id 포함
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# api_server import 전에 환경변수 고정 — 로컬 SQLite 경로 의존성 제거
_tmp_db = tempfile.mktemp(suffix=".sqlite3")
os.environ.setdefault("GOVON_SESSION_DB", _tmp_db)
os.environ.setdefault("SKIP_MODEL_LOAD", "true")


# ---------------------------------------------------------------------------
# Graph invoke / get_state 결과를 시뮬레이션하기 위한 헬퍼
# ---------------------------------------------------------------------------


def _make_graph_state_interrupted(session_id: str = "sess-1", request_id: str = "req-1"):
    """interrupt 대기 중인 graph_state mock을 생성한다."""
    interrupt_mock = MagicMock()
    interrupt_mock.value = {
        "type": "approval_request",
        "goal": "테스트 작업",
        "reason": "테스트 이유",
        "planned_tools": ["rag_search"],
        "prompt": "승인하시겠습니까?",
    }
    task_mock = MagicMock()
    task_mock.interrupts = [interrupt_mock]

    state = MagicMock()
    state.next = ("approval_wait",)
    state.tasks = [task_mock]
    state.values = {
        "session_id": session_id,
        "request_id": request_id,
    }
    return state


def _make_graph_state_done(
    session_id: str = "sess-1",
    request_id: str = "req-1",
    approval_status: str = "approved",
    final_text: str = "완료 텍스트",
):
    """완료된 graph_state mock을 생성한다."""
    state = MagicMock()
    state.next = ()
    state.tasks = []
    state.values = {
        "session_id": session_id,
        "request_id": request_id,
        "approval_status": approval_status,
        "final_text": final_text,
    }
    return state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_manager():
    """api_server.manager를 mock으로 교체하는 fixture."""
    mgr = MagicMock()
    mgr.graph = MagicMock()
    mgr.graph.ainvoke = AsyncMock()
    mgr.graph.aget_state = AsyncMock()
    mgr.session_store = MagicMock()

    session_mock = MagicMock()
    session_mock.session_id = "sess-1"
    session_mock.add_graph_run = MagicMock()
    mgr.session_store.get_or_create.return_value = session_mock

    return mgr


@pytest.fixture()
def patched_app(mock_manager):
    """FastAPI TestClient 대신 직접 함수를 호출하기 위한 fixture.

    manager를 mock으로 교체한다.
    """
    with patch("src.inference.api_server.manager", mock_manager):
        yield mock_manager


# ---------------------------------------------------------------------------
# 테스트 케이스
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_rejected_returns_rejected_status(patched_app):
    """approved=False 시 status=="rejected" 반환을 검증한다."""
    from src.inference.api_server import v2_agent_approve

    mock_graph = patched_app.graph

    # invoke 결과: 거절 후 완료
    mock_graph.ainvoke.return_value = {
        "session_id": "sess-1",
        "request_id": "req-1",
        "final_text": "",
        "tool_results": {},
        "approval_status": "rejected",
    }

    resp = await v2_agent_approve(thread_id="t-1", approved=False, _=None)

    assert resp["status"] == "rejected"
    assert resp["thread_id"] == "t-1"
    assert resp["approval_status"] == "rejected"


@pytest.mark.asyncio
async def test_approve_approved_returns_completed_status(patched_app):
    """approved=True 시 status=="completed" 반환을 검증한다."""
    from src.inference.api_server import v2_agent_approve

    mock_graph = patched_app.graph

    mock_graph.ainvoke.return_value = {
        "session_id": "sess-1",
        "request_id": "req-1",
        "final_text": "결과 텍스트",
        "tool_results": {"rag_search": {"success": True}},
        "approval_status": "approved",
    }

    resp = await v2_agent_approve(thread_id="t-1", approved=True, _=None)

    assert resp["status"] == "completed"
    assert resp["thread_id"] == "t-1"
    assert resp["text"] == "결과 텍스트"
    assert resp["approval_status"] == "approved"


@pytest.mark.asyncio
async def test_cancel_sets_interrupted_status(patched_app):
    """POST /v2/agent/cancel 호출 시 status=="cancelled" 반환을 검증한다."""
    from src.inference.api_server import v2_agent_cancel

    mock_graph = patched_app.graph

    # get_state: interrupt 대기 중
    mock_graph.aget_state.return_value = _make_graph_state_interrupted("sess-1", "req-1")

    # invoke(Command(resume=...)): cancel 후 완료
    mock_graph.ainvoke.return_value = {
        "session_id": "sess-1",
        "request_id": "req-1",
        "final_text": "",
        "approval_status": "rejected",
        "interrupt_reason": "user_cancel",
    }

    resp = await v2_agent_cancel(thread_id="t-1", _=None)

    assert resp["status"] == "cancelled"
    assert resp["thread_id"] == "t-1"
    assert resp["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_run_response_includes_session_id(patched_app):
    """POST /v2/agent/run 응답에 session_id가 포함되는지 검증한다."""
    from src.inference.api_server import v2_agent_run
    from src.inference.schemas import AgentRunRequest

    mock_graph = patched_app.graph

    # invoke: interrupt 대기 상태
    mock_graph.ainvoke.return_value = None
    mock_graph.aget_state.return_value = _make_graph_state_interrupted("sess-run", "req-run")

    request = AgentRunRequest(query="테스트 질의", session_id="sess-run")

    resp = await v2_agent_run(request=request, _=None)

    assert resp["session_id"] == "sess-run"
    assert resp["status"] == "awaiting_approval"
    assert "graph_run_id" in resp


@pytest.mark.asyncio
async def test_approve_response_includes_session_id(patched_app):
    """POST /v2/agent/approve 응답에 session_id가 포함되는지 검증한다."""
    from src.inference.api_server import v2_agent_approve

    mock_graph = patched_app.graph

    mock_graph.ainvoke.return_value = {
        "session_id": "sess-approve",
        "request_id": "req-approve",
        "final_text": "결과",
        "tool_results": {},
        "approval_status": "approved",
    }

    resp = await v2_agent_approve(thread_id="t-2", approved=True, _=None)

    assert resp["session_id"] == "sess-approve"
    assert resp["graph_run_id"] == "req-approve"


@pytest.mark.asyncio
async def test_run_error_returns_error_status(patched_app):
    """POST /v2/agent/run에서 예외 발생 시 status=="error" 반환을 검증한다."""
    from src.inference.api_server import v2_agent_run
    from src.inference.schemas import AgentRunRequest

    mock_graph = patched_app.graph
    mock_graph.ainvoke.side_effect = RuntimeError("테스트 오류")

    request = AgentRunRequest(query="오류 질의", session_id="sess-err")

    resp = await v2_agent_run(request=request, _=None)

    assert resp["status"] == "error"
    assert "테스트 오류" in resp["error"]
    assert resp["session_id"] == "sess-err"


@pytest.mark.asyncio
async def test_approve_error_returns_error_status(patched_app):
    """POST /v2/agent/approve에서 예외 발생 시 status=="error" 반환을 검증한다."""
    from src.inference.api_server import v2_agent_approve

    mock_graph = patched_app.graph
    mock_graph.ainvoke.side_effect = RuntimeError("approve 오류")

    # get_state도 실패하지 않도록 설정
    mock_graph.aget_state.return_value = _make_graph_state_done()

    resp = await v2_agent_approve(thread_id="t-err", approved=True, _=None)

    assert resp["status"] == "error"
    assert "approve 오류" in resp["error"]
