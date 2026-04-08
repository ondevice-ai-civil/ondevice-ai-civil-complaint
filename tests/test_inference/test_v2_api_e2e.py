"""
v2 HTTP 에이전트 API 엔드포인트 E2E 테스트.

FastAPI TestClient를 사용해 HTTP 레이어를 검증한다.
manager.graph를 스텁으로 교체하여 LangGraph 의존성을 격리한다.

테스트 환경:
- conftest.py가 vllm/sentence_transformers/retriever/database.py/konlpy/rank_bm25/faiss를
  미리 mock으로 등록하므로, 이 파일에서는 중복 mock 불필요.
- API_KEY 환경변수가 None이면 verify_api_key가 인증을 건너뛰므로
  기본적으로 인증 우회 상태에서 동작한다.
- 인증 검증 테스트에서만 API_KEY를 명시적으로 설정한다.
"""

import os
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# retriever mock (test_retriever.py와 충돌 방지를 위해 conftest가 아닌 개별 파일에서 처리)
sys.modules.setdefault("src.inference.retriever", MagicMock())

from fastapi.testclient import TestClient

from src.inference.api_server import app, manager

# ---------------------------------------------------------------------------
# 헬퍼: 그래프 스텁
# ---------------------------------------------------------------------------

API_HEADERS = {"X-API-Key": "test-key"}

# 기본 쿼리 페이로드
_BASE_QUERY = {"query": "테스트 민원입니다.", "stream": False}


class FakeGraphState:
    """LangGraph StateSnapshot 최소 구현체."""

    def __init__(self, values: dict, next_nodes: list | None = None, tasks: list | None = None):
        self.values = values
        self.next = next_nodes or []
        self.tasks = tasks or []


class FakeInterruptTask:
    """interrupt 상태를 가진 task 스텁."""

    def __init__(self, approval_payload: dict):
        self.interrupts = [FakeInterrupt(approval_payload)]


class FakeInterrupt:
    """interrupt.value를 담는 스텁."""

    def __init__(self, value: dict):
        self.value = value


def _make_interrupted_graph(approval_payload: dict | None = None) -> AsyncMock:
    """approval_wait 노드에서 interrupt된 그래프 스텁을 반환한다."""
    if approval_payload is None:
        approval_payload = {
            "tool_name": "search_cases",
            "args": {"query": "건축법"},
            "plan_summary": "건축법 판례 검색 필요",
        }

    fake_task = FakeInterruptTask(approval_payload)
    interrupted_state = FakeGraphState(
        values={"session_id": "sess-1", "request_id": "req-1"},
        next_nodes=["approval_wait"],
        tasks=[fake_task],
    )
    completed_state = FakeGraphState(
        values={"session_id": "sess-1", "request_id": "req-1"},
        next_nodes=[],
    )

    mock_graph = AsyncMock()
    # [IMPORTANT] side_effect 순서 의존성:
    # api_server.py의 v2_agent_run은 aget_state를 정확히 2회 호출한다.
    #   1차 호출: ainvoke 전, 기존 interrupt 여부를 확인 (→ completed_state 반환, 즉 "없음")
    #   2차 호출: ainvoke 후,  최종 그래프 상태를 확인 (→ interrupted_state 반환)
    # 이 순서는 api_server.py 구현 흐름에 강하게 결합되어 있으므로,
    # v2_agent_run 내부 aget_state 호출 횟수/순서가 바뀌면 이 스텁도 함께 수정해야 한다.
    mock_graph.aget_state = AsyncMock(
        side_effect=[
            completed_state,  # 1차: "기존 interrupt 확인" 호출 → 없음
            interrupted_state,  # 2차: ainvoke 이후 상태 확인 → interrupted
        ]
    )
    mock_graph.ainvoke = AsyncMock(return_value={})
    return mock_graph


def _make_completed_graph(
    final_text: str = "답변입니다.", evidence: list | None = None
) -> AsyncMock:
    """interrupt 없이 완료된 그래프 스텁을 반환한다."""
    if evidence is None:
        evidence = []

    completed_state = FakeGraphState(
        values={
            "session_id": "sess-done",
            "request_id": "req-done",
            "final_text": final_text,
            "evidence_items": evidence,
        },
        next_nodes=[],  # 완료 = next가 빈 리스트
    )

    mock_graph = AsyncMock()
    # aget_state 모든 호출에서 completed 반환
    mock_graph.aget_state = AsyncMock(return_value=completed_state)
    mock_graph.ainvoke = AsyncMock(return_value={})
    return mock_graph


def _make_approve_graph(approved: bool, final_text: str = "실행 완료.") -> AsyncMock:
    """approve/reject 이후 결과를 반환하는 그래프 스텁."""
    result_payload = {
        "session_id": "sess-approve",
        "request_id": "req-approve",
        "final_text": final_text,
        "evidence_items": [],
        "approval_status": "approved" if approved else "rejected",
    }
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=result_payload)
    return mock_graph


def _make_cancel_ready_graph() -> AsyncMock:
    """interrupt 대기 상태의 그래프 스텁 (cancel 테스트용)."""
    interrupted_state = FakeGraphState(
        values={"session_id": "sess-cancel", "request_id": "req-cancel"},
        next_nodes=["approval_wait"],
    )
    cancel_result = {
        "session_id": "sess-cancel",
        "request_id": "req-cancel-done",
    }
    mock_graph = AsyncMock()
    mock_graph.aget_state = AsyncMock(return_value=interrupted_state)
    mock_graph.ainvoke = AsyncMock(return_value=cancel_result)
    return mock_graph


# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_graph():
    """manager.graph를 테스트 후 복원."""
    original = manager.graph
    yield
    manager.graph = original


@pytest.fixture
def client():
    """FastAPI TestClient를 생성하고 컨텍스트 종료 시 정리한다."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# TestV2RunEndpoint
# ---------------------------------------------------------------------------


class TestV2RunEndpoint:
    """POST /v2/agent/run 엔드포인트 동작을 검증한다."""

    def test_run_returns_awaiting_approval(self, client):
        """approval_wait에서 interrupt된 경우 status=awaiting_approval을 반환해야 한다."""
        manager.graph = _make_interrupted_graph(
            approval_payload={"tool_name": "search_cases", "plan_summary": "검색 승인 필요"}
        )
        resp = client.post("/v2/agent/run", json=_BASE_QUERY)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "awaiting_approval"
        assert "thread_id" in body
        assert "session_id" in body
        assert "approval_request" in body

    def test_run_returns_completed(self, client):
        """interrupt 없이 완료된 경우 status=completed와 text를 반환해야 한다."""
        manager.graph = _make_completed_graph(final_text="민원 처리 완료되었습니다.")
        resp = client.post("/v2/agent/run", json=_BASE_QUERY)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert "text" in body
        assert body["text"] == "민원 처리 완료되었습니다."

    def test_run_includes_thread_id_and_session_id(self, client):
        """응답에 thread_id와 session_id가 반드시 포함되어야 한다."""
        manager.graph = _make_completed_graph()
        resp = client.post("/v2/agent/run", json={**_BASE_QUERY, "session_id": "my-session"})
        assert resp.status_code == 200
        body = resp.json()
        assert "thread_id" in body
        assert "session_id" in body
        # session_id 파라미터를 전달하면 그 값이 thread_id로 사용된다
        assert body["thread_id"] == "my-session"
        assert body["session_id"] == "my-session"

    def test_run_no_graph_returns_503(self, client):
        """manager.graph가 None이면 503을 반환해야 한다."""
        manager.graph = None
        resp = client.post("/v2/agent/run", json=_BASE_QUERY)
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestV2ApproveEndpoint
# ---------------------------------------------------------------------------


class TestV2ApproveEndpoint:
    """POST /v2/agent/approve 엔드포인트 동작을 검증한다."""

    def test_approve_true_returns_completed(self, client):
        """approved=True이면 status=completed를 반환해야 한다."""
        manager.graph = _make_approve_graph(approved=True, final_text="도구 실행 완료.")
        resp = client.post(
            "/v2/agent/approve",
            params={"thread_id": "thread-abc", "approved": "true"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["thread_id"] == "thread-abc"

    def test_approve_false_returns_rejected(self, client):
        """approved=False이면 status=rejected를 반환해야 한다."""
        manager.graph = _make_approve_graph(approved=False)
        resp = client.post(
            "/v2/agent/approve",
            params={"thread_id": "thread-xyz", "approved": "false"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "rejected"
        assert body["thread_id"] == "thread-xyz"

    def test_approve_no_graph_returns_503(self, client):
        """manager.graph가 None이면 503을 반환해야 한다."""
        manager.graph = None
        resp = client.post(
            "/v2/agent/approve",
            params={"thread_id": "thread-none", "approved": "true"},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestV2CancelEndpoint
# ---------------------------------------------------------------------------


class TestV2CancelEndpoint:
    """POST /v2/agent/cancel 엔드포인트 동작을 검증한다."""

    def test_cancel_returns_cancelled(self, client):
        """interrupt 대기 중인 thread를 취소하면 status=cancelled를 반환해야 한다."""
        manager.graph = _make_cancel_ready_graph()
        resp = client.post(
            "/v2/agent/cancel",
            params={"thread_id": "thread-to-cancel"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "cancelled"
        assert body["thread_id"] == "thread-to-cancel"

    def test_cancel_no_graph_returns_503(self, client):
        """manager.graph가 None이면 503을 반환해야 한다."""
        manager.graph = None
        resp = client.post(
            "/v2/agent/cancel",
            params={"thread_id": "thread-no-graph"},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestV2DiverseQueries
# ---------------------------------------------------------------------------


class TestV2DiverseQueries:
    """다양한 도메인 쿼리에 대한 run → approve 플로우를 검증한다."""

    def test_legal_query_run_and_approve(self, client):
        """건축법 관련 법적 쿼리의 run → approve 전체 플로우를 검증한다."""
        legal_query = "건축법 위반 민원에 대한 회신문 작성해줘"
        approval_payload = {
            "tool_name": "search_law",
            "plan_summary": "건축법 위반 관련 법령 검색 필요",
            "args": {"query": "건축법 위반"},
        }
        run_graph = _make_interrupted_graph(approval_payload=approval_payload)
        approve_graph = _make_approve_graph(
            approved=True, final_text="건축법 위반 회신문 작성 완료."
        )

        # 1단계: run → interrupt
        manager.graph = run_graph
        run_resp = client.post("/v2/agent/run", json={"query": legal_query, "stream": False})
        assert run_resp.status_code == 200
        run_body = run_resp.json()
        assert run_body["status"] == "awaiting_approval"
        thread_id = run_body["thread_id"]
        assert thread_id

        # approval_request에 tool 정보가 포함되어야 한다
        approval_req = run_body.get("approval_request")
        assert approval_req is not None
        assert approval_req.get("tool_name") == "search_law"

        # 2단계: approve → completed
        manager.graph = approve_graph
        approve_resp = client.post(
            "/v2/agent/approve",
            params={"thread_id": thread_id, "approved": "true"},
        )
        assert approve_resp.status_code == 200
        approve_body = approve_resp.json()
        assert approve_body["status"] == "completed"
        assert "건축법" in approve_body["text"]

    def test_statistics_query_completes_without_interrupt(self, client):
        """통계 조회 쿼리는 interrupt 없이 바로 완료되어야 한다."""
        stats_query = "이번 달 민원 현황 보여줘"
        manager.graph = _make_completed_graph(
            final_text="이번 달 민원 건수: 총 342건 (처리완료 280건, 처리중 62건)",
            evidence=[{"source": "statistics_db", "content": "민원 통계 데이터"}],
        )
        resp = client.post("/v2/agent/run", json={"query": stats_query, "stream": False})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert "민원" in body["text"]
        assert isinstance(body["evidence_items"], list)
        assert len(body["evidence_items"]) > 0

    def test_issue_detection_query(self, client):
        """급증 민원 이슈 감지 쿼리의 응답 구조를 검증한다."""
        issue_query = "오늘 급증 민원 이슈 분석해줘"
        approval_payload = {
            "tool_name": "detect_issue",
            "plan_summary": "오늘 민원 패턴 이상 감지 분석 필요",
            "args": {"date": "today"},
        }
        manager.graph = _make_interrupted_graph(approval_payload=approval_payload)
        resp = client.post("/v2/agent/run", json={"query": issue_query, "stream": False})
        assert resp.status_code == 200
        body = resp.json()
        # interrupt 또는 completed 모두 허용하되 필수 필드 검증
        assert body["status"] in ("awaiting_approval", "completed")
        assert "thread_id" in body
        assert "session_id" in body
        assert "graph_run_id" in body

        # interrupt인 경우 approval_request가 존재해야 한다
        if body["status"] == "awaiting_approval":
            assert body.get("approval_request") is not None
            assert body["approval_request"].get("tool_name") == "detect_issue"
