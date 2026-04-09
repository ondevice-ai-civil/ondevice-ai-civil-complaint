"""
v3 HTTP 에이전트 API 엔드포인트 E2E 테스트.

FastAPI TestClient를 사용해 HTTP 레이어를 검증한다.
manager.graph_v3를 스텁으로 교체하여 LangGraph 의존성을 격리한다.

테스트 환경:
- conftest.py가 vllm/sentence_transformers/database.py/faiss를
  미리 mock으로 등록하므로, 이 파일에서는 중복 mock 불필요.
- API_KEY 환경변수가 None이면 verify_api_key가 인증을 건너뛰므로
  기본적으로 인증 우회 상태에서 동작한다.

v2와의 차이점:
- approval flow 없음 (awaiting_approval, approve/cancel 엔드포인트 없음)
- 항상 status=completed 또는 error 반환
- 응답에 metadata 딕셔너리 포함 (total_iterations, total_tool_calls, total_latency_ms, node_latencies)
- manager.graph_v3 사용 (manager.graph 아님)
"""

import json as json_mod
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.inference.api_server import app, manager

# ---------------------------------------------------------------------------
# 헬퍼: FakeGraphState
# ---------------------------------------------------------------------------


class FakeGraphState:
    """LangGraph StateSnapshot 최소 구현체."""

    def __init__(self, values: dict, next_nodes: list | None = None, tasks: list | None = None):
        self.values = values
        self.next = next_nodes or []
        self.tasks = tasks or []


# ---------------------------------------------------------------------------
# 헬퍼: v3 그래프 스텁
# ---------------------------------------------------------------------------


def _make_v3_completed_graph(
    final_text="답변입니다.",
    evidence=None,
    iteration_count=0,
    tool_call_history=None,
    node_latencies=None,
):
    """v3 graph 완료 상태 스텁.

    graph_v3.ainvoke는 state 딕셔너리를 반환한다.
    """
    result = {
        "session_id": "sess-v3",
        "request_id": "req-v3",
        "final_text": final_text,
        "evidence_items": evidence or [],
        "iteration_count": iteration_count,
        "tool_call_history": tool_call_history or [],
        "node_latencies": node_latencies or {"session_load": 1.0, "agent": 50.0, "persist": 2.0},
    }
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=result)
    # SSE용: aget_state는 values를 담은 state를 반환
    state = FakeGraphState(values=result)
    mock_graph.aget_state = AsyncMock(return_value=state)
    return mock_graph


def _make_v3_error_graph():
    """graph_v3.ainvoke가 예외를 발생시키는 스텁."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 호출 실패"))
    return mock_graph


def _make_v3_stream_graph(events_list, final_state_values):
    """v3 SSE 스트리밍용 graph 스텁.

    events_list: LangGraph astream_events 형식의 이벤트 딕셔너리 목록.
    astream_events는 AsyncMock이 아닌 async 제너레이터 함수여야 한다.
    """

    async def _fake_astream_events(state, config, version=None):
        for evt in events_list:
            yield evt

    mock_graph = AsyncMock()
    mock_graph.astream_events = _fake_astream_events
    state = FakeGraphState(values=final_state_values)
    mock_graph.aget_state = AsyncMock(return_value=state)
    return mock_graph


# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_graph():
    """manager.graph_v3를 테스트 후 복원."""
    original = manager.graph_v3
    yield
    manager.graph_v3 = original


@pytest.fixture
def client():
    """FastAPI TestClient를 생성하고 컨텍스트 종료 시 정리한다."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# TestV3RunEndpoint
# ---------------------------------------------------------------------------


class TestV3RunEndpoint:
    """POST /v3/agent/run 엔드포인트 검증."""

    def test_run_returns_completed_with_metadata(self, client):
        """정상 완료 시 status=completed + metadata를 반환해야 한다."""
        manager.graph_v3 = _make_v3_completed_graph(
            final_text="v3 답변입니다.",
            iteration_count=1,
            tool_call_history=[{"iteration": 1, "tool": "api_lookup", "success": True}],
            node_latencies={"session_load": 1.0, "agent": 50.0, "tools": 30.0, "persist": 2.0},
        )
        resp = client.post("/v3/agent/run", json={"query": "테스트"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["text"] == "v3 답변입니다."
        assert "metadata" in body
        assert body["metadata"]["total_iterations"] == 1
        assert body["metadata"]["total_tool_calls"] == 1
        assert body["metadata"]["total_latency_ms"] > 0
        assert "node_latencies" in body["metadata"]

    def test_run_includes_iteration_and_tool_history(self, client):
        """iteration_count와 tool_call_history가 metadata에 정확히 반영된다."""
        history = [
            {"iteration": 1, "tool": "api_lookup", "success": True},
            {"iteration": 2, "tool": "stats_lookup", "success": True},
        ]
        manager.graph_v3 = _make_v3_completed_graph(
            iteration_count=2,
            tool_call_history=history,
        )
        resp = client.post("/v3/agent/run", json={"query": "테스트"})
        body = resp.json()
        assert body["metadata"]["total_iterations"] == 2
        assert body["metadata"]["total_tool_calls"] == 2

    def test_run_no_graph_returns_503(self, client):
        """graph_v3가 None이면 503을 반환해야 한다."""
        manager.graph_v3 = None
        resp = client.post("/v3/agent/run", json={"query": "테스트"})
        assert resp.status_code == 503

    def test_run_session_id_passthrough(self, client):
        """요청의 session_id가 응답에 그대로 전달된다."""
        manager.graph_v3 = _make_v3_completed_graph()
        resp = client.post("/v3/agent/run", json={"query": "테스트", "session_id": "my-v3-sess"})
        body = resp.json()
        assert body["session_id"] == "my-v3-sess"

    def test_run_max_iterations_validation(self, client):
        """max_iterations 범위 검증: 0→422, 21→422, 5→200."""
        manager.graph_v3 = _make_v3_completed_graph()
        # 유효하지 않음: 0
        resp0 = client.post("/v3/agent/run", json={"query": "테스트", "max_iterations": 0})
        assert resp0.status_code == 422
        # 유효하지 않음: 21
        resp21 = client.post("/v3/agent/run", json={"query": "테스트", "max_iterations": 21})
        assert resp21.status_code == 422
        # 유효함: 5
        resp5 = client.post("/v3/agent/run", json={"query": "테스트", "max_iterations": 5})
        assert resp5.status_code == 200

    def test_run_internal_error_returns_500(self, client):
        """graph.ainvoke 예외 시 500 + error 메시지를 반환해야 한다."""
        manager.graph_v3 = _make_v3_error_graph()
        resp = client.post("/v3/agent/run", json={"query": "테스트"})
        assert resp.status_code == 500
        body = resp.json()
        assert body["status"] == "error"
        assert "error" in body

    def test_run_response_includes_thread_and_graph_run_id(self, client):
        """응답에 thread_id와 graph_run_id가 포함되어야 한다."""
        manager.graph_v3 = _make_v3_completed_graph()
        resp = client.post("/v3/agent/run", json={"query": "테스트"})
        body = resp.json()
        assert "thread_id" in body
        assert "graph_run_id" in body
        # v3는 매 요청마다 새 UUID를 사용하므로 값이 존재해야 한다
        assert body["thread_id"]
        assert body["graph_run_id"]

    def test_run_evidence_items_included(self, client):
        """evidence_items가 응답에 포함되어야 한다."""
        evidence = [{"source": "api_lookup", "content": "관련 민원 데이터"}]
        manager.graph_v3 = _make_v3_completed_graph(evidence=evidence)
        resp = client.post("/v3/agent/run", json={"query": "민원 조회"})
        body = resp.json()
        assert "evidence_items" in body
        assert len(body["evidence_items"]) == 1
        assert body["evidence_items"][0]["source"] == "api_lookup"


# ---------------------------------------------------------------------------
# TestV3StreamEndpoint
# ---------------------------------------------------------------------------


class TestV3StreamEndpoint:
    """POST /v3/agent/stream 엔드포인트 검증."""

    def test_stream_no_graph_returns_503(self, client):
        """graph_v3가 None이면 503을 반환해야 한다."""
        manager.graph_v3 = None
        resp = client.post("/v3/agent/stream", json={"query": "테스트"})
        assert resp.status_code == 503

    def test_stream_returns_sse_events(self, client):
        """SSE 스트리밍이 data: 라인을 반환해야 한다."""
        # 도구 미사용 최소 이벤트 흐름
        events = [
            {"event": "on_chat_model_start", "name": "agent", "data": {}},
            {
                "event": "on_chat_model_stream",
                "name": "agent",
                "data": {"chunk": type("C", (), {"content": "답변"})()},
            },
            {
                "event": "on_chat_model_end",
                "name": "agent",
                "data": {"output": type("M", (), {"content": "답변", "tool_calls": None})()},
            },
        ]
        final_state = {
            "final_text": "답변",
            "evidence_items": [],
            "iteration_count": 0,
            "tool_call_history": [],
            "node_latencies": {"session_load": 1.0, "agent": 50.0, "persist": 2.0},
        }
        manager.graph_v3 = _make_v3_stream_graph(events, final_state)
        resp = client.post("/v3/agent/stream", json={"query": "테스트"})
        assert resp.status_code == 200
        # SSE data: 라인 파싱
        data_lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        # 최소한 run_complete 이벤트 1개는 있어야 한다
        assert len(data_lines) >= 1

    def test_stream_run_complete_has_metadata(self, client):
        """run_complete 이벤트에 metadata가 포함되어야 한다."""
        events = [
            {
                "event": "on_chat_model_end",
                "name": "agent",
                "data": {"output": type("M", (), {"content": "답변", "tool_calls": None})()},
            },
        ]
        final_state = {
            "final_text": "답변",
            "evidence_items": [],
            "iteration_count": 2,
            "tool_call_history": [{"t": 1}, {"t": 2}],
            "node_latencies": {"session_load": 1.0, "agent": 50.0},
        }
        manager.graph_v3 = _make_v3_stream_graph(events, final_state)
        resp = client.post("/v3/agent/stream", json={"query": "테스트"})

        data_lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        # run_complete 이벤트 탐색
        run_complete = None
        for line in data_lines:
            payload = line[len("data:") :].strip()
            try:
                evt = json_mod.loads(payload)
                if evt.get("type") == "run_complete":
                    run_complete = evt
                    break
            except json_mod.JSONDecodeError:
                continue

        assert run_complete is not None, f"run_complete 이벤트 없음: {data_lines}"
        assert "metadata" in run_complete
        assert run_complete["metadata"]["total_iterations"] == 2
        assert run_complete["metadata"]["total_tool_calls"] == 2

    def test_stream_run_complete_includes_session_id(self, client):
        """run_complete 이벤트에 session_id가 포함되어야 한다."""
        events = []
        final_state = {
            "final_text": "완료",
            "evidence_items": [],
            "iteration_count": 0,
            "tool_call_history": [],
            "node_latencies": {},
        }
        manager.graph_v3 = _make_v3_stream_graph(events, final_state)
        resp = client.post(
            "/v3/agent/stream", json={"query": "테스트", "session_id": "stream-sess"}
        )

        data_lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        run_complete = None
        for line in data_lines:
            payload = line[len("data:") :].strip()
            try:
                evt = json_mod.loads(payload)
                if evt.get("type") == "run_complete":
                    run_complete = evt
                    break
            except json_mod.JSONDecodeError:
                continue

        assert run_complete is not None, f"run_complete 이벤트 없음: {data_lines}"
        assert run_complete.get("session_id") == "stream-sess"

    def test_stream_thinking_start_event(self, client):
        """on_chat_model_start 이벤트가 thinking_start SSE로 변환되어야 한다."""
        events = [
            {"event": "on_chat_model_start", "name": "agent", "data": {}},
            {
                "event": "on_chat_model_end",
                "name": "agent",
                "data": {"output": type("M", (), {"content": "답변", "tool_calls": None})()},
            },
        ]
        final_state = {
            "final_text": "답변",
            "evidence_items": [],
            "iteration_count": 0,
            "tool_call_history": [],
            "node_latencies": {},
        }
        manager.graph_v3 = _make_v3_stream_graph(events, final_state)
        resp = client.post("/v3/agent/stream", json={"query": "테스트"})

        data_lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        types_found = []
        for line in data_lines:
            payload = line[len("data:") :].strip()
            try:
                evt = json_mod.loads(payload)
                types_found.append(evt.get("type"))
            except json_mod.JSONDecodeError:
                continue

        assert "thinking_start" in types_found, f"thinking_start 이벤트 없음: {types_found}"


# ---------------------------------------------------------------------------
# TestV3DiverseQueries
# ---------------------------------------------------------------------------


class TestV3DiverseQueries:
    """다양한 v3 쿼리 시나리오 검증."""

    def test_simple_query_no_tools(self, client):
        """도구 미사용 쿼리 → iteration_count=0, total_tool_calls=0."""
        manager.graph_v3 = _make_v3_completed_graph(
            final_text="직접 답변입니다.",
            iteration_count=0,
            tool_call_history=[],
        )
        resp = client.post("/v3/agent/run", json={"query": "안녕하세요"})
        body = resp.json()
        assert body["status"] == "completed"
        assert body["metadata"]["total_iterations"] == 0
        assert body["metadata"]["total_tool_calls"] == 0

    def test_multi_iteration_query(self, client):
        """다중 iteration 쿼리 → metadata에 정확히 반영."""
        history = [
            {"iteration": 1, "tool": "api_lookup", "success": True},
            {"iteration": 2, "tool": "stats_lookup", "success": True},
            {"iteration": 3, "tool": "keyword_analyzer", "success": True},
        ]
        manager.graph_v3 = _make_v3_completed_graph(
            final_text="3회 반복 후 답변.",
            iteration_count=3,
            tool_call_history=history,
        )
        resp = client.post("/v3/agent/run", json={"query": "복합 분석"})
        body = resp.json()
        assert body["metadata"]["total_iterations"] == 3
        assert body["metadata"]["total_tool_calls"] == 3

    def test_v3_has_no_approval_flow(self, client):
        """v3는 approval flow가 없으므로 항상 completed를 반환해야 한다."""
        # 어떤 쿼리를 보내도 awaiting_approval이 반환되지 않아야 한다
        manager.graph_v3 = _make_v3_completed_graph(
            final_text="자율 실행 완료.",
            iteration_count=2,
            tool_call_history=[
                {"iteration": 1, "tool": "search_law", "success": True},
                {"iteration": 2, "tool": "api_lookup", "success": True},
            ],
        )
        resp = client.post("/v3/agent/run", json={"query": "건축법 위반 민원 회신문 작성"})
        body = resp.json()
        assert body["status"] == "completed"
        assert body["status"] != "awaiting_approval"

    def test_node_latencies_in_metadata(self, client):
        """node_latencies가 metadata에 딕셔너리로 포함되어야 한다."""
        node_latencies = {
            "session_load": 2.5,
            "agent": 120.3,
            "tools": 45.8,
            "persist": 3.1,
        }
        manager.graph_v3 = _make_v3_completed_graph(
            node_latencies=node_latencies,
        )
        resp = client.post("/v3/agent/run", json={"query": "레이턴시 확인"})
        body = resp.json()
        assert body["metadata"]["node_latencies"] == node_latencies
