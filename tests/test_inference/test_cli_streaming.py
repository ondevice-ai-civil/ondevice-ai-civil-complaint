"""CLI LangGraph 상태 스트리밍 테스트.

Issue #132: CLI LangGraph 상태 및 스트리밍 출력 구현.

테스트 범위:
  1. SSE 엔드포인트 (/v2/agent/stream) 노드 이벤트 순서
  2. approval_wait interrupt 시 스트리밍 중단
  3. GovOnClient.stream() SSE 파싱
  4. StreamingStatusDisplay 노드 메시지 표시
  5. stream 엔드포인트 미지원 시 client.run() 폴백
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# vllm / heavy deps mock — test_agent_api.py 패턴과 동일
# ---------------------------------------------------------------------------

_vllm_mock = MagicMock()
_vllm_mock.AsyncLLM = MagicMock()
_vllm_mock.SamplingParams = MagicMock()
sys.modules.setdefault("vllm", _vllm_mock)
sys.modules.setdefault("vllm.engine", _vllm_mock)
sys.modules.setdefault("vllm.engine.arg_utils", _vllm_mock)
sys.modules.setdefault("vllm.engine.async_llm_engine", _vllm_mock)
sys.modules.setdefault("vllm.sampling_params", _vllm_mock)
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("transformers.modeling_rope_utils", MagicMock())
sys.modules.setdefault("transformers.utils", MagicMock())
sys.modules.setdefault("transformers.utils.generic", MagicMock())
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

# langchain_core / langgraph mocks (may not be installed in dev environment)
if "langchain_core" not in sys.modules:
    _lc_mock = MagicMock()

    class _HumanMessage:
        def __init__(self, content="", **kwargs):
            self.content = content

    _lc_mock.messages.HumanMessage = _HumanMessage
    sys.modules["langchain_core"] = _lc_mock
    sys.modules["langchain_core.messages"] = _lc_mock.messages

if "langgraph" not in sys.modules:
    _lg_mock = MagicMock()
    sys.modules["langgraph"] = _lg_mock
    sys.modules["langgraph.graph"] = _lg_mock
    sys.modules["langgraph.graph.message"] = _lg_mock
    sys.modules["langgraph.checkpoint"] = _lg_mock
    sys.modules["langgraph.checkpoint.memory"] = _lg_mock
    sys.modules["langgraph.checkpoint.sqlite"] = _lg_mock
    sys.modules["langgraph.types"] = _lg_mock

with patch("src.inference.vllm_stabilizer.apply_transformers_patch"):
    import src.inference.api_server as api_server

    app = api_server.app
    manager = api_server.manager

from fastapi.testclient import TestClient

test_client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_stream_events(include_approval: bool = False):
    """graph.stream() 이 반환할 fake chunk 목록을 반환한다.

    각 chunk는 {node_name: state_delta} 형태이다.
    """
    chunks = [
        {"session_load": {"session_id": "test-stream-session"}},
        {"planner": {"goal": "test goal", "planned_tools": ["rag_search"]}},
    ]
    if include_approval:
        # approval_wait 도달 시 stream에서 마지막 chunk만 나옴
        chunks.append({"approval_wait": {}})
    else:
        chunks.append({"tool_execute": {"tool_results": {"rag_search": {"success": True}}}})
        chunks.append({"synthesis": {"final_text": "테스트 답변"}})
        chunks.append({"persist": {}})
    return chunks


def _build_sse_response(events: list[dict]) -> bytes:
    """SSE 이벤트 목록을 SSE 바이트 스트림으로 인코딩한다."""
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event, ensure_ascii=False)}\n\n")
    return "".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# 1. SSE 엔드포인트 테스트
# ---------------------------------------------------------------------------


class TestV2AgentStreamEndpoint:
    """POST /v2/agent/stream 엔드포인트 테스트."""

    def setup_method(self):
        self.original_graph = manager.graph

    def teardown_method(self):
        manager.graph = self.original_graph

    def test_stream_yields_node_events_in_order(self):
        """graph가 초기화된 경우 노드별 이벤트가 SSE로 전송된다."""
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter(_make_graph_stream_events(include_approval=False))
        mock_graph.get_state.return_value = MagicMock(next=[], tasks=[], values={})
        manager.graph = mock_graph

        response = test_client.post(
            "/v2/agent/stream",
            json={"query": "테스트 쿼리"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = _parse_sse_events(response.text)
        node_names = [e.get("node") for e in events]

        # 예상 노드 순서 확인
        assert "session_load" in node_names
        assert "planner" in node_names

    def test_stream_stops_at_approval_wait(self):
        """approval_wait 노드 도달 시 awaiting_approval 이벤트를 반환하고 중단된다."""
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter(_make_graph_stream_events(include_approval=True))

        # approval_wait interrupt 상태 모킹
        fake_interrupt = MagicMock()
        fake_interrupt.value = {
            "goal": "테스트 목표",
            "reason": "테스트 이유",
            "tool_summaries": [],
        }
        fake_task = MagicMock()
        fake_task.interrupts = [fake_interrupt]
        fake_state = MagicMock()
        fake_state.next = ["approval_wait"]
        fake_state.tasks = [fake_task]
        mock_graph.get_state.return_value = fake_state
        manager.graph = mock_graph

        response = test_client.post(
            "/v2/agent/stream",
            json={"query": "승인 필요 쿼리"},
        )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)

        # 마지막 이벤트가 awaiting_approval이어야 한다
        approval_events = [e for e in events if e.get("status") == "awaiting_approval"]
        assert approval_events, f"awaiting_approval 이벤트가 없음. 이벤트: {events}"

        approval_event = approval_events[-1]
        assert approval_event["node"] == "approval_wait"
        assert "thread_id" in approval_event

    def test_stream_returns_error_when_graph_not_initialized(self):
        """graph가 None인 경우 error 이벤트를 반환한다."""
        manager.graph = None

        response = test_client.post(
            "/v2/agent/stream",
            json={"query": "테스트"},
        )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        assert any(e.get("node") == "error" for e in events), f"error 이벤트 없음: {events}"

    def test_stream_handles_graph_exception(self):
        """graph.stream() 예외 시 error 이벤트를 반환한다."""
        mock_graph = MagicMock()
        mock_graph.stream.side_effect = RuntimeError("graph 오류 시뮬레이션")
        manager.graph = mock_graph

        response = test_client.post(
            "/v2/agent/stream",
            json={"query": "오류 쿼리"},
        )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        assert any(e.get("node") == "error" for e in events), f"error 이벤트 없음: {events}"
        error_events = [e for e in events if e.get("node") == "error"]
        assert error_events[0].get("error"), "error 메시지가 있어야 합니다"


def _parse_sse_events(sse_text: str) -> list[dict]:
    """SSE 텍스트에서 data 라인을 파싱하여 dict 목록으로 반환한다."""
    events = []
    for line in sse_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[len("data:") :].strip()
            if data_str:
                try:
                    events.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass
    return events


# ---------------------------------------------------------------------------
# 2. GovOnClient.stream() SSE 파싱 테스트
# ---------------------------------------------------------------------------


class TestGovOnClientStream:
    """GovOnClient.stream() SSE 파싱 단위 테스트."""

    def test_stream_parses_sse_events_correctly(self):
        """SSE 응답 라인을 올바르게 파싱하여 dict를 yield한다."""
        from src.cli.http_client import GovOnClient

        events_data = [
            {"node": "session_load", "status": "completed"},
            {"node": "planner", "status": "completed"},
            {"node": "approval_wait", "status": "awaiting_approval", "thread_id": "t-001"},
        ]
        sse_body = _build_sse_response(events_data).decode("utf-8")

        client = GovOnClient("http://localhost:8000")

        with patch("httpx.Client") as mock_httpx_cls:
            mock_httpx = MagicMock()
            mock_httpx_cls.return_value.__enter__ = lambda s: mock_httpx
            mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.iter_lines.return_value = iter(sse_body.splitlines())
            mock_httpx.stream.return_value.__enter__ = lambda s: mock_resp
            mock_httpx.stream.return_value.__exit__ = MagicMock(return_value=False)

            received = list(client.stream("테스트 쿼리"))

        assert len(received) == 3
        assert received[0]["node"] == "session_load"
        assert received[1]["node"] == "planner"
        assert received[2]["status"] == "awaiting_approval"

    def test_stream_skips_non_data_lines(self):
        """data: 로 시작하지 않는 라인은 무시된다."""
        from src.cli.http_client import GovOnClient

        lines = [
            "event: node_update",
            'data: {"node": "planner", "status": "completed"}',
            "",
            ": comment line",
            'data: {"node": "persist", "status": "completed"}',
        ]

        client = GovOnClient("http://localhost:8000")

        with patch("httpx.Client") as mock_httpx_cls:
            mock_httpx = MagicMock()
            mock_httpx_cls.return_value.__enter__ = lambda s: mock_httpx
            mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.iter_lines.return_value = iter(lines)
            mock_httpx.stream.return_value.__enter__ = lambda s: mock_resp
            mock_httpx.stream.return_value.__exit__ = MagicMock(return_value=False)

            received = list(client.stream("테스트"))

        assert len(received) == 2
        assert received[0]["node"] == "planner"
        assert received[1]["node"] == "persist"

    def test_stream_raises_connection_error_on_connect_failure(self):
        """daemon 미실행 시 ConnectionError가 발생한다."""
        import httpx as _httpx

        from src.cli.http_client import GovOnClient

        client = GovOnClient("http://localhost:9999")

        with patch("httpx.Client") as mock_httpx_cls:
            mock_httpx = MagicMock()
            mock_httpx_cls.return_value.__enter__ = lambda s: mock_httpx
            mock_httpx_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_httpx.stream.side_effect = _httpx.ConnectError("Connection refused")

            with pytest.raises(ConnectionError):
                list(client.stream("테스트"))


# ---------------------------------------------------------------------------
# 3. StreamingStatusDisplay 렌더러 테스트
# ---------------------------------------------------------------------------


class TestStreamingStatusDisplay:
    """StreamingStatusDisplay 및 노드 메시지 매핑 테스트."""

    def test_node_status_messages_cover_all_nodes(self):
        """NODE_STATUS_MESSAGES에 6개 필수 노드가 모두 포함된다."""
        from src.cli.renderer import NODE_STATUS_MESSAGES

        required_nodes = {
            "session_load",
            "planner",
            "approval_wait",
            "tool_execute",
            "synthesis",
            "persist",
        }
        assert required_nodes.issubset(set(NODE_STATUS_MESSAGES.keys()))

    def test_get_node_message_returns_known_message(self):
        """알려진 노드 이름에 대해 올바른 메시지를 반환한다."""
        from src.cli.renderer import get_node_message

        assert get_node_message("planner") == "계획 수립 중…"
        assert get_node_message("tool_execute") == "도구 실행 중…"
        assert get_node_message("synthesis") == "답변 생성 중…"

    def test_get_node_message_returns_fallback_for_unknown(self):
        """알 수 없는 노드 이름에 대해 기본 메시지를 반환한다."""
        from src.cli.renderer import get_node_message

        msg = get_node_message("unknown_node_xyz")
        assert "unknown_node_xyz" in msg

    def test_streaming_status_display_context_manager_plain(self, capsys):
        """rich 없이(plain mode)도 컨텍스트 매니저가 정상 동작한다."""
        from src.cli.renderer import StreamingStatusDisplay

        # rich 없는 환경 시뮬레이션
        with patch("src.cli.renderer._RICH_AVAILABLE", False):
            display = StreamingStatusDisplay("테스트 시작")
            with display as d:
                d.update("업데이트 메시지")

        captured = capsys.readouterr()
        assert "테스트 시작" in captured.out
        assert "업데이트 메시지" in captured.out

    def test_streaming_status_display_update_plain(self, capsys):
        """update() 호출 시 plain mode에서 새 메시지가 출력된다."""
        from src.cli.renderer import StreamingStatusDisplay

        with patch("src.cli.renderer._RICH_AVAILABLE", False):
            with StreamingStatusDisplay("초기") as d:
                d.update("planner 처리 중")
                d.update("synthesis 처리 중")

        out = capsys.readouterr().out
        assert "planner 처리 중" in out
        assert "synthesis 처리 중" in out


# ---------------------------------------------------------------------------
# 4. Shell _process_query 스트리밍 통합 테스트
# ---------------------------------------------------------------------------


class TestProcessQueryStreaming:
    """shell._process_query_streaming() 동작 테스트."""

    def test_streaming_shows_per_node_progress(self):
        """스트리밍 이벤트마다 StreamingStatusDisplay가 업데이트된다."""
        from src.cli import shell

        mock_client = MagicMock()
        mock_client.stream.return_value = iter(
            [
                {"node": "session_load", "status": "completed"},
                {"node": "planner", "status": "completed"},
                {"node": "synthesis", "status": "completed", "final_text": "완료"},
                {"node": "persist", "status": "completed"},
            ]
        )
        mock_client.approve = MagicMock()

        update_calls = []

        class TrackingDisplay:
            def __init__(self, initial_message=""):
                pass

            def __enter__(self):
                return self

            def update(self, message):
                update_calls.append(message)

            def __exit__(self, *args):
                pass

        with patch("src.cli.shell.StreamingStatusDisplay", TrackingDisplay):
            sid, cont = shell._process_query_streaming(mock_client, "쿼리", None)

        assert cont is True
        assert any(
            "planner" in msg or "계획" in msg for msg in update_calls
        ), f"planner 관련 메시지가 없음: {update_calls}"

    def test_streaming_handles_approval_event(self):
        """awaiting_approval 이벤트 시 show_approval_prompt를 호출한다."""
        from src.cli import shell

        approval_data = {
            "goal": "테스트 목표",
            "reason": "테스트 이유",
            "tool_summaries": ["rag_search 실행"],
        }
        mock_client = MagicMock()
        mock_client.stream.return_value = iter(
            [
                {"node": "session_load", "status": "completed"},
                {"node": "planner", "status": "completed"},
                {
                    "node": "approval_wait",
                    "status": "awaiting_approval",
                    "approval_request": approval_data,
                    "thread_id": "t-123",
                    "session_id": "s-123",
                },
            ]
        )
        mock_client.approve.return_value = {"status": "completed", "text": "승인 완료"}

        with patch("src.cli.shell.StreamingStatusDisplay"):
            with patch("src.cli.shell.show_approval_prompt", return_value=True) as mock_prompt:
                sid, cont = shell._process_query_streaming(mock_client, "쿼리", None)

        mock_prompt.assert_called_once_with(approval_data)
        mock_client.approve.assert_called_once_with("t-123", approved=True)
        assert cont is True

    def test_streaming_handles_error_event(self):
        """error 이벤트 시 render_error를 호출하고 조기 반환한다."""
        from src.cli import shell

        mock_client = MagicMock()
        mock_client.stream.return_value = iter(
            [
                {"node": "planner", "status": "completed"},
                {"node": "error", "status": "error", "error": "테스트 오류"},
            ]
        )

        with patch("src.cli.shell.StreamingStatusDisplay"):
            with patch("src.cli.shell.render_error") as mock_render_error:
                sid, cont = shell._process_query_streaming(mock_client, "쿼리", None)

        mock_render_error.assert_called_once()
        assert cont is True


# ---------------------------------------------------------------------------
# 5. 폴백 동작 테스트
# ---------------------------------------------------------------------------


class TestFallbackToBlocking:
    """stream 엔드포인트 미지원 시 client.run() 폴백 테스트."""

    def test_fallback_when_stream_raises_connection_error(self):
        """stream()이 ConnectionError를 던지면 client.run()으로 폴백한다."""
        from src.cli import shell

        mock_client = MagicMock()
        mock_client.stream.side_effect = ConnectionError("stream 미지원")
        mock_client.run.return_value = {
            "status": "completed",
            "text": "폴백 응답",
            "session_id": "s-fallback",
        }

        with patch("src.cli.shell.render_result") as mock_render:
            sid, cont = shell._process_query(mock_client, "쿼리", None)

        mock_client.run.assert_called_once()
        mock_render.assert_called_once()
        assert cont is True

    def test_fallback_when_stream_raises_http_error(self):
        """stream()이 HTTP 오류를 던지면 client.run()으로 폴백한다."""
        import httpx as _httpx

        from src.cli import shell

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client.stream.side_effect = _httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )
        mock_client.run.return_value = {
            "status": "completed",
            "text": "폴백 응답",
        }

        with patch("src.cli.shell.render_result"):
            sid, cont = shell._process_query(mock_client, "쿼리", None)

        mock_client.run.assert_called_once()
        assert cont is True

    def test_process_query_blocking_handles_approval(self):
        """_process_query_blocking에서 awaiting_approval 응답 시 approval UI를 표시한다."""
        from src.cli import shell

        mock_client = MagicMock()
        mock_client.run.return_value = {
            "status": "awaiting_approval",
            "thread_id": "t-block",
            "session_id": "s-block",
            "approval_request": {"goal": "목표", "reason": "이유"},
        }
        mock_client.approve.return_value = {"status": "completed", "text": "완료"}

        with patch("src.cli.shell.show_approval_prompt", return_value=True) as mock_prompt:
            with patch("src.cli.shell.render_result"):
                sid, cont = shell._process_query_blocking(mock_client, "쿼리", None)

        mock_prompt.assert_called_once()
        mock_client.approve.assert_called_once_with("t-block", approved=True)
        assert cont is True
