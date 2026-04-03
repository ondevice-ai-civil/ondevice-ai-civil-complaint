import sys
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.inference.agent_loop import AgentTrace, ToolResult
from src.inference.session_context import SessionContext
from src.inference.tool_router import ExecutionPlan, ToolStep, ToolType

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

with patch("src.inference.vllm_stabilizer.apply_transformers_patch"):
    import src.inference.api_server as api_server

    app = api_server.app
    manager = api_server.manager
    trace_to_schema = api_server._trace_to_schema

client = TestClient(app)


def _fake_output(text: str, prompt_tokens: int = 12, completion_tokens: int = 4):
    output = MagicMock()
    output.outputs = [MagicMock()]
    output.outputs[0].text = text
    output.outputs[0].token_ids = list(range(completion_tokens))
    output.prompt_token_ids = list(range(prompt_tokens))
    return output


def _fake_trace(session_id: str) -> AgentTrace:
    trace = AgentTrace(
        request_id="trace-001",
        session_id=session_id,
        plan=ExecutionPlan(
            steps=[
                ToolStep(tool=ToolType.RAG_SEARCH),
                ToolStep(tool=ToolType.API_LOOKUP),
            ],
            reason="runtime contract test",
        ),
        tool_results=[
            ToolResult(
                tool=ToolType.RAG_SEARCH,
                success=True,
                latency_ms=12.5,
                data={"results": [{"title": "도로 보수 매뉴얼"}]},
            ),
            ToolResult(
                tool=ToolType.API_LOOKUP,
                success=True,
                latency_ms=18.0,
                data={"results": [{"title": "유사 민원"}]},
            ),
        ],
        total_latency_ms=30.5,
        final_text="도로 보수 접수를 진행하겠습니다.",
    )
    return trace


async def _stream_events(**_kwargs):
    yield {"type": "plan", "plan": ["rag_search"], "reason": "test"}
    yield {
        "type": "final",
        "text": "완료",
        "trace": {"request_id": "trace-stream"},
        "finished": True,
    }


class TestAgentApi:
    def setup_method(self):
        self.original_agent_loop = manager.agent_loop
        self.original_session_store = manager.session_store
        self.original_generate_civil_response = manager.generate_civil_response

        manager.session_store = MagicMock()
        manager.session_store.get_or_create.side_effect = lambda session_id=None: SessionContext(
            session_id=session_id or "session-auto"
        )
        manager.session_store.db_path = "/tmp/govon-test-sessions.sqlite3"

    def teardown_method(self):
        manager.agent_loop = self.original_agent_loop
        manager.session_store = self.original_session_store
        manager.generate_civil_response = self.original_generate_civil_response

    def test_health_reports_sqlite_session_store(self):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["session_store"]["driver"] == "sqlite"

    def test_generate_civil_response_returns_cases_and_search_results(self):
        manager.generate_civil_response = AsyncMock(
            return_value=(
                _fake_output("민원 답변 초안"),
                [{"complaint": "도로 파손", "answer": "보수 예정", "score": 0.9}],
                [
                    {
                        "doc_id": "case-001",
                        "source_type": "case",
                        "title": "도로 보수 사례",
                        "content": "유사 사례 본문",
                        "score": 0.88,
                    }
                ],
            )
        )

        response = client.post(
            "/v1/generate-civil-response",
            json={"prompt": "도로 보수 민원 답변 초안 작성", "complaint_id": "cmp-001"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["complaint_id"] == "cmp-001"
        assert body["retrieved_cases"][0]["complaint"] == "도로 파손"
        assert body["search_results"][0]["doc_id"] == "case-001"

    def test_agent_run_returns_trace_and_search_results(self):
        manager.agent_loop = MagicMock()
        manager.agent_loop.run = AsyncMock(return_value=_fake_trace("session-123"))

        response = client.post(
            "/v1/agent/run",
            json={"query": "도로 보수 요청", "session_id": "session-123"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "session-123"
        assert body["text"] == "도로 보수 접수를 진행하겠습니다."
        assert body["trace"]["plan"] == ["rag_search", "api_lookup"]
        assert body["search_results"][0]["title"] == "도로 보수 매뉴얼"

    def test_agent_run_rejects_stream_flag(self):
        manager.agent_loop = MagicMock()
        manager.agent_loop.run = AsyncMock(return_value=_fake_trace("session-123"))

        response = client.post("/v1/agent/run", json={"query": "스트림", "stream": True})

        assert response.status_code == 400
        assert "agent/stream" in response.json()["detail"]

    def test_agent_stream_returns_sse_payload(self):
        manager.agent_loop = MagicMock()
        manager.agent_loop.run_stream = _stream_events

        response = client.post("/v1/agent/stream", json={"query": "진행 상황 알려줘"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"type": "plan"' in response.text
        assert '"finished": true' in response.text

    def test_trace_to_schema_serializes_plan_and_tool_results(self):
        schema = trace_to_schema(_fake_trace("session-789"))

        assert schema.session_id == "session-789"
        assert schema.plan == ["rag_search", "api_lookup"]
        assert schema.tool_results[0].tool == "rag_search"
