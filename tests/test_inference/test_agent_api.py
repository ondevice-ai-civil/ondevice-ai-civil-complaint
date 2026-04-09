import sys
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.inference.session_context import SessionContext

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

import src.inference.api_server as api_server

app = api_server.app
manager = api_server.manager

client = TestClient(app)


class TestAgentApi:
    def setup_method(self):
        self.original_session_store = manager.session_store

        manager.session_store = MagicMock()
        manager.session_store.get_or_create.side_effect = lambda session_id=None: SessionContext(
            session_id=session_id or "session-auto"
        )
        manager.session_store.db_path = "/tmp/govon-test-sessions.sqlite3"

    def teardown_method(self):
        manager.session_store = self.original_session_store

    def test_health_reports_sqlite_session_store(self):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["session_store"]["driver"] == "sqlite"
