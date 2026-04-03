import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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

client = TestClient(app)


@pytest.fixture
def mock_manager():
    with patch("src.inference.api_server.manager") as mock:
        mock.agent_manager.list_agents.return_value = [
            "retriever",
            "generator_civil_response",
        ]
        mock.index_manager = MagicMock()
        mock.hybrid_engine.search = AsyncMock(
            return_value=(
                [
                    {
                        "doc_id": "test-1",
                        "doc_type": "case",
                        "title": "테스트 민원",
                        "score": 0.95,
                        "extras": {"complaint_text": "도로 파손", "answer_text": "복구 예정"},
                    }
                ],
                "hybrid",
            )
        )
        yield mock


def test_health_endpoint(mock_manager):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "generator_civil_response" in response.json()["agents_loaded"]


def test_search_logic_mock(mock_manager):
    payload = {
        "query": "도로가 파손되었어요",
        "doc_type": "case",
        "search_mode": "hybrid",
        "top_k": 1,
    }

    response = client.post("/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["doc_id"] == "test-1"
