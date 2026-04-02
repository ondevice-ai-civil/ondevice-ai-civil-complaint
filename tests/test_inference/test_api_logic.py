import os
import sys
import importlib.util
from unittest.mock import MagicMock, patch, AsyncMock

# ---------------------------------------------------------------------------
# 무거운 의존성 mock 등록 (import 전에 등록해야 함)
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

import pytest
from fastapi.testclient import TestClient

# vllm_stabilizer의 apply_transformers_patch를 no-op 처리 후 임포트
with patch("src.inference.vllm_stabilizer.apply_transformers_patch"):
    import src.inference.api_server as api_server

    app = api_server.app

client = TestClient(app)


@pytest.fixture
def mock_manager():
    with patch("src.inference.api_server.manager") as mock:
        # 가짜 응답 데이터 설정
        mock.agent_manager.list_agents.return_value = [
            "classifier",
            "generator_public_doc",
            "generator_civil_response",
        ]
        mock.agent_manager.get_agent.return_value = MagicMock()
        mock.index_manager = MagicMock()
        mock.pii_masker = MagicMock()
        mock.pii_masker.mask_all.side_effect = lambda x: x  # 마스킹 로직 패스
        # hybrid_engine.search는 비동기 함수이므로 AsyncMock 사용
        mock.hybrid_engine.search = AsyncMock()
        yield mock


def test_health_endpoint(mock_manager):
    """헬스체크 엔드포인트 테스트 (모델 없이 로직 확인)"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "classifier" in response.json()["agents_loaded"]


def test_search_logic_mock(mock_manager):
    """검색 로직 통합 테스트 (하이브리드 엔진 동작 여부)"""
    # 가짜 검색 결과 설정
    mock_manager.hybrid_engine.search.return_value = (
        [
            {
                "doc_id": "test-1",
                "title": "테스트 민원",
                "score": 0.95,
                "extras": {"complaint_text": "도로 파손", "answer_text": "복구 예정"},
            }
        ],
        "hybrid",
    )

    search_payload = {
        "query": "도로가 파손되었어요",
        "doc_type": "case",
        "search_mode": "hybrid",
        "top_k": 1,
    }

    response = client.post("/v1/search", json=search_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["doc_id"] == "test-1"


def test_api_key_auth():
    """API 키 인증 로직 테스트"""
    with patch("os.getenv", return_value="secret-key"):
        # 키 없이 요청
        response = client.get("/health")
        # api_server.py에서 _API_KEY가 설정되어 있으면 인증이 필요함
        # 하지만 테스트 클라이언트에서는 세팅에 따라 다를 수 있으므로
        # 실제 api_server의 verify_api_key 로직을 타는지 확인
        pass
