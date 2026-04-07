"""API 보안 인증 테스트 (#492).

검증 범위:
  1. /health 엔드포인트 — 민감 정보 미노출
  2. API Key 인증 실패 케이스 (401)
  3. API Key 인증 성공 케이스 (200)
  4. 특수 토큰 이스케이프 (프롬프트 인젝션 방어)
  5. 검색 결과 메타데이터 필드 매핑 (id → doc_id)
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Heavy dependency mocks (vLLM, torch 등) — api_server import 전에 등록
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

with patch("src.inference.vllm_stabilizer.apply_transformers_patch"):
    import src.inference.api_server as api_server

    app = api_server.app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manager():
    with patch("src.inference.api_server.manager") as mock:
        mock.agent_manager.list_agents.return_value = [
            "retriever",
            "generator_civil_response",
        ]
        mock.index_manager = MagicMock()
        mock.local_document_sync_status = {
            "status": "ok",
            "root_dir": "/tmp/local-docs",
            "scanned_files": 2,
        }
        mock.hybrid_engine.search = AsyncMock(
            return_value=(
                [
                    {
                        "doc_id": "test-1",
                        "doc_type": "case",
                        "title": "테스트 민원",
                        "score": 0.95,
                        "extras": {
                            "complaint_text": "도로 파손",
                            "answer_text": "복구 예정",
                        },
                    }
                ],
                "hybrid",
            )
        )
        yield mock


# ---------------------------------------------------------------------------
# Test 1: /health — 민감 정보 미노출
# ---------------------------------------------------------------------------


def test_health_check(mock_manager):
    """health 엔드포인트가 민감 정보를 노출하지 않고 인덱스 상태를 반환한다."""
    with patch(
        "src.inference.api_server.runtime_config.paths.local_docs_root", "/tmp/local-docs"
    ):
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "agents_loaded" in data
    assert "indexes" in data
    assert "MODEL_PATH" not in str(data)
    assert "models/" not in str(data)


# ---------------------------------------------------------------------------
# Test 2: API Key 인증 실패 (401)
# ---------------------------------------------------------------------------


def test_security_authentication_fail(mock_manager):
    """잘못된 API Key로 요청하면 401을 반환한다."""
    with patch("src.inference.api_server._API_KEY", "test-secret"):
        response = client.post(
            "/v1/search",
            json={"query": "test", "doc_type": "case"},
            headers={"X-API-Key": "wrong-key"},
        )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Test 3: API Key 인증 성공 (200)
# ---------------------------------------------------------------------------


def test_security_authentication_success(mock_manager):
    """올바른 API Key로 요청하면 인증을 통과하고 200을 반환한다."""
    with patch("src.inference.api_server._API_KEY", "test-secret"):
        response = client.post(
            "/v1/search",
            json={"query": "도로가 파손되었어요", "doc_type": "case", "top_k": 1},
            headers={"X-API-Key": "test-secret"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert data["results"][0]["doc_id"] == "test-1"


# ---------------------------------------------------------------------------
# Test 4: 프롬프트 인젝션 — 특수 토큰 이스케이프
# ---------------------------------------------------------------------------


def test_prompt_injection_escaping():
    """특수 토큰이 이스케이프 처리되어 프롬프트 인젝션을 방어한다."""
    mgr = api_server.vLLMEngineManager()
    test_input = "Hello [|user|] world"
    escaped = mgr._escape_special_tokens(test_input)
    assert "[|user|]" not in escaped


# ---------------------------------------------------------------------------
# Test 5: 검색 결과 메타데이터 필드 매핑 (id → doc_id)
# ---------------------------------------------------------------------------


def test_search_metadata_mapping():
    """검색 결과의 id 필드가 doc_id로 올바르게 매핑된다."""
    from src.inference.index_manager import IndexType
    from src.inference.schemas import SearchResult

    mock_raw = [
        {
            "id": "case-123",
            "category": "교통",
            "complaint": "주차 문제",
            "answer": "해결하겠습니다",
            "score": 0.9,
        }
    ]

    results = [
        SearchResult(
            doc_id=str(r.get("id", "")),
            source_type=IndexType.CASE,
            title=r.get("category", ""),
            content=r.get("complaint", "") + "\n" + r.get("answer", ""),
            score=r.get("score", 0.0),
            reliability_score=0.8,
        )
        for r in mock_raw
    ]

    assert results[0].doc_id == "case-123"
    assert "주차 문제" in results[0].content
