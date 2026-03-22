"""
/search 및 /health API 엔드포인트 테스트.

vLLM, SentenceTransformer 등 무거운 의존성을 mock하고
FastAPI TestClient로 API 동작을 검증한다.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 무거운 외부 의존성 mock (api_server import 전에 등록해야 함)
# ---------------------------------------------------------------------------

# vllm 관련 모듈 mock
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.engine", MagicMock())
sys.modules.setdefault("vllm.engine.arg_utils", MagicMock())
sys.modules.setdefault("vllm.engine.async_llm_engine", MagicMock())
sys.modules.setdefault("vllm.sampling_params", MagicMock())

# sentence_transformers mock
sys.modules.setdefault("sentence_transformers", MagicMock())

# vllm_stabilizer 패치 함수 mock
_mock_stabilizer = MagicMock()
_mock_stabilizer.apply_transformers_patch = MagicMock()
sys.modules.setdefault("src.inference.vllm_stabilizer", _mock_stabilizer)

# retriever mock
_mock_retriever_module = MagicMock()
sys.modules.setdefault("src.inference.retriever", _mock_retriever_module)

from fastapi.testclient import TestClient

from src.inference.api_server import app, manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_manager():
    """모든 테스트에서 manager를 초기화된 상태로 mock한다."""
    original_engine = manager.engine
    original_retriever = manager.retriever
    original_generate = manager.generate

    manager.engine = MagicMock()
    manager.retriever = MagicMock()
    manager.retriever.search.return_value = [
        {"id": "case-001", "category": "세금", "complaint": "테스트 민원", "answer": "테스트 답변", "score": 0.95}
    ]
    yield

    manager.engine = original_engine
    manager.retriever = original_retriever
    manager.generate = original_generate


@pytest.fixture
def client():
    """FastAPI TestClient."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health 엔드포인트 테스트
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_indexes(self, client):
        response = client.get("/health")
        data = response.json()
        assert "indexes" in data

    def test_health_contains_rag_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "rag_enabled" in data


# ---------------------------------------------------------------------------
# /v1/generate 엔드포인트 테스트
# ---------------------------------------------------------------------------


class TestGenerateEndpoint:
    def _setup_generate_mock(self):
        """vLLM 엔진의 generate를 mock한다."""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "테스트 응답입니다."
        mock_output.outputs[0].token_ids = list(range(10))
        mock_output.prompt_token_ids = list(range(20))

        async def mock_generate(*args, **kwargs):
            async def gen():
                yield mock_output
            return gen(), []

        manager.generate = AsyncMock(side_effect=mock_generate)
        return mock_output

    def test_generate_returns_200(self, client):
        mock_output = self._setup_generate_mock()

        async def fake_generate(request, request_id):
            async def gen():
                yield mock_output
            return gen(), []

        manager.generate = AsyncMock(side_effect=fake_generate)

        response = client.post("/v1/generate", json={
            "prompt": "테스트 프롬프트",
            "max_tokens": 100,
            "stream": False,
        })
        assert response.status_code == 200

    def test_generate_response_structure(self, client):
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "생성된 답변"
        mock_output.outputs[0].token_ids = list(range(5))
        mock_output.prompt_token_ids = list(range(15))

        async def fake_generate(request, request_id):
            async def gen():
                yield mock_output
            return gen(), []

        manager.generate = AsyncMock(side_effect=fake_generate)

        response = client.post("/v1/generate", json={
            "prompt": "구조 테스트",
            "stream": False,
        })
        data = response.json()
        assert "request_id" in data
        assert "text" in data
        assert "prompt_tokens" in data
        assert "completion_tokens" in data

    def test_generate_stream_flag_rejected(self, client):
        """stream=True로 /v1/generate를 호출하면 400 에러."""
        response = client.post("/v1/generate", json={
            "prompt": "스트림 테스트",
            "stream": True,
        })
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# /search 엔드포인트 테스트
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    """POST /search 엔드포인트 테스트."""

    @pytest.fixture(autouse=True)
    def _setup_search(self):
        """검색에 필요한 index_manager와 embed_model을 mock."""
        # index_manager mock
        mock_mgr = MagicMock()
        mock_search_results = [
            {
                "doc_id": f"case-{i:04d}",
                "doc_type": "case",
                "source": "AI Hub",
                "title": f"테스트 민원 {i}",
                "category": "도로/교통",
                "reliability_score": 0.6,
                "score": round(0.95 - i * 0.03, 4),
                "extras": {"complaint_text": f"민원 {i}", "answer_text": f"답변 {i}"},
            }
            for i in range(5)
        ]
        mock_mgr.search.return_value = mock_search_results
        mock_mgr.get_index_stats.return_value = {
            "base_dir": "/tmp", "embedding_dim": 1024,
            "indexes": {"case": {"loaded": True, "doc_count": 5}}
        }
        manager.index_manager = mock_mgr

        # embed_model mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
        manager.embed_model = mock_model

        yield

        manager.index_manager = None
        manager.embed_model = None

    def test_search_returns_200(self, client):
        """기본 검색 요청이 200을 반환한다."""
        response = client.post("/search", json={
            "query": "도로 보수 요청",
            "top_k": 5,
            "doc_type": "case",
        })
        assert response.status_code == 200

    def test_search_response_structure(self, client):
        """응답에 query, doc_type, results, total이 포함된다."""
        response = client.post("/search", json={
            "query": "도로 보수 요청",
            "top_k": 5,
            "doc_type": "case",
        })
        data = response.json()
        assert "query" in data
        assert "doc_type" in data
        assert "results" in data
        assert "total" in data
        assert data["query"] == "도로 보수 요청"
        assert data["doc_type"] == "case"
        assert isinstance(data["results"], list)
        assert data["total"] == len(data["results"])

    def test_search_results_have_required_fields(self, client):
        """각 결과에 doc_id, title, score, source_type이 포함된다."""
        response = client.post("/search", json={
            "query": "민원 테스트",
            "top_k": 3,
            "doc_type": "case",
        })
        data = response.json()
        for result in data["results"]:
            assert "doc_id" in result
            assert "title" in result
            assert "score" in result
            assert "source_type" in result

    def test_search_empty_query_returns_422(self, client):
        """빈 쿼리 시 422 Validation Error를 반환한다."""
        response = client.post("/search", json={
            "query": "",
            "top_k": 5,
            "doc_type": "case",
        })
        assert response.status_code == 422

    def test_search_invalid_doc_type_returns_400_or_422(self, client):
        """잘못된 doc_type 시 400 또는 422를 반환한다."""
        response = client.post("/search", json={
            "query": "테스트 쿼리",
            "top_k": 5,
            "doc_type": "invalid_type",
        })
        assert response.status_code in (400, 422)

    def test_search_top_k_validation(self, client):
        """top_k=0 또는 51 시 422를 반환한다."""
        # top_k=0
        response = client.post("/search", json={
            "query": "테스트 쿼리",
            "top_k": 0,
            "doc_type": "case",
        })
        assert response.status_code == 422

        # top_k=51
        response = client.post("/search", json={
            "query": "테스트 쿼리",
            "top_k": 51,
            "doc_type": "case",
        })
        assert response.status_code == 422

    def test_search_not_initialized_returns_503(self, client):
        """index_manager=None 시 503을 반환한다."""
        manager.index_manager = None

        response = client.post("/search", json={
            "query": "테스트 쿼리",
            "top_k": 5,
            "doc_type": "case",
        })
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# 에러 케이스 테스트
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_missing_prompt(self, client):
        """prompt 필드가 없으면 422 Validation Error."""
        response = client.post("/v1/generate", json={
            "max_tokens": 100,
        })
        assert response.status_code == 422

    def test_invalid_temperature(self, client):
        """temperature 범위 초과 시 422 Validation Error."""
        response = client.post("/v1/generate", json={
            "prompt": "테스트",
            "temperature": 5.0,
        })
        assert response.status_code == 422

    def test_invalid_top_p(self, client):
        """top_p 범위 초과 시 422 Validation Error."""
        response = client.post("/v1/generate", json={
            "prompt": "테스트",
            "top_p": 2.0,
        })
        assert response.status_code == 422

    def test_negative_max_tokens(self, client):
        """max_tokens가 0 이하이면 422 Validation Error."""
        response = client.post("/v1/generate", json={
            "prompt": "테스트",
            "max_tokens": 0,
        })
        assert response.status_code == 422

    def test_nonexistent_endpoint(self, client):
        """존재하지 않는 경로에 대해 404."""
        response = client.get("/v1/nonexistent")
        assert response.status_code == 404

    def test_wrong_method(self, client):
        """GET으로 /v1/generate 호출 시 405."""
        response = client.get("/v1/generate")
        assert response.status_code == 405
