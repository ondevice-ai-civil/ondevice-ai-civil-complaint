"""
/v1/search 엔드포인트 search_mode별 E2E 통합 테스트.

HybridSearchEngine(dense/sparse/hybrid) 및 레거시 retriever 폴백을
FastAPI TestClient로 검증한다.

Issue: #154
"""

import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# heavy-deps mock은 conftest.py에서 처리
# retriever mock: api_server import 시에만 필요하고, test_retriever.py와
# 충돌하지 않도록 import 후 즉시 제거한다.
_retriever_stub = MagicMock()
with patch.dict(sys.modules, {"src.inference.retriever": _retriever_stub}):
    from fastapi.testclient import TestClient

    from src.inference.api_server import app, manager
    from src.inference.hybrid_search import SearchMode

# ---------------------------------------------------------------------------
# 공통 Fixtures
# ---------------------------------------------------------------------------

_BASE_SEARCH_PAYLOAD: Dict[str, Any] = {
    "query": "도로 파손 민원",
    "doc_type": "case",
    "top_k": 5,
}


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_manager():
    """각 테스트 전후로 manager 상태를 초기화한다."""
    original_engine = manager.engine
    original_retriever = manager.retriever
    original_hybrid = manager.hybrid_engine

    manager.engine = MagicMock()
    manager.retriever = None
    manager.hybrid_engine = None

    yield

    manager.engine = original_engine
    manager.retriever = original_retriever
    manager.hybrid_engine = original_hybrid


def _make_fake_search_results(count: int = 5) -> List[Dict[str, Any]]:
    """HybridSearchEngine.search()가 반환할 테스트 데이터를 생성한다."""
    return [
        {
            "doc_id": f"case-{i:03d}",
            "doc_type": "case",
            "title": f"테스트 민원 {i}",
            "category": "환경/위생",
            "source": "AI Hub",
            "reliability_score": 0.8,
            "score": round(1.0 - i * 0.1, 2),
            "chunk_index": 0,
            "chunk_total": 1,
            "extras": {
                "complaint_text": f"민원 내용 {i}",
                "answer_text": f"답변 {i}",
            },
        }
        for i in range(count)
    ]


@pytest.fixture
def setup_hybrid_engine() -> None:
    """HybridSearchEngine mock 설정 — 여러 테스트 클래스에서 공통 사용."""

    async def fake_search(
        query: str,
        index_type: Any,
        top_k: int = 5,
        mode: Optional[SearchMode] = None,
    ) -> Tuple[List[Dict[str, Any]], SearchMode]:
        return _make_fake_search_results(min(top_k, 5)), mode or SearchMode.HYBRID

    manager.hybrid_engine = MagicMock()
    manager.hybrid_engine.search = AsyncMock(side_effect=fake_search)


# ---------------------------------------------------------------------------
# 1. TestSearchModeEndpoint — search_mode별 정상 동작 검증
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_hybrid_engine")
class TestSearchModeEndpoint:
    """HybridSearchEngine을 mock하여 /v1/search의 search_mode별 동작을 검증한다."""

    def test_search_hybrid_mode_200(self, client: TestClient) -> None:
        """search_mode=hybrid 요청이 200을 반환한다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "search_mode": "hybrid"}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 200

    def test_search_dense_mode_200(self, client: TestClient) -> None:
        """search_mode=dense 요청이 200을 반환한다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "search_mode": "dense"}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 200

    def test_search_sparse_mode_200(self, client: TestClient) -> None:
        """search_mode=sparse 요청이 200을 반환한다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "search_mode": "sparse"}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 200

    def test_search_invalid_mode_422(self, client: TestClient) -> None:
        """유효하지 않은 search_mode 값은 422 Validation Error를 반환한다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "search_mode": "invalid"}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 422

    def test_response_includes_search_mode(self, client: TestClient) -> None:
        """응답 body에 search_mode 필드가 포함된다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "search_mode": "sparse"}
        response = client.post("/v1/search", json=payload)
        data = response.json()
        assert "search_mode" in data
        assert data["search_mode"] == "sparse"

    def test_response_includes_search_time_ms(self, client: TestClient) -> None:
        """응답 body에 search_time_ms 필드가 있고 0보다 크다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "search_mode": "hybrid"}
        response = client.post("/v1/search", json=payload)
        data = response.json()
        assert "search_time_ms" in data
        assert data["search_time_ms"] > 0

    def test_default_search_mode_is_hybrid(self, client: TestClient) -> None:
        """search_mode 미지정 시 기본값 hybrid가 사용된다."""
        payload = {"query": "도로 파손 민원", "doc_type": "case", "top_k": 5}
        response = client.post("/v1/search", json=payload)
        data = response.json()
        assert data["search_mode"] == "hybrid"


# ---------------------------------------------------------------------------
# 2. TestBackwardCompatibility — 하위 호환성 검증
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_hybrid_engine")
class TestBackwardCompatibility:
    """레거시 /search 경로 및 search_mode 미지정 요청의 하위 호환성을 검증한다."""

    def test_legacy_search_endpoint_works(self, client: TestClient) -> None:
        """POST /search (v1 prefix 없이) 요청이 200을 반환한다."""
        response = client.post("/search", json=_BASE_SEARCH_PAYLOAD)
        assert response.status_code == 200

    def test_search_without_mode_uses_default(self, client: TestClient) -> None:
        """search_mode 없는 요청이 정상 동작하고 기본값 hybrid를 사용한다."""
        payload = {"query": "민원 접수", "doc_type": "case", "top_k": 3}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "hybrid"


# ---------------------------------------------------------------------------
# 3. TestFallbackBehavior — retriever 폴백 및 503 검증
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    """HybridSearchEngine 미사용 시 레거시 retriever 폴백 동작을 검증한다."""

    @pytest.fixture
    def _setup_legacy_only(self) -> None:
        """HybridSearchEngine 없고 retriever만 있는 상태."""
        manager.hybrid_engine = None
        manager.retriever = MagicMock()
        manager.retriever.search.return_value = [
            {
                "doc_id": "case-001",
                "category": "환경",
                "complaint": "테스트 민원 내용",
                "answer": "답변 내용",
                "score": 0.9,
            }
        ]
        yield
        manager.retriever = None

    def test_fallback_to_retriever_when_no_hybrid(
        self, client: TestClient, _setup_legacy_only: None
    ) -> None:
        """hybrid_engine=None 시 retriever로 폴백하여 200을 반환한다."""
        response = client.post("/v1/search", json=_BASE_SEARCH_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0

    def test_fallback_returns_dense_mode(
        self, client: TestClient, _setup_legacy_only: None
    ) -> None:
        """retriever 폴백 시 search_mode는 요청값(hybrid) 유지, actual_search_mode가 dense."""
        response = client.post("/v1/search", json=_BASE_SEARCH_PAYLOAD)
        data = response.json()
        assert data["search_mode"] == "hybrid"
        assert data.get("actual_search_mode") == "dense"

    def test_503_when_no_engine_and_no_retriever(self, client: TestClient) -> None:
        """hybrid_engine=None, retriever=None 시 503을 반환한다."""
        manager.hybrid_engine = None
        manager.retriever = None
        response = client.post("/v1/search", json=_BASE_SEARCH_PAYLOAD)
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# 4. TestResponseStructure — 응답 구조 검증
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_hybrid_engine")
class TestResponseStructure:
    """검색 응답의 데이터 구조와 필드 정합성을 검증한다."""

    def test_search_results_have_required_fields(self, client: TestClient) -> None:
        """각 결과에 doc_id, source_type, title, content, score가 포함된다."""
        response = client.post("/v1/search", json=_BASE_SEARCH_PAYLOAD)
        data = response.json()
        required_fields = {"doc_id", "source_type", "title", "content", "score"}
        for result in data["results"]:
            assert required_fields.issubset(
                result.keys()
            ), f"누락된 필드: {required_fields - result.keys()}"

    def test_search_total_matches_results_length(self, client: TestClient) -> None:
        """total 값이 results 리스트의 길이와 일치한다."""
        response = client.post("/v1/search", json=_BASE_SEARCH_PAYLOAD)
        data = response.json()
        assert data["total"] == len(data["results"])

    def test_top_k_respected(self, client: TestClient) -> None:
        """top_k=2 요청 시 결과가 최대 2개이다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "top_k": 2}
        response = client.post("/v1/search", json=payload)
        data = response.json()
        assert len(data["results"]) <= 2


# ---------------------------------------------------------------------------
# 5. TestSearchQueryVariety — 다양한 쿼리/파라미터 시나리오
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_hybrid_engine")
class TestSearchQueryVariety:
    """다양한 쿼리 패턴과 파라미터 조합의 검색 E2E 검증."""

    def test_search_with_long_query(self, client: TestClient) -> None:
        """100자 이상 긴 쿼리로 검색 시 정상 처리한다."""
        long_query = "서울시 강남구에서 접수된 도로 파손 관련 민원으로 " * 5
        payload = {**_BASE_SEARCH_PAYLOAD, "query": long_query}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 200
        assert len(response.json()["results"]) > 0

    def test_search_with_doc_type_law(self, client: TestClient) -> None:
        """doc_type=law 검색이 정상 동작한다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "doc_type": "law"}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 200

    def test_search_with_doc_type_manual(self, client: TestClient) -> None:
        """doc_type=manual 검색이 정상 동작한다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "doc_type": "manual"}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 200

    def test_search_with_top_k_1(self, client: TestClient) -> None:
        """top_k=1 검색 시 결과가 최대 1개이다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "top_k": 1}
        response = client.post("/v1/search", json=payload)
        data = response.json()
        assert response.status_code == 200
        assert len(data["results"]) <= 1

    def test_search_with_max_top_k(self, client: TestClient) -> None:
        """top_k=50 검색 시 정상 처리한다."""
        payload = {**_BASE_SEARCH_PAYLOAD, "top_k": 50}
        response = client.post("/v1/search", json=payload)
        assert response.status_code == 200
