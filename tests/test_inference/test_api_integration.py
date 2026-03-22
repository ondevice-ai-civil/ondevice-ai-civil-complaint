"""
FAISS 벡터 검색 시스템 API 통합 테스트.

테스트 대상:
- POST /search : MultiIndexManager 기반 문서 검색
- POST /v1/generate : RAG 기반 텍스트 생성 (use_rag=true/false)
- GET /health : 인덱스 상태 포함 헬스체크

무거운 의존성(vLLM, SentenceTransformer, transformers)은 import 전에
sys.modules 패치로 mock 처리한다.

NOTE: /search 엔드포인트와 SearchRequest/SearchResponse 스키마는
Phase 1 구현(이슈 #53)에서 추가된다. 엔드포인트가 미구현 상태에서는
해당 테스트가 xfail 처리된다.
"""

import sys
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. 무거운 의존성을 import 전에 mock 등록
# ---------------------------------------------------------------------------

# vllm 관련 모듈
_vllm_mock = MagicMock()
sys.modules.setdefault("vllm", _vllm_mock)
sys.modules.setdefault("vllm.engine", _vllm_mock)
sys.modules.setdefault("vllm.engine.arg_utils", _vllm_mock)
sys.modules.setdefault("vllm.engine.async_llm_engine", _vllm_mock)
sys.modules.setdefault("vllm.sampling_params", _vllm_mock)

# sentence_transformers mock
_st_mock = MagicMock()
sys.modules.setdefault("sentence_transformers", _st_mock)

# transformers mock (vllm_stabilizer에서 사용)
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("transformers.modeling_rope_utils", MagicMock())
sys.modules.setdefault("transformers.utils", MagicMock())
sys.modules.setdefault("transformers.utils.generic", MagicMock())

# torch mock (vllm_stabilizer에서 사용)
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

# ---------------------------------------------------------------------------
# 2. api_server import (vllm_stabilizer.apply_transformers_patch 를 no-op 처리)
# ---------------------------------------------------------------------------

with patch("src.inference.vllm_stabilizer.apply_transformers_patch"):
    from src.inference.api_server import app, manager

from src.inference.index_manager import DocumentMetadata, IndexType, MultiIndexManager

from fastapi.testclient import TestClient

# /search 엔드포인트 존재 여부 확인
_has_search_endpoint = any(
    route.path == "/search" for route in app.routes if hasattr(route, "path")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vllm_output_mock(text="테스트 응답입니다."):
    """vLLM 엔진의 generate 결과를 시뮬레이션하는 async generator를 반환한다."""
    output_mock = MagicMock()
    output_mock.outputs = [MagicMock()]
    output_mock.outputs[0].text = text
    output_mock.outputs[0].token_ids = list(range(10))
    output_mock.prompt_token_ids = list(range(5))
    output_mock.finished = True

    async def _gen(*args, **kwargs):
        yield output_mock

    return _gen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_lifespan():
    """lifespan 이벤트의 manager.initialize()를 mock하여 무거운 초기화를 방지한다."""
    with patch.object(manager, "initialize", new_callable=AsyncMock):
        yield


@pytest.fixture
def client(_patch_lifespan):
    """FastAPI TestClient."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def test_index_manager(tmp_path):
    """테스트용 MultiIndexManager를 생성하고 case 인덱스에 샘플 데이터를 추가한다.

    faiss가 mock 상태인 환경에서는 실제 인덱스 대신 mock 기반 매니저를 반환한다.
    """
    import faiss as _faiss

    # faiss가 mock인지 확인
    if isinstance(_faiss, MagicMock):
        # mock 기반 MultiIndexManager 생성
        mock_mgr = MagicMock(spec=MultiIndexManager)
        mock_mgr.base_dir = str(tmp_path)
        mock_mgr.embedding_dim = 1024

        # search 결과 mock
        now = datetime.now(timezone.utc).isoformat()
        mock_search_results = [
            {
                "doc_id": f"case-{i:04d}",
                "doc_type": "case",
                "source": "AI Hub",
                "title": f"테스트 민원 {i}",
                "category": "도로/교통",
                "reliability_score": 0.6,
                "created_at": now,
                "updated_at": now,
                "score": round(0.95 - i * 0.03, 4),
                "extras": {
                    "complaint_text": f"테스트 민원 내용 {i}",
                    "answer_text": f"테스트 답변 {i}",
                },
            }
            for i in range(20)
        ]

        def _mock_search(index_type, query_vector, top_k=5):
            return mock_search_results[:top_k]

        mock_mgr.search = MagicMock(side_effect=_mock_search)

        # get_index_stats mock
        mock_mgr.get_index_stats.return_value = {
            "base_dir": str(tmp_path),
            "embedding_dim": 1024,
            "indexes": {
                "case": {
                    "loaded": True,
                    "doc_count": 20,
                    "index_class": "IndexFlatIP",
                    "metadata_count": 20,
                    "last_updated": now,
                },
                "law": {"loaded": False, "doc_count": 0, "last_updated": None},
                "manual": {"loaded": False, "doc_count": 0, "last_updated": None},
                "notice": {"loaded": False, "doc_count": 0, "last_updated": None},
            },
        }
        return mock_mgr

    # faiss가 실제로 설치된 환경
    mgr = MultiIndexManager(base_dir=str(tmp_path), embedding_dim=1024)

    rng = np.random.default_rng(42)
    vectors = rng.random((20, 1024), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    now = datetime.now(timezone.utc).isoformat()
    metadata_list = [
        DocumentMetadata(
            doc_id=f"case-{i:04d}",
            doc_type="case",
            source="AI Hub",
            title=f"테스트 민원 {i}",
            category="도로/교통",
            reliability_score=0.6,
            created_at=now,
            updated_at=now,
            extras={
                "complaint_text": f"테스트 민원 내용 {i}",
                "answer_text": f"테스트 답변 {i}",
            },
        )
        for i in range(20)
    ]

    mgr.add_documents(IndexType.CASE, vectors, metadata_list)
    return mgr


@pytest.fixture
def mock_embed_model():
    """SentenceTransformer.encode()를 mock하여 정규화된 1024차원 벡터를 반환한다."""
    model = MagicMock()

    def _encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        rng = np.random.default_rng(42)
        vecs = rng.random((n, 1024)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    model.encode = MagicMock(side_effect=_encode)
    return model


@pytest.fixture
def client_with_index(client, test_index_manager, mock_embed_model):
    """MultiIndexManager와 embed_model이 설정된 상태의 TestClient."""
    original_index_manager = getattr(manager, "index_manager", None)
    original_embed_model = getattr(manager, "embed_model", None)
    original_retriever = getattr(manager, "retriever", None)

    manager.index_manager = test_index_manager
    manager.embed_model = mock_embed_model
    # retriever도 설정하여 /health의 rag_enabled 검증 지원
    if original_retriever is None:
        manager.retriever = MagicMock()

    yield client

    # 복원
    manager.index_manager = original_index_manager
    manager.embed_model = original_embed_model
    manager.retriever = original_retriever


# ---------------------------------------------------------------------------
# /search 엔드포인트 미구현 시 skip 처리용 마커
# ---------------------------------------------------------------------------

requires_search = pytest.mark.skipif(
    not _has_search_endpoint,
    reason="/search 엔드포인트 미구현 (Phase 1 이슈 #53 대기)",
)


# ---------------------------------------------------------------------------
# 정상 동작 테스트: POST /search
# ---------------------------------------------------------------------------


@requires_search
class TestSearchEndpoint:
    """POST /search 엔드포인트 테스트."""

    def test_search_basic(self, client_with_index):
        """기본 검색: query='도로 보수 요청', top_k=5, doc_type='case'."""
        payload = {
            "query": "도로 보수 요청",
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "도로 보수 요청"
        assert body["doc_type"] == "case"
        assert isinstance(body["results"], list)
        assert len(body["results"]) <= 5
        assert body["total"] == len(body["results"])

        # 각 결과에 필수 필드 존재 확인
        for result in body["results"]:
            assert "doc_id" in result
            assert "title" in result
            assert "score" in result
            assert isinstance(result["score"], float)

    @pytest.mark.parametrize("top_k", [1, 10, 50])
    def test_search_top_k_variations(self, client_with_index, top_k):
        """top_k 파라미터(1, 10, 50)에 따라 결과 수가 올바르게 제한된다."""
        payload = {
            "query": "민원 상담",
            "top_k": top_k,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        # 인덱스에 20개 문서가 있으므로 min(top_k, 20) 이하
        expected_max = min(top_k, 20)
        assert len(body["results"]) <= expected_max
        assert body["total"] == len(body["results"])

    def test_search_doc_type_case(self, client_with_index):
        """doc_type='case'로 검색 시 결과의 source_type이 'case'이다."""
        payload = {
            "query": "환경 민원",
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] == "case"
        for result in body["results"]:
            assert result.get("source_type") == "case"

    def test_search_results_have_score_descending(self, client_with_index):
        """검색 결과가 score 내림차순으로 정렬되어 있다."""
        payload = {
            "query": "도로 보수",
            "top_k": 10,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code == 200
        results = resp.json()["results"]
        if len(results) > 1:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 에러 케이스 테스트: POST /search
# ---------------------------------------------------------------------------


@requires_search
class TestSearchErrorCases:
    """POST /search 에러 케이스 테스트."""

    def test_search_empty_query_returns_422(self, client_with_index):
        """빈 쿼리 문자열이 422 Validation Error를 반환한다."""
        payload = {
            "query": "",
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body

    def test_search_missing_query_returns_422(self, client_with_index):
        """query 필드 누락 시 422 Validation Error를 반환한다."""
        payload = {
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code == 422

    def test_search_invalid_doc_type_returns_400(self, client_with_index):
        """존재하지 않는 doc_type은 400 또는 422를 반환한다."""
        payload = {
            "query": "테스트 쿼리",
            "top_k": 5,
            "doc_type": "invalid_type",
        }
        resp = client_with_index.post("/search", json=payload)

        # doc_type 유효성 검증 방식에 따라 400 또는 422
        assert resp.status_code in (400, 422)

    def test_search_top_k_zero_returns_422(self, client_with_index):
        """top_k=0은 422 Validation Error를 반환한다 (ge=1 제약 위반)."""
        payload = {
            "query": "테스트 쿼리",
            "top_k": 0,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code == 422

    def test_search_top_k_exceeds_max_returns_422(self, client_with_index):
        """top_k=51은 422 Validation Error를 반환한다 (le=50 제약 위반)."""
        payload = {
            "query": "테스트 쿼리",
            "top_k": 51,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code == 422

    def test_search_index_not_initialized_returns_503(self, client):
        """MultiIndexManager가 초기화되지 않은 경우 503 Service Unavailable을 반환한다."""
        original = getattr(manager, "index_manager", None)
        manager.index_manager = None

        try:
            payload = {
                "query": "테스트 쿼리",
                "top_k": 5,
                "doc_type": "case",
            }
            resp = client.post("/search", json=payload)

            assert resp.status_code == 503
        finally:
            manager.index_manager = original


# ---------------------------------------------------------------------------
# 성능 관련 기본 검증: POST /search
# ---------------------------------------------------------------------------


@requires_search
class TestSearchPerformance:
    """검색 API 기본 성능 검증."""

    def test_search_response_time_under_threshold(self, client_with_index):
        """단일 검색 요청의 응답 시간이 2초 이내이다."""
        payload = {
            "query": "도로 보수 요청",
            "top_k": 5,
            "doc_type": "case",
        }

        start = time.monotonic()
        resp = client_with_index.post("/search", json=payload)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert resp.status_code == 200
        # 테스트 환경(mock + 20개 문서)에서 2초 이내
        assert elapsed_ms < 2000, f"응답 시간 {elapsed_ms:.1f}ms가 2초를 초과"

    def test_search_multiple_sequential_requests(self, client_with_index):
        """다수의 순차 요청이 모두 정상 응답한다."""
        payload = {
            "query": "민원 처리 절차",
            "top_k": 5,
            "doc_type": "case",
        }

        results = []
        for _ in range(10):
            resp = client_with_index.post("/search", json=payload)
            results.append(resp.status_code)

        assert all(code == 200 for code in results)


# ---------------------------------------------------------------------------
# 보안 기본 검증: POST /search
# ---------------------------------------------------------------------------


@requires_search
class TestSearchSecurity:
    """검색 API 보안 기본 검증."""

    def test_search_sql_injection_attempt(self, client_with_index):
        """SQL 인젝션 시도가 500 에러를 유발하지 않는다."""
        payload = {
            "query": "'; DROP TABLE users; --",
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        # 500이 아닌 정상 응답(200) 또는 클라이언트 에러(4xx)
        assert resp.status_code != 500

    def test_search_xss_attempt(self, client_with_index):
        """XSS 시도가 500 에러를 유발하지 않는다."""
        payload = {
            "query": "<script>alert('xss')</script>",
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        assert resp.status_code != 500

    def test_search_oversized_query(self, client_with_index):
        """매우 긴 쿼리 문자열이 서버 크래시를 유발하지 않는다."""
        payload = {
            "query": "테스트 " * 10000,
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_index.post("/search", json=payload)

        # 서버가 크래시하지 않으면 성공 (200 또는 4xx)
        assert resp.status_code < 500


# ---------------------------------------------------------------------------
# 정상 동작 테스트: POST /v1/generate
# ---------------------------------------------------------------------------


class TestGenerateEndpoint:
    """POST /v1/generate 엔드포인트 테스트."""

    def test_generate_with_rag_returns_retrieved_cases(self, client_with_index):
        """use_rag=true 시 retrieved_cases 필드가 포함된다."""
        # retriever.search mock 설정
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(
            return_value=[
                {
                    "complaint": "도로 파손 민원",
                    "answer": "도로 보수 처리 완료",
                    "score": 0.95,
                },
            ]
        )
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock()()
        )

        payload = {
            "prompt": "도로 보수 관련 민원에 대해 답변해주세요.",
            "max_tokens": 128,
            "temperature": 0.7,
            "use_rag": True,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "request_id" in body
        assert "text" in body
        assert body["text"] != ""
        # use_rag=true이므로 retrieved_cases가 포함
        assert "retrieved_cases" in body
        assert body["retrieved_cases"] is not None
        assert len(body["retrieved_cases"]) > 0

    def test_generate_with_rag_search_results_field(self, client_with_index):
        """use_rag=true 시 search_results 필드가 포함된다 (Phase 1 확장 응답).

        Phase 1 이전에는 search_results가 None일 수 있으며,
        Phase 1 구현 후에는 리스트가 반환되어야 한다.
        """
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock()()
        )

        payload = {
            "prompt": "도로 보수 관련 민원",
            "max_tokens": 128,
            "use_rag": True,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        # search_results 필드는 스키마에 정의되어 있으므로 응답에 존재해야 함
        assert "search_results" in body

    def test_generate_without_rag_no_retrieved_cases(self, client_with_index):
        """use_rag=false 시 retrieved_cases가 빈 리스트이다."""
        manager.retriever = MagicMock()
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock()()
        )

        payload = {
            "prompt": "일반 질문입니다.",
            "max_tokens": 128,
            "temperature": 0.7,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body
        # use_rag=false이므로 retrieved_cases가 빈 리스트
        retrieved = body.get("retrieved_cases")
        assert retrieved is None or retrieved == []

    def test_generate_response_has_token_counts(self, client_with_index):
        """응답에 prompt_tokens, completion_tokens가 포함된다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock()()
        )

        payload = {
            "prompt": "테스트 프롬프트",
            "max_tokens": 64,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "prompt_tokens" in body
        assert "completion_tokens" in body
        assert isinstance(body["prompt_tokens"], int)
        assert isinstance(body["completion_tokens"], int)
        assert body["prompt_tokens"] > 0
        assert body["completion_tokens"] > 0

    def test_generate_stream_true_returns_400(self, client_with_index):
        """stream=true로 /v1/generate 호출 시 400 에러를 반환한다."""
        payload = {
            "prompt": "테스트",
            "stream": True,
        }
        resp = client_with_index.post("/v1/generate", json=payload)

        assert resp.status_code == 400

    def test_generate_request_id_is_uuid(self, client_with_index):
        """응답의 request_id가 유효한 UUID 형식이다."""
        import uuid

        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock()()
        )

        payload = {
            "prompt": "테스트",
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        # request_id가 유효한 UUID인지 확인
        try:
            uuid.UUID(body["request_id"])
        except ValueError:
            pytest.fail(f"request_id가 유효한 UUID가 아닙니다: {body['request_id']}")


# ---------------------------------------------------------------------------
# 정상 동작 테스트: GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """GET /health 엔드포인트 테스트."""

    def test_health_basic(self, client):
        """기본 헬스체크가 'healthy' 상태를 반환한다."""
        resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "indexes" in body

    def test_health_with_rag_enabled(self, client_with_index):
        """RAG가 활성화된 상태에서 rag_enabled=true를 반환한다."""
        resp = client_with_index.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("rag_enabled") is True

    def test_health_without_rag(self, client):
        """retriever가 None인 경우 rag_enabled=false를 반환한다."""
        original = manager.retriever
        manager.retriever = None

        try:
            resp = client.get("/health")

            assert resp.status_code == 200
            body = resp.json()
            assert body.get("rag_enabled") is False
        finally:
            manager.retriever = original

    def test_health_index_stats_fields(self, client_with_index):
        """index_stats 필드에 base_dir, embedding_dim, indexes 정보가 포함된다.

        Phase 1 구현 후 /health 응답에 index_stats가 추가된다.
        미구현 시 이 테스트는 통과하되 검증을 건너뛴다.
        """
        resp = client_with_index.get("/health")

        assert resp.status_code == 200
        body = resp.json()

        # index_stats가 포함된 경우에만 검증 (Phase 1 업데이트 후)
        if "index_stats" in body:
            stats = body["index_stats"]
            assert "base_dir" in stats
            assert "embedding_dim" in stats
            assert "indexes" in stats
            assert isinstance(stats["indexes"], dict)

            # case 인덱스가 로드된 상태
            if "case" in stats["indexes"]:
                case_info = stats["indexes"]["case"]
                assert case_info["loaded"] is True
                assert case_info["doc_count"] == 20

    def test_health_response_time(self, client):
        """헬스체크 응답 시간이 500ms 이내이다."""
        start = time.monotonic()
        resp = client.get("/health")
        elapsed_ms = (time.monotonic() - start) * 1000

        assert resp.status_code == 200
        assert elapsed_ms < 500, f"헬스체크 응답 시간 {elapsed_ms:.1f}ms가 500ms를 초과"
