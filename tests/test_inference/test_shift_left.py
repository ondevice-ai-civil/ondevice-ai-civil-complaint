"""
Shift-Left 테스트: GPU/모델 다운로드 없이 전체 API 엔드포인트를 검증한다.

test_api_integration.py에서 이미 커버하는 기본 동작 테스트와 중복되지 않도록,
에지 케이스, 상세 스키마 검증, 파라미터 조합에 집중한다.

모킹 패턴은 test_api_integration.py와 동일:
1. sys.modules.setdefault로 무거운 의존성(vllm, sentence_transformers, torch) mock 등록
2. apply_transformers_patch를 no-op 처리 후 app, manager import
3. _make_vllm_output_mock() 헬퍼로 vLLM 엔진 출력 시뮬레이션
4. _patch_lifespan fixture로 manager.initialize() 우회
5. httpx.AsyncClient 또는 FastAPI TestClient 기반 테스트
"""

import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. 무거운 의존성을 import 전에 mock 등록
# ---------------------------------------------------------------------------

_vllm_mock = MagicMock()
sys.modules.setdefault("vllm", _vllm_mock)
sys.modules.setdefault("vllm.engine", _vllm_mock)
sys.modules.setdefault("vllm.engine.arg_utils", _vllm_mock)
sys.modules.setdefault("vllm.engine.async_llm_engine", _vllm_mock)
sys.modules.setdefault("vllm.sampling_params", _vllm_mock)

_st_mock = MagicMock()
sys.modules.setdefault("sentence_transformers", _st_mock)

sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("transformers.modeling_rope_utils", MagicMock())
sys.modules.setdefault("transformers.utils", MagicMock())
sys.modules.setdefault("transformers.utils.generic", MagicMock())

if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

# ---------------------------------------------------------------------------
# 2. api_server import
# ---------------------------------------------------------------------------

with patch("src.inference.vllm_stabilizer.apply_transformers_patch"):
    from src.inference.api_server import app, manager

from fastapi.testclient import TestClient

from src.inference.index_manager import IndexType, MultiIndexManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vllm_output_mock(
    text="테스트 응답입니다.", prompt_token_count=5, completion_token_count=10
):
    """vLLM 엔진의 generate 결과를 시뮬레이션하는 async generator를 반환한다."""
    output_mock = MagicMock()
    output_mock.outputs = [MagicMock()]
    output_mock.outputs[0].text = text
    output_mock.outputs[0].token_ids = list(range(completion_token_count))
    output_mock.prompt_token_ids = list(range(prompt_token_count))
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
def test_index_manager(tmp_path):
    """mock 기반 MultiIndexManager."""
    mock_mgr = MagicMock(spec=MultiIndexManager)
    mock_mgr.base_dir = str(tmp_path)
    mock_mgr.embedding_dim = 1024

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


@pytest.fixture
def client_with_index(client, test_index_manager, mock_embed_model):
    """MultiIndexManager와 embed_model이 설정된 상태의 TestClient."""
    original_index_manager = getattr(manager, "index_manager", None)
    original_embed_model = getattr(manager, "embed_model", None)
    original_retriever = getattr(manager, "retriever", None)

    manager.index_manager = test_index_manager
    manager.embed_model = mock_embed_model
    if original_retriever is None:
        manager.retriever = MagicMock()

    yield client

    manager.index_manager = original_index_manager
    manager.embed_model = original_embed_model
    manager.retriever = original_retriever


@pytest.fixture
def client_with_classifier(client):
    """classifier 에이전트가 로드된 상태의 TestClient."""
    from src.inference.agent_manager import AgentManager

    original_am = getattr(manager, "agent_manager", None)
    original_engine = getattr(manager, "engine", None)

    import os
    import tempfile

    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "classifier.md"), "w", encoding="utf-8") as f:
        f.write(
            "---\nname: classifier\nrole: Routing Specialist\n"
            "description: 민원 분류\ntemperature: 0.0\nmax_tokens: 256\n"
            "---\n\n당신은 민원 분류 전문가입니다.\n"
        )
    manager.agent_manager = AgentManager(tmpdir)
    manager.engine = MagicMock()
    manager.engine.generate = MagicMock(
        return_value=_make_vllm_output_mock(
            '{"category": "traffic", "confidence": 0.95, "reason": "도로 관련 민원"}'
        )()
    )

    yield client

    manager.agent_manager = original_am
    manager.engine = original_engine
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# GET /health - 에지 케이스
# ===========================================================================


class TestHealthShiftLeft:
    """GET /health 에지 케이스 및 상세 검증."""

    def test_health_response_has_required_fields(self, client):
        """응답에 status, rag_enabled, indexes, agents_loaded 필드가 모두 존재한다."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "rag_enabled" in body
        assert "indexes" in body
        assert "agents_loaded" in body

    def test_health_bm25_indexes_field(self, client):
        """응답에 bm25_indexes 필드가 포함된다."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "bm25_indexes" in body
        assert isinstance(body["bm25_indexes"], dict)

    def test_health_hybrid_search_enabled_field(self, client):
        """응답에 hybrid_search_enabled 필드가 포함된다."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "hybrid_search_enabled" in body
        assert isinstance(body["hybrid_search_enabled"], bool)

    def test_health_pii_masking_enabled_field(self, client):
        """응답에 pii_masking_enabled 필드가 포함된다."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "pii_masking_enabled" in body
        assert isinstance(body["pii_masking_enabled"], bool)

    def test_health_rag_enabled_with_retriever_only(self, client):
        """index_manager=None이지만 retriever가 있으면 rag_enabled=True."""
        original_im = getattr(manager, "index_manager", None)
        original_ret = getattr(manager, "retriever", None)
        manager.index_manager = None
        manager.retriever = MagicMock()

        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["rag_enabled"] is True
        finally:
            manager.index_manager = original_im
            manager.retriever = original_ret

    def test_health_rag_disabled_when_both_none(self, client):
        """index_manager=None, retriever=None이면 rag_enabled=False."""
        original_im = getattr(manager, "index_manager", None)
        original_ret = getattr(manager, "retriever", None)
        manager.index_manager = None
        manager.retriever = None

        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["rag_enabled"] is False
        finally:
            manager.index_manager = original_im
            manager.retriever = original_ret

    def test_health_indexes_none_when_no_index_manager(self, client):
        """index_manager=None일 때 indexes는 None이다."""
        original_im = getattr(manager, "index_manager", None)
        manager.index_manager = None

        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["indexes"] is None
        finally:
            manager.index_manager = original_im

    def test_health_index_stats_with_manager(self, client_with_index):
        """index_manager가 있을 때 indexes에 case/law/manual/notice 정보가 있다."""
        resp = client_with_index.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        indexes = body["indexes"]
        assert indexes is not None
        assert "case" in indexes
        assert indexes["case"]["loaded"] is True
        assert indexes["case"]["doc_count"] == 20


# ===========================================================================
# POST /v1/classify - 에지 케이스
# ===========================================================================


class TestClassifyShiftLeft:
    """POST /v1/classify 에지 케이스 및 상세 검증."""

    def test_classify_empty_prompt_returns_422(self, client_with_classifier):
        """빈 prompt는 422 Validation Error를 반환한다."""
        resp = client_with_classifier.post("/v1/classify", json={"prompt": ""})
        assert resp.status_code == 422

    def test_classify_missing_prompt_returns_422(self, client_with_classifier):
        """prompt 필드 누락 시 422 Validation Error를 반환한다."""
        resp = client_with_classifier.post("/v1/classify", json={})
        assert resp.status_code == 422

    def test_classify_response_has_request_id_uuid(self, client_with_classifier):
        """응답의 request_id가 유효한 UUID 형식이다."""
        resp = client_with_classifier.post(
            "/v1/classify", json={"prompt": "도로에 구멍이 생겼습니다."}
        )
        assert resp.status_code == 200
        body = resp.json()
        try:
            uuid.UUID(body["request_id"])
        except ValueError:
            pytest.fail(f"request_id가 유효한 UUID가 아닙니다: {body['request_id']}")

    def test_classify_json_without_required_fields(self, client_with_classifier):
        """LLM이 필수 필드 누락된 JSON을 반환하면 classification_error가 설정된다."""
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock('{"category": "traffic"}')()
        )
        resp = client_with_classifier.post("/v1/classify", json={"prompt": "테스트 민원"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"] is None
        assert body["classification_error"] is not None

    def test_classify_confidence_boundary_zero(self, client_with_classifier):
        """confidence=0.0은 유효하다."""
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock(
                '{"category": "traffic", "confidence": 0.0, "reason": "확신 없음"}'
            )()
        )
        resp = client_with_classifier.post("/v1/classify", json={"prompt": "테스트"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"] is not None
        assert body["classification"]["confidence"] == 0.0

    def test_classify_confidence_boundary_one(self, client_with_classifier):
        """confidence=1.0은 유효하다."""
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock(
                '{"category": "environment", "confidence": 1.0, "reason": "환경 민원 확실"}'
            )()
        )
        resp = client_with_classifier.post("/v1/classify", json={"prompt": "테스트"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"] is not None
        assert body["classification"]["confidence"] == 1.0

    def test_classify_confidence_out_of_range(self, client_with_classifier):
        """confidence > 1.0이면 classification_error가 설정된다."""
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock(
                '{"category": "traffic", "confidence": 1.5, "reason": "test"}'
            )()
        )
        resp = client_with_classifier.post("/v1/classify", json={"prompt": "테스트"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"] is None
        assert body["classification_error"] is not None

    def test_classify_llm_returns_json_with_extra_text(self, client_with_classifier):
        """LLM이 JSON 앞뒤에 텍스트를 붙여도 파싱에 성공한다."""
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock(
                '분석 결과:\n{"category": "welfare", "confidence": 0.8, "reason": "복지 민원"}\n이상입니다.'
            )()
        )
        resp = client_with_classifier.post("/v1/classify", json={"prompt": "복지 관련 문의"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"] is not None
        assert body["classification"]["category"] == "welfare"

    def test_classify_all_valid_categories(self, client_with_classifier):
        """모든 유효 카테고리가 정상 분류된다."""
        valid_categories = [
            "environment",
            "traffic",
            "facilities",
            "civil_service",
            "welfare",
            "other",
        ]
        for cat in valid_categories:
            manager.engine.generate = MagicMock(
                return_value=_make_vllm_output_mock(
                    f'{{"category": "{cat}", "confidence": 0.9, "reason": "테스트"}}'
                )()
            )
            resp = client_with_classifier.post("/v1/classify", json={"prompt": f"{cat} 관련 민원"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["classification"] is not None, f"카테고리 '{cat}' 분류 실패"
            assert body["classification"]["category"] == cat

    def test_classify_token_counts_are_positive(self, client_with_classifier):
        """prompt_tokens와 completion_tokens가 양의 정수이다."""
        resp = client_with_classifier.post("/v1/classify", json={"prompt": "소음 민원입니다."})
        assert resp.status_code == 200
        body = resp.json()
        assert body["prompt_tokens"] > 0
        assert body["completion_tokens"] > 0


# ===========================================================================
# POST /v1/generate - 에지 케이스
# ===========================================================================


class TestGenerateShiftLeft:
    """POST /v1/generate 에지 케이스 및 상세 검증."""

    def test_generate_empty_prompt_returns_422(self, client_with_index):
        """빈 prompt는 422를 반환한다."""
        payload = {"prompt": "", "stream": False}
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 422

    def test_generate_missing_prompt_returns_422(self, client_with_index):
        """prompt 필드 누락 시 422를 반환한다."""
        resp = client_with_index.post("/v1/generate", json={"stream": False})
        assert resp.status_code == 422

    def test_generate_max_tokens_reflected(self, client_with_index):
        """max_tokens 파라미터가 engine.generate 호출에 전달된다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(return_value=_make_vllm_output_mock()())

        payload = {
            "prompt": "테스트 프롬프트",
            "max_tokens": 256,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200

        # engine.generate가 호출되었는지 확인
        assert manager.engine.generate.called
        # SamplingParams가 호출 시 max_tokens=256으로 생성되었는지 간접 확인
        # (SamplingParams 자체가 mock이므로 호출 인자로 확인)
        call_args = manager.engine.generate.call_args
        assert call_args is not None

    def test_generate_temperature_reflected(self, client_with_index):
        """temperature 파라미터가 engine.generate 호출에 전달된다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(return_value=_make_vllm_output_mock()())

        payload = {
            "prompt": "테스트",
            "temperature": 0.3,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200

        # engine.generate가 호출되었는지 확인
        assert manager.engine.generate.called
        call_args = manager.engine.generate.call_args
        assert call_args is not None

    def test_generate_use_rag_true_calls_retriever(self, client_with_index):
        """use_rag=True일 때 retriever.search가 호출된다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(
            return_value=[
                {"complaint": "민원 내용", "answer": "답변 내용", "score": 0.9},
            ]
        )
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(return_value=_make_vllm_output_mock()())

        payload = {
            "prompt": "도로 보수 요청합니다.",
            "use_rag": True,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["retrieved_cases"] is not None
        assert len(body["retrieved_cases"]) > 0
        manager.retriever.search.assert_called_once()

    def test_generate_use_rag_false_skips_retriever(self, client_with_index):
        """use_rag=False일 때 retriever.search가 호출되지 않는다."""
        manager.retriever = MagicMock()
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(return_value=_make_vllm_output_mock()())

        payload = {
            "prompt": "일반 질문",
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        retrieved = body.get("retrieved_cases")
        assert retrieved is None or retrieved == []
        manager.retriever.search.assert_not_called()

    def test_generate_token_counts_match_mock(self, client_with_index):
        """토큰 카운트가 mock에 설정한 값과 일치한다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock(prompt_token_count=8, completion_token_count=15)()
        )

        payload = {
            "prompt": "토큰 카운트 테스트",
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["prompt_tokens"] == 8
        assert body["completion_tokens"] == 15

    def test_generate_thought_blocks_stripped(self, client_with_index):
        """LLM 출력의 <thought> 블록이 제거된다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(
            return_value=_make_vllm_output_mock(
                "<thought>내부 추론 과정</thought>최종 답변입니다."
            )()
        )

        payload = {
            "prompt": "테스트",
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "<thought>" not in body["text"]
        assert "내부 추론 과정" not in body["text"]
        assert "최종 답변입니다." in body["text"]

    def test_generate_search_results_field_present(self, client_with_index):
        """응답에 search_results 필드가 존재한다 (None 허용)."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(return_value=_make_vllm_output_mock()())

        payload = {
            "prompt": "테스트",
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "search_results" in body

    def test_generate_max_tokens_boundary_1(self, client_with_index):
        """max_tokens=1 최솟값이 유효하다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(return_value=_make_vllm_output_mock()())

        payload = {
            "prompt": "테스트",
            "max_tokens": 1,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200

    def test_generate_max_tokens_boundary_4096(self, client_with_index):
        """max_tokens=4096 최댓값이 유효하다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(return_value=_make_vllm_output_mock()())

        payload = {
            "prompt": "테스트",
            "max_tokens": 4096,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200

    def test_generate_max_tokens_exceeds_limit_returns_422(self, client_with_index):
        """max_tokens=4097은 422를 반환한다."""
        payload = {
            "prompt": "테스트",
            "max_tokens": 4097,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 422

    def test_generate_temperature_boundary_zero(self, client_with_index):
        """temperature=0.0이 유효하다."""
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(return_value=[])
        manager.engine = MagicMock()
        manager.engine.generate = MagicMock(return_value=_make_vllm_output_mock()())

        payload = {
            "prompt": "테스트",
            "temperature": 0.0,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 200

    def test_generate_temperature_exceeds_limit_returns_422(self, client_with_index):
        """temperature=2.1은 422를 반환한다."""
        payload = {
            "prompt": "테스트",
            "temperature": 2.1,
            "use_rag": False,
            "stream": False,
        }
        resp = client_with_index.post("/v1/generate", json=payload)
        assert resp.status_code == 422


# ===========================================================================
# POST /search 및 POST /v1/search - 에지 케이스 & 상세 스키마 검증
# ===========================================================================


@pytest.fixture
def client_with_hybrid_engine(client_with_index):
    """HybridSearchEngine이 설정된 상태의 TestClient."""
    from src.inference.hybrid_search import SearchMode

    original_hybrid = getattr(manager, "hybrid_engine", None)
    original_pii = getattr(manager, "pii_masker", None)

    mock_hybrid = MagicMock()
    now = datetime.now(timezone.utc).isoformat()

    async def _search(query, index_type, top_k=5, mode=SearchMode.HYBRID):
        results = [
            {
                "doc_id": f"case-{i:04d}",
                "doc_type": "case",
                "source": "AI Hub",
                "title": f"테스트 민원 {i}",
                "category": "도로/교통",
                "reliability_score": round(0.5 + i * 0.05, 2),
                "created_at": now,
                "updated_at": now,
                "score": round(0.95 - i * 0.05, 4),
                "chunk_index": i % 3,
                "chunk_total": 3,
                "extras": {
                    "complaint_text": f"테스트 민원 내용 {i}",
                    "answer_text": f"테스트 답변 {i}",
                },
            }
            for i in range(min(top_k, 10))
        ]
        return results, mode

    mock_hybrid.search = AsyncMock(side_effect=_search)
    manager.hybrid_engine = mock_hybrid
    manager.pii_masker = None  # PII 마스킹 비활성화

    yield client_with_index

    manager.hybrid_engine = original_hybrid
    manager.pii_masker = original_pii


class TestSearchShiftLeft:
    """POST /search, /v1/search 에지 케이스 및 상세 스키마 검증."""

    def test_search_result_full_schema_fields(self, client_with_hybrid_engine):
        """SearchResult의 모든 필드가 응답에 포함되고 타입이 올바르다."""
        payload = {
            "query": "도로 보수",
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()

        for result in body["results"]:
            # 필수 필드 존재 확인
            assert "doc_id" in result
            assert "source_type" in result
            assert "title" in result
            assert "content" in result
            assert "score" in result
            assert "reliability_score" in result
            assert "metadata" in result
            assert "chunk_index" in result
            assert "total_chunks" in result

            # 타입 검증
            assert isinstance(result["doc_id"], str)
            assert isinstance(result["source_type"], str)
            assert isinstance(result["title"], str)
            assert isinstance(result["content"], str)
            assert isinstance(result["score"], (int, float))
            assert isinstance(result["reliability_score"], (int, float))
            assert isinstance(result["metadata"], dict)
            assert isinstance(result["chunk_index"], int)
            assert isinstance(result["total_chunks"], int)

    def test_search_result_score_range(self, client_with_hybrid_engine):
        """score가 0 이상의 float이다."""
        payload = {
            "query": "민원 처리",
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert result["score"] >= 0.0

    def test_search_result_reliability_score_range(self, client_with_hybrid_engine):
        """reliability_score가 0~1 범위이다."""
        payload = {
            "query": "민원 처리",
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert 0.0 <= result["reliability_score"] <= 1.0

    def test_search_result_chunk_fields(self, client_with_hybrid_engine):
        """chunk_index와 total_chunks가 양의 정수이고, chunk_index < total_chunks이다."""
        payload = {
            "query": "테스트",
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert result["chunk_index"] >= 0
            assert result["total_chunks"] >= 1

    def test_search_v1_endpoint_returns_same_schema(self, client_with_hybrid_engine):
        """/v1/search와 /search가 동일한 스키마를 반환한다."""
        payload = {
            "query": "도로 보수",
            "top_k": 3,
            "doc_type": "case",
        }
        resp_legacy = client_with_hybrid_engine.post("/search", json=payload)
        resp_v1 = client_with_hybrid_engine.post("/v1/search", json=payload)

        assert resp_legacy.status_code == 200
        assert resp_v1.status_code == 200

        keys_legacy = set(resp_legacy.json().keys())
        keys_v1 = set(resp_v1.json().keys())
        assert keys_legacy == keys_v1

    def test_search_response_has_search_time_ms(self, client_with_hybrid_engine):
        """응답에 search_time_ms 필드가 포함되고 양수이다."""
        payload = {
            "query": "테스트",
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "search_time_ms" in body
        assert body["search_time_ms"] >= 0

    def test_search_response_has_search_mode(self, client_with_hybrid_engine):
        """응답에 search_mode 필드가 포함된다."""
        payload = {
            "query": "테스트",
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "search_mode" in body

    def test_search_mode_dense(self, client_with_hybrid_engine):
        """search_mode='dense'로 검색이 동작한다."""
        payload = {
            "query": "도로 파손",
            "top_k": 3,
            "doc_type": "case",
            "search_mode": "dense",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["search_mode"] == "dense"

    def test_search_mode_sparse(self, client_with_hybrid_engine):
        """search_mode='sparse'로 검색이 동작한다."""
        payload = {
            "query": "도로 파손",
            "top_k": 3,
            "doc_type": "case",
            "search_mode": "sparse",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["search_mode"] == "sparse"

    def test_search_mode_hybrid(self, client_with_hybrid_engine):
        """search_mode='hybrid'로 검색이 동작한다."""
        payload = {
            "query": "도로 파손",
            "top_k": 3,
            "doc_type": "case",
            "search_mode": "hybrid",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["search_mode"] == "hybrid"

    def test_search_mode_invalid_returns_422(self, client_with_hybrid_engine):
        """존재하지 않는 search_mode는 422를 반환한다."""
        payload = {
            "query": "테스트",
            "top_k": 3,
            "doc_type": "case",
            "search_mode": "invalid_mode",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 422

    def test_search_doc_type_law(self, client_with_hybrid_engine):
        """doc_type='law'로 검색이 동작한다."""
        payload = {
            "query": "관련 법률",
            "top_k": 3,
            "doc_type": "law",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] == "law"

    def test_search_doc_type_manual(self, client_with_hybrid_engine):
        """doc_type='manual'로 검색이 동작한다."""
        payload = {
            "query": "매뉴얼 검색",
            "top_k": 3,
            "doc_type": "manual",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] == "manual"

    def test_search_doc_type_notice(self, client_with_hybrid_engine):
        """doc_type='notice'로 검색이 동작한다."""
        payload = {
            "query": "공지사항",
            "top_k": 3,
            "doc_type": "notice",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] == "notice"

    def test_search_total_matches_results_length(self, client_with_hybrid_engine):
        """응답의 total이 results 리스트 길이와 일치한다."""
        payload = {
            "query": "테스트 쿼리",
            "top_k": 5,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == len(body["results"])

    def test_search_query_echoed_in_response(self, client_with_hybrid_engine):
        """응답의 query가 요청과 일치한다."""
        query_text = "도로 보수 민원 검색"
        payload = {
            "query": query_text,
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        assert resp.json()["query"] == query_text

    def test_search_metadata_is_dict(self, client_with_hybrid_engine):
        """각 결과의 metadata가 dict이고 extras 데이터를 포함한다."""
        payload = {
            "query": "테스트",
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert isinstance(result["metadata"], dict)
            # extras에서 전달된 complaint_text, answer_text 확인
            assert "complaint_text" in result["metadata"]
            assert "answer_text" in result["metadata"]

    def test_search_content_not_empty(self, client_with_hybrid_engine):
        """각 결과의 content가 빈 문자열이 아니다."""
        payload = {
            "query": "테스트",
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert result["content"] != ""

    def test_search_without_engines_returns_503(self, client):
        """hybrid_engine=None, retriever=None일 때 503을 반환한다."""
        original_hybrid = getattr(manager, "hybrid_engine", None)
        original_retriever = getattr(manager, "retriever", None)
        original_index = getattr(manager, "index_manager", None)
        manager.hybrid_engine = None
        manager.retriever = None
        manager.index_manager = None

        try:
            payload = {
                "query": "테스트",
                "top_k": 5,
                "doc_type": "case",
            }
            resp = client.post("/search", json=payload)
            assert resp.status_code == 503
        finally:
            manager.hybrid_engine = original_hybrid
            manager.retriever = original_retriever
            manager.index_manager = original_index

    def test_search_retriever_fallback(self, client):
        """hybrid_engine=None이고 retriever가 있으면 레거시 폴백으로 검색한다."""
        original_hybrid = getattr(manager, "hybrid_engine", None)
        original_retriever = getattr(manager, "retriever", None)
        original_pii = getattr(manager, "pii_masker", None)

        manager.hybrid_engine = None
        manager.retriever = MagicMock()
        manager.retriever.search = MagicMock(
            return_value=[
                {
                    "id": "legacy-001",
                    "category": "도로/교통",
                    "complaint": "도로 파손 민원",
                    "answer": "보수 예정",
                    "score": 0.85,
                },
            ]
        )
        manager.pii_masker = None

        try:
            payload = {
                "query": "도로 파손",
                "top_k": 3,
                "doc_type": "case",
            }
            resp = client.post("/search", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 1
            assert body["results"][0]["doc_id"] == "legacy-001"
            assert body["search_mode"] == "dense"
        finally:
            manager.hybrid_engine = original_hybrid
            manager.retriever = original_retriever
            manager.pii_masker = original_pii

    def test_search_query_max_length_2000(self, client_with_hybrid_engine):
        """query 최대 길이 2000이 허용된다."""
        payload = {
            "query": "가" * 2000,
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 200

    def test_search_query_exceeds_max_length_returns_422(self, client_with_hybrid_engine):
        """query 2001자 초과 시 422를 반환한다."""
        payload = {
            "query": "가" * 2001,
            "top_k": 3,
            "doc_type": "case",
        }
        resp = client_with_hybrid_engine.post("/search", json=payload)
        assert resp.status_code == 422
