"""
HybridSearchEngine 단위 테스트.

Dense/Sparse/Hybrid 검색 모드별 분기와 RRF 융합 로직을 검증한다.
GPU/모델 없이 실행 가능.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 무거운 의존성 mock 등록
# ---------------------------------------------------------------------------
_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, MagicMock)
if not _faiss_is_real:
    _faiss_mock = MagicMock()
    _faiss_mock.IndexIVFFlat = type("IndexIVFFlat", (), {})
    _faiss_mock.IndexFlatIP = type("IndexFlatIP", (), {})
    sys.modules.setdefault("faiss", _faiss_mock)

sys.modules.setdefault("sentence_transformers", MagicMock())

from src.inference.hybrid_search import (
    DEFAULT_RRF_WEIGHTS,
    HybridSearchEngine,
    RRFWeightConfig,
    SearchMode,
)
from src.inference.index_manager import IndexType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_index_manager():
    mgr = MagicMock()
    mgr.indexes = {IndexType.CASE: MagicMock(ntotal=100)}
    mgr.metadata = {IndexType.CASE: []}
    mgr.search.return_value = [
        {"doc_id": "d1", "title": "사례1", "score": 0.9},
        {"doc_id": "d2", "title": "사례2", "score": 0.8},
    ]
    return mgr


@pytest.fixture
def mock_bm25():
    indexer = MagicMock()
    indexer.is_ready.return_value = True
    indexer.search.return_value = [(0, 5.0), (1, 3.0)]
    return indexer


_EMBED_DIM = 1024  # multilingual-e5-large 임베딩 차원


@pytest.fixture
def mock_embed_model():
    model = MagicMock()
    model.encode.return_value = np.random.rand(1, _EMBED_DIM).astype("float32")
    return model


@pytest.fixture
def engine(mock_index_manager, mock_bm25, mock_embed_model):
    return HybridSearchEngine(
        index_manager=mock_index_manager,
        bm25_indexers={IndexType.CASE: mock_bm25},
        embed_model=mock_embed_model,
    )


# ---------------------------------------------------------------------------
# 초기화 테스트
# ---------------------------------------------------------------------------


class TestInit:
    def test_raises_on_none_index_manager(self, mock_embed_model):
        """index_manager가 None이면 RuntimeError를 발생시킨다."""
        with pytest.raises(RuntimeError, match="None"):
            HybridSearchEngine(
                index_manager=None,
                bm25_indexers={},
                embed_model=mock_embed_model,
            )

    def test_raises_on_invalid_rrf_k(self, mock_index_manager, mock_embed_model):
        """rrf_k < 1이면 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="rrf_k"):
            HybridSearchEngine(
                index_manager=mock_index_manager,
                bm25_indexers={},
                embed_model=mock_embed_model,
                rrf_k=0,
            )

    def test_default_rrf_weights(self, engine):
        """기본 RRF 가중치가 설정된다."""
        assert engine.rrf_weights == DEFAULT_RRF_WEIGHTS

    def test_custom_rrf_weights(self, mock_index_manager, mock_embed_model):
        """커스텀 RRF 가중치를 설정할 수 있다."""
        custom = {IndexType.CASE: RRFWeightConfig(dense_weight=2.0, sparse_weight=0.5)}
        eng = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=mock_embed_model,
            rrf_weights=custom,
        )
        assert eng.rrf_weights[IndexType.CASE].dense_weight == 2.0


# ---------------------------------------------------------------------------
# Dense 검색 테스트
# ---------------------------------------------------------------------------


class TestDenseSearch:
    @pytest.mark.asyncio
    async def test_dense_search_returns_results(self, engine):
        """Dense 모드 검색 결과를 반환한다."""
        results, mode = await engine.search("도로 파손", IndexType.CASE, mode=SearchMode.DENSE)
        assert mode == SearchMode.DENSE
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_dense_search_empty_query(self, engine):
        """빈 쿼리는 빈 결과를 반환한다."""
        results, mode = await engine.search("", IndexType.CASE, mode=SearchMode.DENSE)
        assert results == []

    @pytest.mark.asyncio
    async def test_dense_search_zero_topk(self, engine):
        """top_k=0은 빈 결과를 반환한다."""
        results, mode = await engine.search(
            "테스트", IndexType.CASE, top_k=0, mode=SearchMode.DENSE
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_dense_search_no_embed_model(self, mock_index_manager):
        """embed_model이 None이면 RuntimeError를 발생시킨다."""
        eng = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=None,
        )
        with pytest.raises(RuntimeError, match="embed_model"):
            await eng.search("테스트", IndexType.CASE, mode=SearchMode.DENSE)


# ---------------------------------------------------------------------------
# Sparse 검색 테스트
# ---------------------------------------------------------------------------


class TestSparseSearch:
    @pytest.mark.asyncio
    async def test_sparse_search_no_bm25(self, mock_index_manager, mock_embed_model):
        """BM25 인덱서가 없으면 빈 결과를 반환한다."""
        eng = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=mock_embed_model,
        )
        results, mode = await eng.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        assert results == []
        assert mode == SearchMode.SPARSE

    @pytest.mark.asyncio
    async def test_sparse_search_not_ready(self, mock_index_manager, mock_embed_model):
        """BM25가 준비되지 않으면 빈 결과를 반환한다."""
        bm25 = MagicMock()
        bm25.is_ready.return_value = False
        eng = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: bm25},
            embed_model=mock_embed_model,
        )
        results, mode = await eng.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        assert results == []


# ---------------------------------------------------------------------------
# Hybrid 검색 테스트
# ---------------------------------------------------------------------------


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_fallback_to_dense_when_no_bm25(
        self, mock_index_manager, mock_embed_model
    ):
        """BM25 미사용 시 Dense로 폴백한다."""
        eng = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=mock_embed_model,
        )
        results, mode = await eng.search("테스트", IndexType.CASE, mode=SearchMode.HYBRID)
        assert mode == SearchMode.DENSE

    @pytest.mark.asyncio
    async def test_hybrid_fallback_when_bm25_not_ready(self, mock_index_manager, mock_embed_model):
        """BM25가 준비되지 않으면 Dense로 폴백한다."""
        bm25 = MagicMock()
        bm25.is_ready.return_value = False
        eng = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: bm25},
            embed_model=mock_embed_model,
        )
        results, mode = await eng.search("테스트", IndexType.CASE, mode=SearchMode.HYBRID)
        assert mode == SearchMode.DENSE

    @pytest.mark.asyncio
    async def test_hybrid_no_embed_model_raises(self, mock_index_manager):
        """Hybrid 모드에서 embed_model이 None이면 RuntimeError."""
        eng = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=None,
        )
        with pytest.raises(RuntimeError, match="embed_model"):
            await eng.search("테스트", IndexType.CASE, mode=SearchMode.HYBRID)


# ---------------------------------------------------------------------------
# RRF 테스트
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_rrf_merges_results(self, engine):
        """Dense + Sparse 결과를 RRF로 병합한다."""
        dense = [
            {"doc_id": "d1", "title": "사례1", "score": 0.9},
            {"doc_id": "d2", "title": "사례2", "score": 0.8},
        ]
        sparse = [
            {"doc_id": "d2", "title": "사례2", "score": 5.0},
            {"doc_id": "d3", "title": "사례3", "score": 3.0},
        ]
        fused = engine._reciprocal_rank_fusion(dense, sparse, IndexType.CASE)

        assert len(fused) == 3
        # 모든 결과의 score는 0~1 범위
        for item in fused:
            assert 0 <= item["score"] <= 1.0
        # 최고 점수는 1.0
        assert fused[0]["score"] == 1.0

    def test_rrf_empty_both(self, engine):
        """양쪽 모두 빈 결과면 빈 리스트를 반환한다."""
        fused = engine._reciprocal_rank_fusion([], [], IndexType.CASE)
        assert fused == []

    def test_rrf_dense_only(self, engine):
        """Dense 결과만 있을 때."""
        dense = [{"doc_id": "d1", "title": "사례1", "score": 0.9}]
        fused = engine._reciprocal_rank_fusion(dense, [], IndexType.CASE)
        assert len(fused) == 1
        assert fused[0]["score"] == 1.0

    def test_rrf_sparse_only(self, engine):
        """Sparse 결과만 있을 때."""
        sparse = [{"doc_id": "d1", "title": "사례1", "score": 5.0}]
        fused = engine._reciprocal_rank_fusion([], sparse, IndexType.CASE)
        assert len(fused) == 1
        assert fused[0]["score"] == 1.0

    def test_rrf_skips_empty_doc_id(self, engine):
        """doc_id가 비어있는 결과는 제외한다."""
        dense = [{"doc_id": "", "title": "무명", "score": 0.5}]
        sparse = [{"doc_id": "d1", "title": "사례1", "score": 3.0}]
        fused = engine._reciprocal_rank_fusion(dense, sparse, IndexType.CASE)
        assert len(fused) == 1
        assert fused[0]["doc_id"] == "d1"

    def test_rrf_duplicate_doc_combined(self, engine):
        """동일 doc_id는 점수가 합산된다."""
        dense = [{"doc_id": "d1", "title": "사례1", "score": 0.9}]
        sparse = [{"doc_id": "d1", "title": "사례1", "score": 5.0}]
        fused = engine._reciprocal_rank_fusion(dense, sparse, IndexType.CASE)
        assert len(fused) == 1
        # Dense + Sparse 양쪽에서 점수를 받으므로 1.0이어야 함
        assert fused[0]["score"] == 1.0


# ---------------------------------------------------------------------------
# _embed_query 테스트
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    def test_embed_query_adds_prefix(self, engine):
        """쿼리에 'query: ' 접두사를 추가한다."""
        engine._embed_query("도로 파손")
        engine.embed_model.encode.assert_called_once()
        call_args = engine.embed_model.encode.call_args[0][0]
        assert call_args == ["query: 도로 파손"]

    def test_embed_query_returns_vector(self, engine):
        """1차원 벡터를 반환한다."""
        engine.embed_model.encode.return_value = np.random.rand(1, _EMBED_DIM).astype("float32")
        result = engine._embed_query("테스트")
        assert result.ndim == 1

    def test_embed_query_cache_hit_calls_encode_once(self, engine):
        """동일 쿼리를 두 번 호출해도 encode는 1회만 호출된다."""
        engine.embed_model.encode.return_value = np.random.rand(1, _EMBED_DIM).astype("float32")
        engine._embed_query("캐시 테스트")
        engine._embed_query("캐시 테스트")
        engine.embed_model.encode.assert_called_once()

    def test_embed_query_cache_hit_returns_same_vector(self, engine):
        """캐시 히트 시 동일한 벡터 객체를 반환한다."""
        engine.embed_model.encode.return_value = np.random.rand(1, _EMBED_DIM).astype("float32")
        first = engine._embed_query("동일 벡터 확인")
        second = engine._embed_query("동일 벡터 확인")
        np.testing.assert_array_equal(first, second)

    def test_embed_query_lru_eviction_removes_least_recently_used(self, engine):
        """캐시가 64개를 초과하면 가장 오래 사용되지 않은 항목이 제거된다."""
        engine.embed_model.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts), _EMBED_DIM
        ).astype("float32")

        # q0 ~ q63 을 채워서 캐시를 꽉 채운다 (64개)
        for i in range(64):
            engine._embed_query(f"query_{i}")

        # q0 에 접근하여 LRU 순서를 최신으로 갱신
        engine._embed_query("query_0")

        # q64 를 추가 → 캐시 크기 초과 → LRU인 q1이 제거되어야 한다
        engine._embed_query("query_64")

        assert "query_0" in engine._embed_cache, "최근 접근한 query_0은 캐시에 남아야 한다"
        assert "query_1" not in engine._embed_cache, "가장 오래된 query_1은 제거되어야 한다"
        assert "query_64" in engine._embed_cache, "새로 추가된 query_64는 캐시에 있어야 한다"
        assert len(engine._embed_cache) == 64

    def test_embed_query_cache_max_size_is_64(self, engine):
        """캐시는 최대 64개를 초과하지 않는다."""
        engine.embed_model.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts), _EMBED_DIM
        ).astype("float32")
        for i in range(80):
            engine._embed_query(f"q_{i}")
        assert len(engine._embed_cache) == 64


# ---------------------------------------------------------------------------
# is_ready 테스트
# ---------------------------------------------------------------------------


class TestIsReady:
    def test_ready_when_index_loaded(self, engine):
        """인덱스가 로드되고 문서가 있으면 True."""
        engine.index_manager.indexes = {IndexType.CASE: MagicMock(ntotal=100)}
        assert engine.is_ready(IndexType.CASE) is True

    def test_not_ready_when_no_index(self, engine):
        """인덱스가 없으면 False."""
        engine.index_manager.indexes = {}
        assert engine.is_ready(IndexType.CASE) is False

    def test_not_ready_when_empty_index(self, engine):
        """인덱스가 비어있으면 False."""
        engine.index_manager.indexes = {IndexType.CASE: MagicMock(ntotal=0)}
        assert engine.is_ready(IndexType.CASE) is False
