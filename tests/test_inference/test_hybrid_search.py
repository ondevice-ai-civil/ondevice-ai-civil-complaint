"""
Unit tests for HybridSearchEngine (Issue #154).

Tests cover:
- SearchMode enum values
- RRFWeightConfig defaults and customization
- Reciprocal Rank Fusion scoring logic
- Dense-only search mode
- Sparse-only search mode
- Hybrid search mode (Dense + Sparse RRF fusion)
- is_ready() index availability check
- Edge cases (empty results, missing indexer, fallback behavior)

All heavy dependencies (vllm, SentenceTransformer, faiss) are mocked
for CI compatibility.
"""

import sys
import unittest.mock as mock

# 무거운 모듈 mock (import 전에)
sys.modules.setdefault("vllm", mock.MagicMock())
sys.modules.setdefault("vllm.engine.arg_utils", mock.MagicMock())
sys.modules.setdefault("vllm.engine.async_llm_engine", mock.MagicMock())
sys.modules.setdefault("vllm.sampling_params", mock.MagicMock())
sys.modules.setdefault("sentence_transformers", mock.MagicMock())

# faiss mock
_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, mock.MagicMock)
if not _faiss_is_real:
    _faiss_mock = mock.MagicMock()
    _faiss_mock.IndexIVFFlat = type("IndexIVFFlat", (), {})
    _faiss_mock.IndexFlatIP = type("IndexFlatIP", (), {})
    sys.modules["faiss"] = _faiss_mock

import numpy as np
import pytest

from src.inference.hybrid_search import (
    DEFAULT_RRF_WEIGHTS,
    HybridSearchEngine,
    RRFWeightConfig,
    SearchMode,
)
from src.inference.index_manager import DocumentMetadata, IndexType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metadata() -> list[DocumentMetadata]:
    """테스트용 DocumentMetadata 리스트 생성 (CASE 타입 10개)."""
    return [
        DocumentMetadata(
            doc_id=f"case-{i:03d}",
            doc_type=IndexType.CASE.value,
            source="AI Hub",
            title=f"민원 사례 {i}",
            category="환경/위생",
            reliability_score=0.8,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
            extras={"complaint_text": f"민원 내용 {i}", "answer_text": f"답변 {i}"},
        )
        for i in range(10)
    ]


@pytest.fixture
def mock_index_manager(sample_metadata: list[DocumentMetadata]) -> mock.MagicMock:
    """MultiIndexManager mock."""
    manager = mock.MagicMock()
    manager.metadata = {IndexType.CASE: sample_metadata}
    manager.indexes = {IndexType.CASE: mock.MagicMock(ntotal=10)}

    # search()가 호출되면 상위 5개 문서 반환
    def fake_search(index_type: IndexType, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        meta_list = manager.metadata.get(index_type, [])
        results = []
        for i in range(min(top_k, len(meta_list))):
            d = meta_list[i].to_dict()
            d["score"] = 1.0 - i * 0.1  # 1.0, 0.9, 0.8, ...
            results.append(d)
        return results

    manager.search.side_effect = fake_search
    return manager


@pytest.fixture
def mock_bm25_indexer() -> mock.MagicMock:
    """BM25Indexer mock.

    dense와 다른 순서로 반환하여 RRF 융합 테스트를 용이하게 한다.
    """
    indexer = mock.MagicMock()
    indexer.is_ready.return_value = True
    indexer.doc_count = 10
    # search() 결과: [(corpus_index, bm25_score), ...]
    indexer.search.return_value = [
        (4, 5.2),
        (2, 4.8),
        (0, 3.5),
        (7, 2.1),
        (1, 1.5),
    ]
    return indexer


@pytest.fixture
def mock_embed_model() -> mock.MagicMock:
    """SentenceTransformer mock."""
    model = mock.MagicMock()
    model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
    return model


@pytest.fixture
def engine(
    mock_index_manager: mock.MagicMock,
    mock_bm25_indexer: mock.MagicMock,
    mock_embed_model: mock.MagicMock,
) -> HybridSearchEngine:
    """HybridSearchEngine 인스턴스."""
    return HybridSearchEngine(
        index_manager=mock_index_manager,
        bm25_indexers={IndexType.CASE: mock_bm25_indexer},
        embed_model=mock_embed_model,
    )


# ---------------------------------------------------------------------------
# TestSearchMode
# ---------------------------------------------------------------------------


class TestSearchMode:
    def test_enum_values(self) -> None:
        assert SearchMode.DENSE == "dense"
        assert SearchMode.SPARSE == "sparse"
        assert SearchMode.HYBRID == "hybrid"

    def test_enum_from_string(self) -> None:
        assert SearchMode("dense") == SearchMode.DENSE
        assert SearchMode("sparse") == SearchMode.SPARSE
        assert SearchMode("hybrid") == SearchMode.HYBRID

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SearchMode("invalid")


# ---------------------------------------------------------------------------
# TestRRFWeightConfig
# ---------------------------------------------------------------------------


class TestRRFWeightConfig:
    def test_default_weights(self) -> None:
        config = RRFWeightConfig()
        assert config.dense_weight == 1.0
        assert config.sparse_weight == 0.7

    def test_custom_weights(self) -> None:
        config = RRFWeightConfig(dense_weight=0.5, sparse_weight=1.5)
        assert config.dense_weight == 0.5
        assert config.sparse_weight == 1.5

    def test_default_rrf_weights_keys(self) -> None:
        assert IndexType.CASE in DEFAULT_RRF_WEIGHTS
        assert IndexType.LAW in DEFAULT_RRF_WEIGHTS
        assert IndexType.MANUAL in DEFAULT_RRF_WEIGHTS
        assert IndexType.NOTICE in DEFAULT_RRF_WEIGHTS

    def test_default_rrf_weights_law_sparse(self) -> None:
        assert DEFAULT_RRF_WEIGHTS[IndexType.LAW].sparse_weight == 1.2

    def test_default_rrf_weights_case_values(self) -> None:
        case_config = DEFAULT_RRF_WEIGHTS[IndexType.CASE]
        assert case_config.dense_weight == 1.0
        assert case_config.sparse_weight == 0.7


# ---------------------------------------------------------------------------
# TestReciprocalRankFusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_single_dense_list_preserves_order(self, engine: HybridSearchEngine) -> None:
        """dense만 결과가 있고 sparse가 비어있을 때 순서가 유지되는지 확인."""
        dense_results = [
            {"doc_id": "case-000", "score": 1.0, "title": "A"},
            {"doc_id": "case-001", "score": 0.9, "title": "B"},
            {"doc_id": "case-002", "score": 0.8, "title": "C"},
        ]
        sparse_results: list[dict] = []

        fused = engine._reciprocal_rank_fusion(dense_results, sparse_results, IndexType.CASE)
        assert len(fused) == 3
        assert fused[0]["doc_id"] == "case-000"
        assert fused[1]["doc_id"] == "case-001"
        assert fused[2]["doc_id"] == "case-002"

    def test_disjoint_docs_merge(self, engine: HybridSearchEngine) -> None:
        """dense와 sparse가 완전히 다른 문서를 반환할 때 모두 병합되는지 확인."""
        dense_results = [
            {"doc_id": "case-000", "score": 1.0},
            {"doc_id": "case-001", "score": 0.9},
        ]
        sparse_results = [
            {"doc_id": "case-002", "score": 5.0},
            {"doc_id": "case-003", "score": 4.0},
        ]

        fused = engine._reciprocal_rank_fusion(dense_results, sparse_results, IndexType.CASE)
        doc_ids = {r["doc_id"] for r in fused}
        assert doc_ids == {"case-000", "case-001", "case-002", "case-003"}

    def test_overlapping_docs_boost(self, engine: HybridSearchEngine) -> None:
        """양쪽에 모두 등장하는 문서가 점수 합산으로 상위에 오는지 확인."""
        # case-002는 dense 3위, sparse 1위 -> 양쪽 점수 합산
        dense_results = [
            {"doc_id": "case-000", "score": 1.0},
            {"doc_id": "case-001", "score": 0.9},
            {"doc_id": "case-002", "score": 0.8},
        ]
        sparse_results = [
            {"doc_id": "case-002", "score": 5.0},
            {"doc_id": "case-003", "score": 4.0},
        ]

        fused = engine._reciprocal_rank_fusion(dense_results, sparse_results, IndexType.CASE)
        # case-002는 양쪽에서 점수를 받으므로 상위에 위치해야 함
        top_doc_id = fused[0]["doc_id"]
        assert (
            top_doc_id == "case-002"
        ), f"양쪽에 등장하는 case-002가 1위여야 하지만 {top_doc_id}가 1위"

    def test_empty_input_returns_empty(self, engine: HybridSearchEngine) -> None:
        fused = engine._reciprocal_rank_fusion([], [], IndexType.CASE)
        assert fused == []

    def test_weights_affect_ranking(self, engine: HybridSearchEngine) -> None:
        """LAW 타입(sparse_weight=1.2)에서 sparse 상위 문서가 더 높은 점수를 받는지 확인."""
        dense_results = [
            {"doc_id": "law-000", "score": 1.0},
        ]
        sparse_results = [
            {"doc_id": "law-001", "score": 5.0},
        ]

        # LAW: dense_weight=0.9, sparse_weight=1.2
        fused = engine._reciprocal_rank_fusion(dense_results, sparse_results, IndexType.LAW)
        assert len(fused) == 2

        # sparse_weight(1.2) > dense_weight(0.9) 이므로 sparse 1위 문서가 상위
        scores = {r["doc_id"]: r["score"] for r in fused}
        assert scores["law-001"] > scores["law-000"]

    def test_score_normalized_0_to_1(self, engine: HybridSearchEngine) -> None:
        """결과의 score가 모두 0~1 범위인지 확인."""
        dense_results = [{"doc_id": f"case-{i:03d}", "score": 1.0 - i * 0.1} for i in range(5)]
        sparse_results = [{"doc_id": f"case-{i:03d}", "score": 5.0 - i * 0.5} for i in range(3, 8)]

        fused = engine._reciprocal_rank_fusion(dense_results, sparse_results, IndexType.CASE)
        for r in fused:
            assert 0.0 <= r["score"] <= 1.0, f"score {r['score']} is out of [0, 1]"

    def test_top_score_is_1(self, engine: HybridSearchEngine) -> None:
        """정규화 후 최상위 문서의 score는 정확히 1.0이어야 한다."""
        dense_results = [
            {"doc_id": "case-000", "score": 1.0},
            {"doc_id": "case-001", "score": 0.5},
        ]
        sparse_results = [
            {"doc_id": "case-001", "score": 3.0},
        ]

        fused = engine._reciprocal_rank_fusion(dense_results, sparse_results, IndexType.CASE)
        assert fused[0]["score"] == 1.0


# ---------------------------------------------------------------------------
# TestHybridSearchDense
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHybridSearchDense:
    async def test_dense_mode_skips_bm25(
        self,
        engine: HybridSearchEngine,
        mock_bm25_indexer: mock.MagicMock,
    ) -> None:
        results, actual_mode = await engine.search("테스트", IndexType.CASE, mode=SearchMode.DENSE)
        mock_bm25_indexer.search.assert_not_called()
        assert len(results) > 0
        assert actual_mode == SearchMode.DENSE

    async def test_dense_returns_correct_structure(self, engine: HybridSearchEngine) -> None:
        results, _ = await engine.search("테스트", IndexType.CASE, mode=SearchMode.DENSE)
        for r in results:
            assert "doc_id" in r
            assert "score" in r

    async def test_dense_respects_top_k(self, engine: HybridSearchEngine) -> None:
        results, _ = await engine.search("테스트", IndexType.CASE, top_k=3, mode=SearchMode.DENSE)
        assert len(results) <= 3

    async def test_dense_calls_embed_model(
        self,
        engine: HybridSearchEngine,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        await engine.search("테스트 쿼리", IndexType.CASE, mode=SearchMode.DENSE)
        mock_embed_model.encode.assert_called_once()
        call_args = mock_embed_model.encode.call_args
        assert "query: 테스트 쿼리" in call_args[0][0]

    async def test_dense_calls_index_manager_search(
        self,
        engine: HybridSearchEngine,
        mock_index_manager: mock.MagicMock,
    ) -> None:
        await engine.search("테스트", IndexType.CASE, mode=SearchMode.DENSE)
        mock_index_manager.search.assert_called_once()


# ---------------------------------------------------------------------------
# TestHybridSearchSparse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHybridSearchSparse:
    async def test_sparse_mode_skips_dense(
        self,
        engine: HybridSearchEngine,
        mock_index_manager: mock.MagicMock,
    ) -> None:
        results, actual_mode = await engine.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        mock_index_manager.search.assert_not_called()
        assert len(results) > 0
        assert actual_mode == SearchMode.SPARSE

    async def test_sparse_maps_corpus_index_to_metadata(self, engine: HybridSearchEngine) -> None:
        """BM25가 반환한 corpus index가 metadata로 올바르게 매핑되는지 확인."""
        results, _ = await engine.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        for r in results:
            assert "doc_id" in r
            assert r["doc_id"].startswith("case-")

    async def test_sparse_result_scores_are_bm25_raw(self, engine: HybridSearchEngine) -> None:
        """sparse 모드에서 score가 BM25 원시 점수인지 확인."""
        results, _ = await engine.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        # BM25 mock이 반환한 점수: 5.2, 4.8, 3.5, 2.1, 1.5
        expected_scores = [5.2, 4.8, 3.5, 2.1, 1.5]
        actual_scores = [r["score"] for r in results]
        assert actual_scores == expected_scores

    async def test_sparse_with_missing_indexer_returns_empty(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """해당 타입에 BM25 인덱서가 없을 때 빈 결과를 반환하는지 확인."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},  # 비어있음
            embed_model=mock_embed_model,
        )
        results, _ = await engine.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        assert results == []

    async def test_sparse_with_not_ready_indexer_returns_empty(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """BM25 인덱서가 ready 상태가 아닐 때 빈 결과를 반환하는지 확인."""
        not_ready_indexer = mock.MagicMock()
        not_ready_indexer.is_ready.return_value = False
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: not_ready_indexer},
            embed_model=mock_embed_model,
        )
        results, _ = await engine.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        assert results == []

    async def test_sparse_respects_top_k(self, engine: HybridSearchEngine) -> None:
        results, _ = await engine.search("테스트", IndexType.CASE, top_k=2, mode=SearchMode.SPARSE)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# TestHybridSearchHybrid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHybridSearchHybrid:
    async def test_hybrid_calls_both_searches(
        self,
        engine: HybridSearchEngine,
        mock_index_manager: mock.MagicMock,
        mock_bm25_indexer: mock.MagicMock,
    ) -> None:
        results, actual_mode = await engine.search("테스트", IndexType.CASE, mode=SearchMode.HYBRID)
        mock_index_manager.search.assert_called_once()
        mock_bm25_indexer.search.assert_called_once()
        assert len(results) > 0
        assert actual_mode == SearchMode.HYBRID

    async def test_hybrid_respects_top_k(self, engine: HybridSearchEngine) -> None:
        results, _ = await engine.search("테스트", IndexType.CASE, top_k=3, mode=SearchMode.HYBRID)
        assert len(results) <= 3

    async def test_hybrid_fallback_to_dense_when_no_bm25(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """BM25 인덱서가 없을 때 dense 폴백이 동작하는지 확인."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=mock_embed_model,
        )
        results, actual_mode = await engine.search("테스트", IndexType.CASE, mode=SearchMode.HYBRID)
        # BM25 없으므로 dense 폴백
        assert len(results) > 0
        assert actual_mode == SearchMode.DENSE
        mock_index_manager.search.assert_called_once()

    async def test_hybrid_fallback_when_bm25_not_ready(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """BM25 인덱서가 ready 상태가 아닐 때 dense 폴백이 동작하는지 확인."""
        not_ready_indexer = mock.MagicMock()
        not_ready_indexer.is_ready.return_value = False
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: not_ready_indexer},
            embed_model=mock_embed_model,
        )
        results, actual_mode = await engine.search("테스트", IndexType.CASE, mode=SearchMode.HYBRID)
        assert len(results) > 0
        assert actual_mode == SearchMode.DENSE
        mock_index_manager.search.assert_called_once()
        not_ready_indexer.search.assert_not_called()

    async def test_hybrid_scores_in_0_to_1(self, engine: HybridSearchEngine) -> None:
        results, _ = await engine.search("테스트", IndexType.CASE, mode=SearchMode.HYBRID)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"score {r['score']} is out of [0, 1]"

    async def test_hybrid_merges_dense_and_sparse_docs(self, engine: HybridSearchEngine) -> None:
        """hybrid 결과에 dense와 sparse 양쪽의 문서가 포함되는지 확인."""
        # top_k를 충분히 크게 설정하여 모든 문서가 반환되도록 함
        results, _ = await engine.search("테스트", IndexType.CASE, top_k=10, mode=SearchMode.HYBRID)
        doc_ids = {r["doc_id"] for r in results}
        # dense mock: case-000 ~ case-004, sparse mock: case-004, 002, 000, 007, 001
        # case-007은 sparse에만 존재
        assert "case-007" in doc_ids, "sparse 전용 문서(case-007)가 hybrid 결과에 포함되어야 함"
        # dense 전용 문서(case-003)도 포함되어야 함
        assert "case-003" in doc_ids, "dense 전용 문서(case-003)가 hybrid 결과에 포함되어야 함"

    async def test_hybrid_default_mode(self, engine: HybridSearchEngine) -> None:
        """mode 파라미터 생략 시 기본값이 HYBRID인지 확인."""
        results, actual_mode = await engine.search("테스트", IndexType.CASE)
        # hybrid는 RRF 정규화로 score가 0~1 범위
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
        assert actual_mode == SearchMode.HYBRID


# ---------------------------------------------------------------------------
# TestIsReady
# ---------------------------------------------------------------------------


class TestIsReady:
    def test_ready_when_index_exists(self, engine: HybridSearchEngine) -> None:
        assert engine.is_ready(IndexType.CASE) is True

    def test_not_ready_when_no_index(self, engine: HybridSearchEngine) -> None:
        assert engine.is_ready(IndexType.LAW) is False

    def test_not_ready_when_index_empty(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """인덱스는 존재하지만 문서가 0개인 경우 False를 반환하는지 확인."""
        mock_index_manager.indexes[IndexType.MANUAL] = mock.MagicMock(ntotal=0)
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=mock_embed_model,
        )
        assert engine.is_ready(IndexType.MANUAL) is False


# ---------------------------------------------------------------------------
# TestInitialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_none_index_manager_raises(self, mock_embed_model: mock.MagicMock) -> None:
        with pytest.raises(RuntimeError, match="None"):
            HybridSearchEngine(
                index_manager=None,
                bm25_indexers={},
                embed_model=mock_embed_model,
            )

    def test_none_bm25_indexers_defaults_to_empty(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers=None,
            embed_model=mock_embed_model,
        )
        assert engine.bm25_indexers == {}

    def test_custom_rrf_k(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=mock_embed_model,
            rrf_k=30,
        )
        assert engine.rrf_k == 30

    def test_custom_rrf_weights(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        custom_weights = {IndexType.CASE: RRFWeightConfig(dense_weight=2.0, sparse_weight=0.5)}
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=mock_embed_model,
            rrf_weights=custom_weights,
        )
        assert engine.rrf_weights[IndexType.CASE].dense_weight == 2.0


# ---------------------------------------------------------------------------
# TestEmbedQuery
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    def test_embed_adds_query_prefix(
        self,
        engine: HybridSearchEngine,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """_embed_query가 'query:' 접두사를 붙이는지 확인."""
        engine._embed_query("민원 내용")
        mock_embed_model.encode.assert_called_once()
        call_args = mock_embed_model.encode.call_args[0][0]
        assert call_args == ["query: 민원 내용"]

    def test_embed_uses_normalize_embeddings(
        self,
        engine: HybridSearchEngine,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """_embed_query가 normalize_embeddings=True를 전달하는지 확인."""
        engine._embed_query("테스트")
        call_kwargs = mock_embed_model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True

    def test_embed_returns_single_vector(
        self,
        engine: HybridSearchEngine,
    ) -> None:
        """_embed_query가 1차원 벡터를 반환하는지 확인."""
        result = engine._embed_query("테스트")
        # encode()는 (1, 1024) shape -> [0]으로 (1024,) 반환
        assert result is not None


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrorHandling:
    async def test_dense_without_embed_model_raises(
        self,
        mock_index_manager: mock.MagicMock,
    ) -> None:
        """embed_model이 None일 때 dense 모드에서 RuntimeError 발생 확인."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={},
            embed_model=None,
        )
        with pytest.raises(RuntimeError, match="embed_model"):
            await engine.search("테스트", IndexType.CASE, mode=SearchMode.DENSE)

    async def test_hybrid_without_embed_model_raises(
        self,
        mock_index_manager: mock.MagicMock,
        mock_bm25_indexer: mock.MagicMock,
    ) -> None:
        """embed_model이 None일 때 hybrid 모드에서 RuntimeError 발생 확인."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: mock_bm25_indexer},
            embed_model=None,
        )
        with pytest.raises(RuntimeError, match="embed_model"):
            await engine.search("테스트", IndexType.CASE, mode=SearchMode.HYBRID)

    async def test_bm25_search_exception_returns_empty(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """BM25 search에서 예외 발생 시 빈 결과를 반환하는지 확인."""
        failing_indexer = mock.MagicMock()
        failing_indexer.is_ready.return_value = True
        failing_indexer.search.side_effect = Exception("BM25 내부 오류")
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: failing_indexer},
            embed_model=mock_embed_model,
        )
        results, _ = await engine.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        assert results == []

    async def test_corpus_index_out_of_range_skipped(
        self,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """BM25가 반환한 corpus_index가 메타데이터 범위를 초과하면 건너뛰는지 확인."""
        oob_indexer = mock.MagicMock()
        oob_indexer.is_ready.return_value = True
        # corpus_index 999는 메타데이터 범위(10) 초과
        oob_indexer.search.return_value = [(999, 5.0), (0, 3.0)]
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: oob_indexer},
            embed_model=mock_embed_model,
        )
        results, _ = await engine.search("테스트", IndexType.CASE, mode=SearchMode.SPARSE)
        # 999는 건너뛰고 0만 포함
        assert len(results) == 1
        assert results[0]["doc_id"] == "case-000"
