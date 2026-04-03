"""
빌드 → 검색 → RRF 전체 경로 통합 테스트 (Issue #154).

PR #192 리뷰 지적 해결: E2E 테스트가 BM25를 mock하므로 실제 빌드-검색 경로가 미검증.
이 테스트는 **실제 BM25Indexer**를 사용하되, FAISS/vLLM 등 GPU 의존성은 mock한다.

Tests cover:
- BM25 인덱스 빌드 (input/output 형식 JSONL)
- BM25 인덱스 빌드 (EXAONE template 형식 JSONL)
- 실제 BM25 검색 결과 검증
- RRF 융합 (실제 BM25 + mock Dense)
- 타입별 content 추출 (_extract_content_by_type)
- 양측 검색 실패 시 에러 전파
"""

import json
import os
import sys
import unittest.mock as mock

# ---------------------------------------------------------------------------
# 무거운 외부 의존성 mock (import 전에 등록)
# ---------------------------------------------------------------------------
_vllm_mock = mock.MagicMock()
_vllm_mock.AsyncLLM = mock.MagicMock()
_vllm_mock.SamplingParams = mock.MagicMock()
sys.modules.setdefault("vllm", _vllm_mock)
sys.modules.setdefault("vllm.engine", _vllm_mock)
sys.modules.setdefault("vllm.engine.arg_utils", _vllm_mock)
sys.modules.setdefault("vllm.engine.async_llm_engine", _vllm_mock)
sys.modules.setdefault("vllm.sampling_params", _vllm_mock)
sys.modules.setdefault("sentence_transformers", mock.MagicMock())

_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, mock.MagicMock)
if not _faiss_is_real:
    _faiss_mock = mock.MagicMock()
    _faiss_mock.IndexIVFFlat = type("IndexIVFFlat", (), {})
    _faiss_mock.IndexFlatIP = type("IndexFlatIP", (), {})
    sys.modules["faiss"] = _faiss_mock

_mock_stabilizer = mock.MagicMock()
_mock_stabilizer.apply_transformers_patch = mock.MagicMock()
sys.modules.setdefault("src.inference.vllm_stabilizer", _mock_stabilizer)

_mock_retriever_module = mock.MagicMock()
sys.modules.setdefault("src.inference.retriever", _mock_retriever_module)

import numpy as np
import pytest

from src.inference.hybrid_search import HybridSearchEngine, SearchMode
from src.inference.index_manager import DocumentMetadata, IndexType

# konlpy 설치 여부에 따라 BM25 의존 테스트를 건너뛴다.
try:
    import konlpy  # noqa: F401
    from src.inference.bm25_indexer import BM25Indexer

    HAS_KONLPY = True
except ImportError:
    HAS_KONLPY = False
    BM25Indexer = None  # type: ignore[assignment,misc]

requires_konlpy = pytest.mark.skipif(not HAS_KONLPY, reason="konlpy가 필요한 테스트입니다")

# ---------------------------------------------------------------------------
# 샘플 데이터 생성 헬퍼
# ---------------------------------------------------------------------------

_INPUT_OUTPUT_SAMPLES = [
    {
        "id": str(i),
        "instruction": "다음 민원에 답변하세요",
        "input": text,
        "output": answer,
        "category": category,
        "source": "AI Hub",
    }
    for i, (text, answer, category) in enumerate(
        [
            (
                "도로 포장이 파손되어 보행에 불편합니다",
                "해당 지역 도로 보수 공사를 진행하겠습니다",
                "도로/교통",
            ),
            ("가로등이 고장나서 밤에 어둡습니다", "가로등 수리를 요청하였습니다", "도로/교통"),
            ("소음이 심하여 잠을 잘 수 없습니다", "소음 측정 후 조치하겠습니다", "환경/위생"),
            ("쓰레기가 방치되어 악취가 납니다", "해당 구역 청소를 진행하겠습니다", "환경/위생"),
            ("주차 위반 차량이 많아 통행이 어렵습니다", "단속을 강화하겠습니다", "도로/교통"),
            ("하수구가 막혀서 물이 넘칩니다", "하수관 점검을 실시하겠습니다", "환경/위생"),
            ("보도블록이 파손되어 위험합니다", "보도블록 교체 공사를 진행하겠습니다", "도로/교통"),
            ("공원 벤치가 파손되었습니다", "벤치 교체를 요청하였습니다", "공원/녹지"),
            ("신호등이 제대로 작동하지 않습니다", "신호등 점검을 실시하겠습니다", "도로/교통"),
            (
                "도로 포장 균열이 심하여 차량 통행에 위험합니다",
                "도로 보수를 진행하겠습니다",
                "도로/교통",
            ),
        ]
    )
]

_EXAONE_TEMPLATE_SAMPLES = [
    {
        "text": (
            f"[|system|]당신은 민원 상담 도우미입니다[|endofturn|]\n"
            f"[|user|]민원 내용: {text}[|endofturn|]\n"
            f"[|assistant|]{answer}[|endofturn|]"
        )
    }
    for text, answer in [
        ("도로 포장이 파손되어 보행에 불편합니다", "해당 지역 도로 보수 공사를 진행하겠습니다"),
        ("가로등이 고장나서 밤에 어둡습니다", "가로등 수리를 요청하였습니다"),
        ("소음이 심하여 잠을 잘 수 없습니다", "소음 측정 후 조치하겠습니다"),
        ("쓰레기가 방치되어 악취가 납니다", "해당 구역 청소를 진행하겠습니다"),
        ("주차 위반 차량이 많아 통행이 어렵습니다", "단속을 강화하겠습니다"),
        ("하수구가 막혀서 물이 넘칩니다", "하수관 점검을 실시하겠습니다"),
        ("보도블록이 파손되어 위험합니다", "보도블록 교체 공사를 진행하겠습니다"),
        ("공원 벤치가 파손되었습니다", "벤치 교체를 요청하였습니다"),
        ("신호등이 제대로 작동하지 않습니다", "신호등 점검을 실시하겠습니다"),
        ("도로 포장 균열이 심하여 차량 통행에 위험합니다", "도로 보수를 진행하겠습니다"),
    ]
]


def _write_jsonl(path: str, records: list) -> None:
    """JSONL 파일을 생성한다."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def input_output_jsonl(tmp_path) -> str:
    """input/output 형식 JSONL 파일 경로를 반환한다."""
    path = str(tmp_path / "sample_input_output.jsonl")
    _write_jsonl(path, _INPUT_OUTPUT_SAMPLES)
    return path


@pytest.fixture
def exaone_template_jsonl(tmp_path) -> str:
    """EXAONE template 형식 JSONL 파일 경로를 반환한다."""
    path = str(tmp_path / "sample_template.jsonl")
    _write_jsonl(path, _EXAONE_TEMPLATE_SAMPLES)
    return path


@pytest.fixture
def bm25_indexer_input_output(input_output_jsonl: str) -> BM25Indexer:
    """input/output 형식 JSONL로 빌드된 BM25Indexer를 반환한다."""
    indexer = BM25Indexer()
    indexer.build_index_from_jsonl(input_output_jsonl)
    return indexer


@pytest.fixture
def sample_metadata() -> list:
    """BM25 인덱스와 동일한 순서의 DocumentMetadata 리스트 (10개)."""
    return [
        DocumentMetadata(
            doc_id=f"case-{i:03d}",
            doc_type=IndexType.CASE.value,
            source="AI Hub",
            title=_INPUT_OUTPUT_SAMPLES[i]["input"][:20],
            category=_INPUT_OUTPUT_SAMPLES[i]["category"],
            reliability_score=0.8,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
            extras={
                "complaint_text": _INPUT_OUTPUT_SAMPLES[i]["input"],
                "answer_text": _INPUT_OUTPUT_SAMPLES[i]["output"],
            },
        )
        for i in range(10)
    ]


@pytest.fixture
def mock_index_manager(sample_metadata: list) -> mock.MagicMock:
    """FAISS MultiIndexManager mock (Dense 검색 결과 시뮬레이션)."""
    manager = mock.MagicMock()
    manager.metadata = {IndexType.CASE: sample_metadata}
    manager.indexes = {IndexType.CASE: mock.MagicMock(ntotal=10)}

    def fake_search(index_type: IndexType, query_vector: np.ndarray, top_k: int = 5):
        meta_list = manager.metadata.get(index_type, [])
        results = []
        for i in range(min(top_k, len(meta_list))):
            d = meta_list[i].to_dict()
            d["score"] = 1.0 - i * 0.1
            results.append(d)
        return results

    manager.search.side_effect = fake_search
    return manager


@pytest.fixture
def mock_embed_model() -> mock.MagicMock:
    """SentenceTransformer mock."""
    model = mock.MagicMock()
    model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
    return model


# ---------------------------------------------------------------------------
# 1. BM25 빌드 테스트 — input/output 형식 JSONL
# ---------------------------------------------------------------------------


@requires_konlpy
class TestBM25BuildInputOutput:
    """input/output 형식 JSONL로 BM25 인덱스를 빌드한다."""

    def test_build_from_input_output_jsonl(self, input_output_jsonl: str) -> None:
        """10건의 input/output 형식 JSONL에서 인덱스를 빌드하고 상태를 확인한다."""
        indexer = BM25Indexer()
        indexer.build_index_from_jsonl(input_output_jsonl)

        assert indexer.doc_count == 10, f"문서 수가 10이어야 하지만 {indexer.doc_count}"
        assert indexer.is_ready() is True, "빌드 후 is_ready()가 True여야 함"

    def test_build_input_field_fallback(self, tmp_path) -> None:
        """text 필드 없이 input 필드만 있는 JSONL도 빌드가 성공해야 한다."""
        path = str(tmp_path / "input_only.jsonl")
        records = [{"input": f"민원 내용 {i}", "output": f"답변 {i}"} for i in range(5)]
        _write_jsonl(path, records)

        indexer = BM25Indexer()
        indexer.build_index_from_jsonl(path)
        assert indexer.doc_count == 5
        assert indexer.is_ready() is True


# ---------------------------------------------------------------------------
# 2. BM25 빌드 테스트 — EXAONE template 형식 JSONL
# ---------------------------------------------------------------------------


@requires_konlpy
class TestBM25BuildExaoneTemplate:
    """EXAONE chat template 형식 JSONL로 BM25 인덱스를 빌드한다."""

    def test_build_from_exaone_template_jsonl(self, exaone_template_jsonl: str) -> None:
        """10건의 EXAONE template 형식 JSONL에서 빌드가 성공해야 한다."""
        indexer = BM25Indexer()
        indexer.build_index_from_jsonl(exaone_template_jsonl)

        assert indexer.doc_count == 10, f"문서 수가 10이어야 하지만 {indexer.doc_count}"
        assert indexer.is_ready() is True, "빌드 후 is_ready()가 True여야 함"

    def test_exaone_template_extracts_complaint(self, exaone_template_jsonl: str) -> None:
        """EXAONE template에서 민원 내용이 올바르게 추출되어 검색 가능해야 한다."""
        indexer = BM25Indexer()
        indexer.build_index_from_jsonl(exaone_template_jsonl)

        # 도로 포장 관련 검색이 결과를 반환해야 한다
        results = indexer.search("도로 포장 파손", top_k=3)
        assert len(results) > 0, "EXAONE template에서 추출한 민원으로 검색이 가능해야 함"


# ---------------------------------------------------------------------------
# 3. BM25 검색 테스트
# ---------------------------------------------------------------------------


@requires_konlpy
class TestBM25Search:
    """실제 BM25 인덱스에서 검색을 수행한다."""

    def test_search_returns_results(self, bm25_indexer_input_output: BM25Indexer) -> None:
        """도로 포장 민원 검색 시 결과가 비어있지 않아야 한다."""
        results = bm25_indexer_input_output.search("도로 포장 민원", top_k=5)
        assert len(results) > 0, "검색 결과가 비어있으면 안 됨"

    def test_search_result_format(self, bm25_indexer_input_output: BM25Indexer) -> None:
        """검색 결과가 List[Tuple[int, float]] 형식이어야 한다."""
        results = bm25_indexer_input_output.search("도로 포장 민원", top_k=5)
        for item in results:
            assert isinstance(item, tuple), f"결과 항목이 tuple이어야 하지만 {type(item)}"
            assert len(item) == 2, f"결과 항목이 2-tuple이어야 하지만 길이 {len(item)}"
            idx, score = item
            assert isinstance(idx, int), f"인덱스가 int여야 하지만 {type(idx)}"
            assert isinstance(score, float), f"점수가 float여야 하지만 {type(score)}"

    def test_search_scores_positive(self, bm25_indexer_input_output: BM25Indexer) -> None:
        """검색 결과의 score가 모두 양수여야 한다."""
        results = bm25_indexer_input_output.search("도로 포장 민원", top_k=5)
        for idx, score in results:
            assert score > 0, f"score가 양수여야 하지만 {score} (idx={idx})"

    def test_search_respects_top_k(self, bm25_indexer_input_output: BM25Indexer) -> None:
        """top_k 제한이 올바르게 적용되어야 한다."""
        results = bm25_indexer_input_output.search("도로 포장 민원", top_k=3)
        assert len(results) <= 3, f"top_k=3인데 결과가 {len(results)}개"

    def test_search_scores_descending(self, bm25_indexer_input_output: BM25Indexer) -> None:
        """검색 결과가 점수 내림차순으로 정렬되어야 한다."""
        results = bm25_indexer_input_output.search("도로 포장 민원", top_k=5)
        if len(results) > 1:
            scores = [score for _, score in results]
            for i in range(len(scores) - 1):
                assert (
                    scores[i] >= scores[i + 1]
                ), f"점수가 내림차순이 아님: {scores[i]} < {scores[i + 1]}"

    def test_search_index_in_range(self, bm25_indexer_input_output: BM25Indexer) -> None:
        """검색 결과의 인덱스가 문서 범위(0~9) 내에 있어야 한다."""
        results = bm25_indexer_input_output.search("도로 포장 민원", top_k=5)
        for idx, _ in results:
            assert 0 <= idx < 10, f"인덱스가 범위 밖: {idx}"


# ---------------------------------------------------------------------------
# 4. RRF 융합 테스트 (실제 BM25 + mock Dense)
# ---------------------------------------------------------------------------


@requires_konlpy
@pytest.mark.asyncio
class TestRRFFusionIntegration:
    """실제 BM25Indexer + mock Dense 검색으로 RRF 융합을 검증한다."""

    async def test_hybrid_search_rrf_scores_normalized(
        self,
        bm25_indexer_input_output: BM25Indexer,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """RRF 융합 결과의 score가 0~1 범위여야 한다."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: bm25_indexer_input_output},
            embed_model=mock_embed_model,
        )

        results, actual_mode = await engine.search(
            "도로 포장 민원", IndexType.CASE, top_k=5, mode=SearchMode.HYBRID
        )

        assert len(results) > 0, "RRF 융합 결과가 비어있으면 안 됨"
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"score {r['score']}이 [0, 1] 범위 밖"

    async def test_hybrid_search_returns_hybrid_mode(
        self,
        bm25_indexer_input_output: BM25Indexer,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """실제 BM25가 준비된 상태에서 actual_mode가 HYBRID여야 한다."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: bm25_indexer_input_output},
            embed_model=mock_embed_model,
        )

        results, actual_mode = await engine.search(
            "도로 포장 민원", IndexType.CASE, top_k=5, mode=SearchMode.HYBRID
        )

        assert actual_mode == SearchMode.HYBRID, f"actual_mode가 HYBRID여야 하지만 {actual_mode}"

    async def test_hybrid_search_merges_both_sources(
        self,
        bm25_indexer_input_output: BM25Indexer,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """RRF 융합 결과에 dense와 sparse 양쪽 결과가 반영되어야 한다."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: bm25_indexer_input_output},
            embed_model=mock_embed_model,
        )

        results, _ = await engine.search(
            "도로 포장 민원", IndexType.CASE, top_k=10, mode=SearchMode.HYBRID
        )

        # Dense mock은 case-000 ~ case-009를 순서대로, BM25는 관련성 기반
        doc_ids = {r["doc_id"] for r in results}
        assert len(doc_ids) > 0, "결과에 문서가 포함되어야 함"
        # dense 결과의 case-000이 포함되어야 함 (1위 문서)
        assert "case-000" in doc_ids, "Dense 1위 문서(case-000)가 결과에 포함되어야 함"

    async def test_hybrid_search_top_score_is_1(
        self,
        bm25_indexer_input_output: BM25Indexer,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """RRF 정규화 후 최상위 문서의 score는 정확히 1.0이어야 한다."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: bm25_indexer_input_output},
            embed_model=mock_embed_model,
        )

        results, _ = await engine.search(
            "도로 포장 민원", IndexType.CASE, top_k=5, mode=SearchMode.HYBRID
        )

        assert results[0]["score"] == 1.0, f"최상위 score가 1.0이어야 하지만 {results[0]['score']}"

    async def test_sparse_only_with_real_bm25(
        self,
        bm25_indexer_input_output: BM25Indexer,
        mock_index_manager: mock.MagicMock,
        mock_embed_model: mock.MagicMock,
    ) -> None:
        """SPARSE 모드에서 실제 BM25 검색 결과가 메타데이터로 매핑되어야 한다."""
        engine = HybridSearchEngine(
            index_manager=mock_index_manager,
            bm25_indexers={IndexType.CASE: bm25_indexer_input_output},
            embed_model=mock_embed_model,
        )

        results, actual_mode = await engine.search(
            "도로 포장 민원", IndexType.CASE, top_k=5, mode=SearchMode.SPARSE
        )

        assert actual_mode == SearchMode.SPARSE
        assert len(results) > 0, "SPARSE 모드에서 결과가 비어있으면 안 됨"
        for r in results:
            assert "doc_id" in r, "결과에 doc_id 필드가 있어야 함"
            assert r["doc_id"].startswith("case-"), f"doc_id가 case-로 시작해야 함: {r['doc_id']}"
            assert r["score"] > 0, f"BM25 원시 점수가 양수여야 함: {r['score']}"


# ---------------------------------------------------------------------------
# 5. 타입별 content 추출 테스트 (_extract_content_by_type)
# ---------------------------------------------------------------------------


class TestExtractContentByType:
    """api_server.py의 _extract_content_by_type() 함수를 직접 호출하여 검증한다."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        """_extract_content_by_type를 import한다."""
        from src.inference.api_server import _extract_content_by_type

        self.extract = _extract_content_by_type

    def test_case_type_combines_complaint_and_answer(self) -> None:
        """CASE 타입: complaint_text + answer_text 조합."""
        result = {
            "extras": {
                "complaint_text": "도로 포장이 파손되었습니다",
                "answer_text": "보수 공사를 진행하겠습니다",
            },
            "title": "도로 민원",
        }
        content = self.extract(result, IndexType.CASE)
        assert "도로 포장이 파손되었습니다" in content
        assert "보수 공사를 진행하겠습니다" in content

    def test_law_type_uses_law_text(self) -> None:
        """LAW 타입: law_text 필드 사용."""
        result = {
            "extras": {"law_text": "도로법 제50조 제1항"},
            "title": "도로법",
        }
        content = self.extract(result, IndexType.LAW)
        assert content == "도로법 제50조 제1항"

    def test_law_type_fallback_to_content(self) -> None:
        """LAW 타입: law_text가 없으면 content 필드를 사용."""
        result = {
            "extras": {"content": "법률 내용입니다"},
            "title": "법령",
        }
        content = self.extract(result, IndexType.LAW)
        assert content == "법률 내용입니다"

    def test_manual_type_uses_manual_text(self) -> None:
        """MANUAL 타입: manual_text 필드 사용."""
        result = {
            "extras": {"manual_text": "업무 매뉴얼 내용입니다"},
            "title": "업무 매뉴얼",
        }
        content = self.extract(result, IndexType.MANUAL)
        assert content == "업무 매뉴얼 내용입니다"

    def test_manual_type_fallback_to_content(self) -> None:
        """MANUAL 타입: manual_text가 없으면 content 필드를 사용."""
        result = {
            "extras": {"content": "매뉴얼 내용"},
            "title": "매뉴얼",
        }
        content = self.extract(result, IndexType.MANUAL)
        assert content == "매뉴얼 내용"

    def test_notice_type_uses_notice_text(self) -> None:
        """NOTICE 타입: notice_text 필드 사용."""
        result = {
            "extras": {"notice_text": "공시 정보 내용입니다"},
            "title": "공시",
        }
        content = self.extract(result, IndexType.NOTICE)
        assert content == "공시 정보 내용입니다"

    def test_notice_type_fallback_to_content(self) -> None:
        """NOTICE 타입: notice_text가 없으면 content 필드를 사용."""
        result = {
            "extras": {"content": "공시 내용"},
            "title": "공시",
        }
        content = self.extract(result, IndexType.NOTICE)
        assert content == "공시 내용"

    def test_empty_extras_falls_back_to_title(self) -> None:
        """extras가 비어있을 때 title을 폴백으로 반환한다."""
        result = {
            "extras": {},
            "title": "폴백 타이틀",
        }
        content = self.extract(result, IndexType.CASE)
        assert content == "폴백 타이틀"

    def test_missing_extras_falls_back_to_title(self) -> None:
        """extras 키 자체가 없을 때 title을 폴백으로 반환한다."""
        result = {
            "title": "타이틀 폴백",
        }
        content = self.extract(result, IndexType.LAW)
        assert content == "타이틀 폴백"

    def test_all_empty_returns_empty_string(self) -> None:
        """extras도 title도 없으면 빈 문자열을 반환한다."""
        result = {}
        content = self.extract(result, IndexType.CASE)
        assert content == ""


# ---------------------------------------------------------------------------
# 6. 양측 검색 실패 시 에러 전파 테스트
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBothSearchFailureErrorPropagation:
    """Dense/Sparse 모두 예외 발생 시 RuntimeError가 전파되는지 검증한다."""

    async def test_both_sides_fail_raises_runtime_error(self) -> None:
        """Dense와 Sparse 검색 모두 실패하면 RuntimeError가 발생해야 한다.

        _sparse_search 내부에서 BM25 예외를 잡아 []를 반환하므로,
        asyncio.gather(return_exceptions=True) 수준에서 양측 모두
        예외 객체가 되려면 _dense_search와 _sparse_search를 직접 mock해야 한다.
        """
        failing_index_manager = mock.MagicMock()
        failing_index_manager.indexes = {IndexType.CASE: mock.MagicMock(ntotal=10)}
        failing_index_manager.metadata = {IndexType.CASE: []}

        failing_embed = mock.MagicMock()
        failing_embed.encode.return_value = np.random.randn(1, 1024).astype(np.float32)

        failing_bm25 = mock.MagicMock()
        failing_bm25.is_ready.return_value = True

        engine = HybridSearchEngine(
            index_manager=failing_index_manager,
            bm25_indexers={IndexType.CASE: failing_bm25},
            embed_model=failing_embed,
        )

        # _dense_search와 _sparse_search 모두 예외를 발생시키도록 mock
        engine._dense_search = mock.AsyncMock(side_effect=RuntimeError("Dense 검색 실패"))
        engine._sparse_search = mock.AsyncMock(side_effect=RuntimeError("Sparse 검색 실패"))

        with pytest.raises(RuntimeError, match="Dense 및 Sparse 검색 모두 실패"):
            await engine.search("테스트 쿼리", IndexType.CASE, top_k=5, mode=SearchMode.HYBRID)

    async def test_dense_fails_sparse_succeeds_returns_sparse(self) -> None:
        """Dense만 실패하면 Sparse 결과로 대체되고 SPARSE 모드를 반환한다."""
        failing_index_manager = mock.MagicMock()
        failing_index_manager.indexes = {IndexType.CASE: mock.MagicMock(ntotal=10)}
        meta = [
            DocumentMetadata(
                doc_id="case-000",
                doc_type="case",
                source="AI Hub",
                title="테스트",
                category="도로/교통",
                reliability_score=0.8,
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
                extras={"complaint_text": "민원", "answer_text": "답변"},
            )
        ]
        failing_index_manager.metadata = {IndexType.CASE: meta}
        failing_index_manager.search.side_effect = RuntimeError("Dense 검색 실패")

        working_bm25 = mock.MagicMock()
        working_bm25.is_ready.return_value = True
        working_bm25.search.return_value = [(0, 3.5)]

        embed_model = mock.MagicMock()
        embed_model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)

        engine = HybridSearchEngine(
            index_manager=failing_index_manager,
            bm25_indexers={IndexType.CASE: working_bm25},
            embed_model=embed_model,
        )

        results, actual_mode = await engine.search(
            "테스트", IndexType.CASE, top_k=5, mode=SearchMode.HYBRID
        )

        assert actual_mode == SearchMode.SPARSE
        assert len(results) > 0

    async def test_sparse_fails_dense_succeeds_returns_dense(self) -> None:
        """_sparse_search가 예외를 raise하면 Dense 결과로 대체되고 DENSE 모드를 반환한다.

        _sparse_search 내부에서 BM25 예외를 잡아 []를 반환하므로,
        asyncio.gather 수준에서 예외가 되려면 _sparse_search를 직접 mock해야 한다.
        """
        working_index_manager = mock.MagicMock()
        working_index_manager.indexes = {IndexType.CASE: mock.MagicMock(ntotal=10)}
        meta = [
            DocumentMetadata(
                doc_id="case-000",
                doc_type="case",
                source="AI Hub",
                title="테스트",
                category="도로/교통",
                reliability_score=0.8,
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
                extras={"complaint_text": "민원", "answer_text": "답변"},
            )
        ]
        working_index_manager.metadata = {IndexType.CASE: meta}
        working_index_manager.search.return_value = [
            {
                "doc_id": "case-000",
                "score": 0.9,
                "doc_type": "case",
                "title": "테스트",
            }
        ]

        working_bm25 = mock.MagicMock()
        working_bm25.is_ready.return_value = True

        embed_model = mock.MagicMock()
        embed_model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)

        engine = HybridSearchEngine(
            index_manager=working_index_manager,
            bm25_indexers={IndexType.CASE: working_bm25},
            embed_model=embed_model,
        )

        # _sparse_search만 예외를 발생시키도록 mock
        engine._sparse_search = mock.AsyncMock(side_effect=RuntimeError("Sparse 검색 실패"))

        results, actual_mode = await engine.search(
            "테스트", IndexType.CASE, top_k=5, mode=SearchMode.HYBRID
        )

        assert actual_mode == SearchMode.DENSE
        assert len(results) > 0
