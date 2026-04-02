"""
RAGSearchTool 단위 테스트.

HybridSearchEngine / CivilComplaintRetriever를 mock하여
tool 인터페이스의 입출력 계약과 에러 처리를 검증한다.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

# 무거운 의존성 mock 등록
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("faiss", MagicMock())

import pytest

from src.inference.hybrid_search import SearchMode
from src.inference.index_manager import IndexType
from src.inference.tools.base import ToolInput, ToolOutput
from src.inference.tools.rag_search import RAGSearchInput, RAGSearchTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hybrid_engine():
    """HybridSearchEngine mock."""
    engine = AsyncMock()
    engine.search = AsyncMock(
        return_value=(
            [
                {
                    "doc_id": "CASE-001",
                    "doc_type": "case",
                    "title": "도로 파손 민원",
                    "score": 0.95,
                    "reliability_score": 0.9,
                    "extras": {
                        "complaint_text": "도로가 파손되어 위험합니다",
                        "answer_text": "보수 공사 예정입니다",
                    },
                    "chunk_index": 0,
                    "chunk_total": 1,
                },
                {
                    "doc_id": "CASE-002",
                    "doc_type": "case",
                    "title": "보도블록 파손",
                    "score": 0.82,
                    "reliability_score": 0.85,
                    "extras": {
                        "complaint_text": "보도블록이 파손되었습니다",
                        "answer_text": "교체 예정입니다",
                    },
                    "chunk_index": 0,
                    "chunk_total": 1,
                },
            ],
            SearchMode.HYBRID,
        )
    )
    return engine


@pytest.fixture
def mock_retriever():
    """CivilComplaintRetriever mock."""
    retriever = MagicMock()
    retriever.search.return_value = [
        {
            "id": "LEGACY-001",
            "category": "도로/교통",
            "complaint": "도로 파손 신고",
            "answer": "즉시 보수하겠습니다",
            "score": 0.88,
        }
    ]
    return retriever


@pytest.fixture
def mock_pii_masker():
    """PIIMasker mock."""
    masker = MagicMock()
    masker.mask_all.side_effect = lambda text: text.replace("홍길동", "***")
    return masker


# ---------------------------------------------------------------------------
# 초기화 테스트
# ---------------------------------------------------------------------------


class TestRAGSearchToolInit:
    def test_requires_at_least_one_engine(self):
        with pytest.raises(ValueError, match="필수"):
            RAGSearchTool(hybrid_engine=None, retriever=None)

    def test_hybrid_engine_only(self, mock_hybrid_engine):
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine)
        assert tool.name == "rag_search"

    def test_retriever_only(self, mock_retriever):
        tool = RAGSearchTool(retriever=mock_retriever)
        assert tool.name == "rag_search"


# ---------------------------------------------------------------------------
# Hybrid 검색 테스트
# ---------------------------------------------------------------------------


class TestRAGSearchHybrid:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_hybrid_engine):
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine)
        inp = RAGSearchInput(query="도로 파손", doc_type=IndexType.CASE, top_k=5)
        output = await tool.run(inp)

        assert output.success is True
        assert output.tool_name == "rag_search"
        assert output.data["total"] == 2
        assert output.data["query"] == "도로 파손"
        assert output.data["doc_type"] == "case"
        assert output.data["search_mode"] == "hybrid"

        first = output.data["results"][0]
        assert first["doc_id"] == "CASE-001"
        assert first["score"] == 0.95
        assert "도로가 파손" in first["content"]

    @pytest.mark.asyncio
    async def test_search_passes_params_correctly(self, mock_hybrid_engine):
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine)
        inp = RAGSearchInput(
            query="법령 조회",
            doc_type=IndexType.LAW,
            top_k=10,
            search_mode=SearchMode.DENSE,
        )
        await tool.run(inp)

        mock_hybrid_engine.search.assert_called_once_with(
            query="법령 조회",
            index_type=IndexType.LAW,
            top_k=10,
            mode=SearchMode.DENSE,
        )

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_hybrid_engine):
        mock_hybrid_engine.search.return_value = ([], SearchMode.HYBRID)
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine)
        inp = RAGSearchInput(query="존재하지 않는 쿼리")
        output = await tool.run(inp)

        assert output.success is True
        assert output.data["total"] == 0
        assert output.data["results"] == []


# ---------------------------------------------------------------------------
# 레거시 폴백 테스트
# ---------------------------------------------------------------------------


class TestRAGSearchLegacy:
    @pytest.mark.asyncio
    async def test_legacy_fallback(self, mock_retriever):
        tool = RAGSearchTool(retriever=mock_retriever)
        inp = RAGSearchInput(query="도로 파손 신고")
        output = await tool.run(inp)

        assert output.success is True
        assert output.data["search_mode"] == "dense"
        assert output.data["total"] == 1

        first = output.data["results"][0]
        assert first["doc_id"] == "LEGACY-001"
        assert "도로 파손 신고" in first["content"]

    @pytest.mark.asyncio
    async def test_legacy_prefers_hybrid(self, mock_hybrid_engine, mock_retriever):
        """hybrid_engine과 retriever 모두 있으면 hybrid_engine 우선."""
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine, retriever=mock_retriever)
        inp = RAGSearchInput(query="도로")
        await tool.run(inp)

        mock_hybrid_engine.search.assert_called_once()
        mock_retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# 에러 처리 테스트
# ---------------------------------------------------------------------------


class TestRAGSearchErrors:
    @pytest.mark.asyncio
    async def test_invalid_input_type(self, mock_hybrid_engine):
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine)
        output = await tool.run(ToolInput())

        assert output.success is False
        assert "잘못된 입력 타입" in output.error

    @pytest.mark.asyncio
    async def test_engine_exception_handled(self, mock_hybrid_engine):
        mock_hybrid_engine.search.side_effect = RuntimeError("인덱스 손상")
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine)
        inp = RAGSearchInput(query="테스트")
        output = await tool.run(inp)

        assert output.success is False
        assert "인덱스 손상" in output.error


# ---------------------------------------------------------------------------
# PII 마스킹 테스트
# ---------------------------------------------------------------------------


class TestRAGSearchPIIMasking:
    @pytest.mark.asyncio
    async def test_pii_masking_applied(self, mock_hybrid_engine, mock_pii_masker):
        mock_hybrid_engine.search.return_value = (
            [
                {
                    "doc_id": "CASE-PII",
                    "doc_type": "case",
                    "title": "개인정보 테스트",
                    "score": 0.9,
                    "reliability_score": 1.0,
                    "extras": {
                        "complaint_text": "홍길동이 신고했습니다",
                        "answer_text": "처리 완료",
                    },
                    "chunk_index": 0,
                    "chunk_total": 1,
                }
            ],
            SearchMode.HYBRID,
        )
        tool = RAGSearchTool(
            hybrid_engine=mock_hybrid_engine,
            pii_masker=mock_pii_masker,
        )
        inp = RAGSearchInput(query="개인정보 테스트")
        output = await tool.run(inp)

        assert output.success is True
        first = output.data["results"][0]
        # content에서 홍길동 → *** 마스킹 확인
        assert "홍길동" not in first["content"]
        assert "***" in first["content"]


# ---------------------------------------------------------------------------
# 스키마 테스트
# ---------------------------------------------------------------------------


class TestRAGSearchSchema:
    def test_get_schema(self, mock_hybrid_engine):
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine)
        schema = tool.get_schema()
        assert schema["name"] == "rag_search"
        assert "parameters" in schema
        params = schema["parameters"]
        assert "properties" in params
        assert "query" in params["properties"]
        assert "doc_type" in params["properties"]
        assert "top_k" in params["properties"]

    def test_input_validation(self):
        with pytest.raises(Exception):
            RAGSearchInput(query="")  # min_length=1

        inp = RAGSearchInput(query="정상 쿼리", top_k=3)
        assert inp.top_k == 3
        assert inp.doc_type == IndexType.CASE
        assert inp.search_mode == SearchMode.HYBRID


# ---------------------------------------------------------------------------
# 성능 테스트 (응답 시간 기록)
# ---------------------------------------------------------------------------


class TestRAGSearchPerformance:
    @pytest.mark.asyncio
    async def test_elapsed_time_recorded(self, mock_hybrid_engine):
        tool = RAGSearchTool(hybrid_engine=mock_hybrid_engine)
        inp = RAGSearchInput(query="성능 테스트")
        output = await tool.run(inp)

        assert output.elapsed_ms is not None
        assert output.elapsed_ms >= 0
