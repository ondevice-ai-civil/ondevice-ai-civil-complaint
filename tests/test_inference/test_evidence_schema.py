"""EvidenceItem/EvidenceEnvelope 정규화 스키마 단위 테스트.

Issue #155: mixed evidence 응답 스키마 정규화.

테스트 케이스:
  1. EvidenceItem/EvidenceEnvelope 직렬화
  2. RagSearchCapability evidence 필드 채워지는지 확인
  3. ApiLookupCapability evidence 필드 채워지는지 확인
  4. 정상/빈결과/부분결과/에러 4케이스
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("SKIP_MODEL_LOAD", "true")

from src.inference.graph.capabilities.base import (
    EvidenceEnvelope,
    EvidenceItem,
    LookupResult,
)

# ---------------------------------------------------------------------------
# 1. EvidenceItem / EvidenceEnvelope 직렬화
# ---------------------------------------------------------------------------


class TestEvidenceItemSerialization:
    def test_to_dict_basic(self):
        item = EvidenceItem(
            source_type="rag",
            title="테스트 문서",
            excerpt="본문 내용입니다.",
            link_or_path="/docs/test.pdf",
            page=3,
            score=0.92,
        )
        d = item.to_dict()
        assert d["source_type"] == "rag"
        assert d["title"] == "테스트 문서"
        assert d["excerpt"] == "본문 내용입니다."
        assert d["link_or_path"] == "/docs/test.pdf"
        assert d["page"] == 3
        assert d["score"] == 0.92

    def test_to_dict_defaults(self):
        item = EvidenceItem(source_type="api", title="", excerpt="")
        d = item.to_dict()
        assert d["link_or_path"] == ""
        assert d["page"] is None
        assert d["score"] == 0.0
        assert d["provider_meta"] == {}

    def test_api_type(self):
        item = EvidenceItem(
            source_type="api",
            title="유사 민원 사례",
            excerpt="민원 내용 요약",
            link_or_path="https://data.go.kr/example",
            score=5.0,
            provider_meta={"provider": "data.go.kr"},
        )
        d = item.to_dict()
        assert d["source_type"] == "api"
        assert d["provider_meta"]["provider"] == "data.go.kr"


class TestEvidenceEnvelopeSerialization:
    def test_to_dict_ok(self):
        items = [
            EvidenceItem(source_type="rag", title="doc1", excerpt="내용1"),
            EvidenceItem(source_type="api", title="case1", excerpt="사례1"),
        ]
        env = EvidenceEnvelope(items=items, summary_text="요약", status="ok")
        d = env.to_dict()
        assert d["status"] == "ok"
        assert len(d["items"]) == 2
        assert d["summary_text"] == "요약"
        assert d["errors"] == []

    def test_to_dict_error(self):
        env = EvidenceEnvelope(
            status="error",
            errors=["API 호출 실패"],
        )
        d = env.to_dict()
        assert d["status"] == "error"
        assert "API 호출 실패" in d["errors"]
        assert d["items"] == []

    def test_to_dict_empty(self):
        env = EvidenceEnvelope(status="empty")
        d = env.to_dict()
        assert d["status"] == "empty"
        assert d["items"] == []

    def test_to_dict_partial(self):
        items = [EvidenceItem(source_type="rag", title="doc1", excerpt="내용")]
        env = EvidenceEnvelope(
            items=items,
            status="partial",
            errors=["api provider error"],
        )
        d = env.to_dict()
        assert d["status"] == "partial"
        assert len(d["items"]) == 1
        assert len(d["errors"]) == 1


class TestLookupResultWithEvidence:
    def test_to_dict_includes_evidence(self):
        items = [EvidenceItem(source_type="rag", title="doc", excerpt="text")]
        env = EvidenceEnvelope(items=items, status="ok")
        result = LookupResult(
            success=True,
            query="테스트",
            evidence=env,
        )
        d = result.to_dict()
        assert "evidence" in d
        assert d["evidence"]["status"] == "ok"
        assert len(d["evidence"]["items"]) == 1

    def test_to_dict_without_evidence(self):
        result = LookupResult(success=True, query="테스트")
        d = result.to_dict()
        # evidence 필드는 None일 때 직렬화에 포함되지 않음
        assert "evidence" not in d


# ---------------------------------------------------------------------------
# 2. RagSearchCapability evidence 필드
# ---------------------------------------------------------------------------


class TestRagSearchCapabilityEvidence:
    @pytest.mark.asyncio
    async def test_ok_result_has_evidence(self):
        from src.inference.graph.capabilities.rag_search import RagSearchCapability

        raw_results = [
            {
                "title": "행정절차법",
                "content": "행정절차에 관한 법률입니다." * 5,
                "score": 0.85,
                "metadata": {"file_path": "/laws/admin.pdf", "page": 12},
            }
        ]
        execute_fn = AsyncMock(
            return_value={"query": "행정절차", "results": raw_results, "context_text": "요약"}
        )
        cap = RagSearchCapability(execute_fn)
        result = await cap.execute("행정절차", {}, MagicMock())

        assert result.evidence is not None
        assert result.evidence.status == "ok"
        assert len(result.evidence.items) == 1
        item = result.evidence.items[0]
        assert item.source_type == "rag"
        assert item.title == "행정절차법"
        assert item.link_or_path == "/laws/admin.pdf"
        assert item.page == 12
        assert item.score == 0.85

    @pytest.mark.asyncio
    async def test_empty_result_has_empty_envelope(self):
        from src.inference.graph.capabilities.rag_search import RagSearchCapability

        execute_fn = AsyncMock(return_value={"query": "없는검색어", "results": []})
        cap = RagSearchCapability(execute_fn)
        result = await cap.execute("없는검색어", {}, MagicMock())

        assert result.evidence is not None
        assert result.evidence.status == "empty"
        assert result.evidence.items == []

    @pytest.mark.asyncio
    async def test_error_result_has_error_envelope(self):
        from src.inference.graph.capabilities.rag_search import RagSearchCapability

        execute_fn = AsyncMock(return_value={"error": "DB 연결 실패", "query": "검색어"})
        cap = RagSearchCapability(execute_fn)
        result = await cap.execute("검색어", {}, MagicMock())

        assert result.success is False
        assert result.evidence is not None
        assert result.evidence.status == "error"
        assert "DB 연결 실패" in result.evidence.errors

    @pytest.mark.asyncio
    async def test_excerpt_truncated_to_500(self):
        from src.inference.graph.capabilities.rag_search import RagSearchCapability

        long_content = "가" * 1000
        execute_fn = AsyncMock(
            return_value={
                "query": "q",
                "results": [{"title": "t", "content": long_content, "score": 0.5}],
            }
        )
        cap = RagSearchCapability(execute_fn)
        result = await cap.execute("q", {}, MagicMock())

        assert result.evidence is not None
        assert len(result.evidence.items[0].excerpt) == 500


# ---------------------------------------------------------------------------
# 3. ApiLookupCapability evidence 필드
# ---------------------------------------------------------------------------


class TestApiLookupCapabilityEvidence:
    @pytest.mark.asyncio
    async def test_ok_result_has_api_evidence(self):
        from src.inference.graph.capabilities.api_lookup import ApiLookupCapability

        mock_action = MagicMock()
        mock_action.fetch_similar_cases = AsyncMock(
            return_value={
                "query": "소음 민원",
                "results": [
                    {
                        "qnaTitle": "층간소음 민원",
                        "qnaContent": "층간소음 관련 처리 절차",
                        "detailUrl": "https://data.go.kr/case/1",
                        "score": 4,
                    }
                ],
                "citations": [],
                "context_text": "관련 사례 요약",
            }
        )
        cap = ApiLookupCapability(action=mock_action)
        result = await cap.execute("소음 민원", {}, MagicMock())

        assert result.evidence is not None
        assert result.evidence.status == "ok"
        assert len(result.evidence.items) == 1
        item = result.evidence.items[0]
        assert item.source_type == "api"
        assert item.title == "층간소음 민원"
        assert item.link_or_path == "https://data.go.kr/case/1"
        assert item.score == 4.0

    @pytest.mark.asyncio
    async def test_no_action_returns_empty_envelope(self):
        from src.inference.graph.capabilities.api_lookup import ApiLookupCapability

        cap = ApiLookupCapability(action=None)
        result = await cap.execute("검색어", {}, MagicMock())

        assert result.success is True
        assert result.evidence is not None
        assert result.evidence.status == "empty"

    @pytest.mark.asyncio
    async def test_validation_error_returns_error_envelope(self):
        from src.inference.graph.capabilities.api_lookup import ApiLookupCapability

        cap = ApiLookupCapability(action=None)
        # 빈 query는 validation 실패
        result = await cap.execute("", {}, MagicMock())

        assert result.success is False
        assert result.evidence is not None
        assert result.evidence.status == "error"
        assert len(result.evidence.errors) > 0

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_envelope(self):
        from src.inference.graph.capabilities.api_lookup import ApiLookupCapability

        mock_action = MagicMock()
        mock_action.fetch_similar_cases = AsyncMock(
            return_value={"query": "희귀민원", "results": [], "citations": []}
        )
        cap = ApiLookupCapability(action=mock_action)
        result = await cap.execute("희귀민원", {}, MagicMock())

        assert result.success is True
        assert result.evidence is not None
        assert result.evidence.status == "empty"
