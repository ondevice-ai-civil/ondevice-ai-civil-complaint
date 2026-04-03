"""GovOn MVP Pydantic 스키마 테스트."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.inference.index_manager import DocumentMetadata, IndexType
from src.inference.schemas import (
    AgentRunResponse,
    AgentTraceSchema,
    DocumentMetadataSchema,
    GenerateCivilResponseRequest,
    GenerateCivilResponseResponse,
    GenerateResponse,
    RetrievedCase,
    SearchResult,
    StreamResponse,
    ToolResultSchema,
    from_internal_metadata,
)

_NOW = datetime(2026, 3, 22, 12, 0, 0)


def _make_metadata_schema(source_type: IndexType, **overrides) -> DocumentMetadataSchema:
    defaults = dict(
        doc_id="doc-001",
        source_type=source_type,
        source_id="src-001",
        title="테스트 문서",
        content="본문 내용입니다.",
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(overrides)
    return DocumentMetadataSchema(**defaults)


class TestDocumentMetadataSchema:
    @pytest.mark.parametrize("index_type", list(IndexType))
    def test_create_all_document_types(self, index_type: IndexType):
        schema = _make_metadata_schema(index_type)
        assert schema.source_type == index_type
        assert schema.doc_id == "doc-001"

    def test_json_roundtrip(self):
        original = _make_metadata_schema(IndexType.LAW, metadata={"law_number": "제1234호"})
        restored = DocumentMetadataSchema.model_validate_json(original.model_dump_json())

        assert restored.doc_id == original.doc_id
        assert restored.source_type == original.source_type
        assert restored.metadata == original.metadata

    def test_reliability_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            _make_metadata_schema(IndexType.CASE, reliability_score=1.5)
        with pytest.raises(ValidationError):
            _make_metadata_schema(IndexType.CASE, reliability_score=-0.1)

    def test_chunk_index_exceeds_total_raises(self):
        with pytest.raises(ValidationError):
            _make_metadata_schema(IndexType.CASE, chunk_index=1, total_chunks=1)


class TestSearchResult:
    def test_required_fields(self):
        result = SearchResult(
            doc_id="doc-001",
            source_type=IndexType.NOTICE,
            title="공시 정보",
            content="공시 본문",
            score=0.95,
        )
        assert result.source_type == IndexType.NOTICE
        assert result.reliability_score == 1.0
        assert result.metadata == {}

    def test_reliability_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(
                doc_id="doc-001",
                source_type=IndexType.NOTICE,
                title="공시",
                content="본문",
                score=0.9,
                reliability_score=2.0,
            )


class TestGenerateSchemas:
    def test_generate_civil_response_request_defaults(self):
        request = GenerateCivilResponseRequest(prompt="민원 답변 초안 작성")
        assert request.use_rag is True
        assert request.stream is False

    def test_generate_response_supports_retrieved_cases_and_search_results(self):
        case = RetrievedCase(complaint="민원 내용", answer="답변", score=0.9)
        search = SearchResult(
            doc_id="doc-001",
            source_type=IndexType.CASE,
            title="유사 사례",
            content="사례 본문",
            score=0.88,
        )
        response = GenerateResponse(
            request_id="req-001",
            text="생성된 응답",
            prompt_tokens=100,
            completion_tokens=50,
            retrieved_cases=[case],
            search_results=[search],
        )
        assert len(response.retrieved_cases) == 1
        assert len(response.search_results) == 1

    def test_stream_response_supports_search_results(self):
        search = SearchResult(
            doc_id="doc-002",
            source_type=IndexType.LAW,
            title="법령 결과",
            content="법령 본문",
            score=0.75,
        )
        response = StreamResponse(
            request_id="req-002",
            text="스트리밍 응답",
            search_results=[search],
        )
        assert response.search_results[0].doc_id == "doc-002"

    def test_generate_civil_response_response_accepts_optional_fields(self):
        response = GenerateCivilResponseResponse(
            request_id="req-003",
            complaint_id="complaint-1",
            text="민원 답변",
            prompt_tokens=50,
            completion_tokens=20,
        )
        assert response.complaint_id == "complaint-1"


class TestAgentSchemas:
    def test_agent_run_response_has_no_classification_field(self):
        trace = AgentTraceSchema(
            request_id="req-100",
            session_id="session-100",
            plan=["rag_search", "api_lookup", "draft_civil_response"],
            plan_reason="민원 답변 작성 또는 수정 작업으로 판단",
            tool_results=[
                ToolResultSchema(tool="rag_search", success=True, latency_ms=10.5, data={}),
            ],
            total_latency_ms=25.0,
        )
        response = AgentRunResponse(
            request_id="req-100",
            session_id="session-100",
            text="근거 요약\n\n최종 초안\n답변",
            trace=trace,
        )
        assert "classification" not in response.model_dump()
        assert response.trace.plan[0] == "rag_search"


class TestFromInternalMetadata:
    def test_from_internal_metadata(self):
        meta = DocumentMetadata(
            doc_id="doc-10",
            doc_type="case",
            source="src-10",
            title="테스트 민원",
            category="교통",
            chunk_index=0,
            chunk_total=1,
            created_at=_NOW.isoformat(),
            updated_at=_NOW.isoformat(),
            valid_from=None,
            valid_until=None,
            reliability_score=0.8,
            extras={"category": "교통"},
        )

        schema = from_internal_metadata(meta, content="본문")

        assert schema.doc_id == "doc-10"
        assert schema.source_type == IndexType.CASE
        assert schema.metadata["category"] == "교통"
