"""DocumentMetadataSchema, SearchResult 및 하위 호환성 테스트.

이슈 #151: DocumentMetadata 스키마 확장 검증.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.inference.index_manager import DocumentMetadata, IndexType
from src.inference.schemas import (
    DocumentMetadataSchema,
    GenerateCivilResponseRequest,
    GenerateCivilResponseResponse,
    GeneratePublicDocRequest,
    GeneratePublicDocResponse,
    GenerateResponse,
    RetrievedCase,
    SearchResult,
    StreamResponse,
    from_internal_metadata,
)

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 22, 12, 0, 0)
_NOW_ISO = _NOW.isoformat()


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


# ---------------------------------------------------------------------------
# 1. DocumentMetadataSchema 테스트
# ---------------------------------------------------------------------------


class TestDocumentMetadataSchema:
    """DocumentMetadataSchema 모델 유효성 검증."""

    @pytest.mark.parametrize("index_type", list(IndexType))
    def test_create_all_document_types(self, index_type: IndexType):
        """4종 문서 타입(CASE, LAW, MANUAL, NOTICE) 각각 생성 확인."""
        schema = _make_metadata_schema(index_type)
        assert schema.source_type == index_type
        assert schema.doc_id == "doc-001"

    def test_json_roundtrip(self):
        """JSON 직렬화(model_dump_json) / 역직렬화(model_validate_json) 왕복 테스트."""
        original = _make_metadata_schema(
            IndexType.LAW,
            metadata={"law_number": "제1234호"},
        )
        json_str = original.model_dump_json()
        restored = DocumentMetadataSchema.model_validate_json(json_str)

        assert restored.doc_id == original.doc_id
        assert restored.source_type == original.source_type
        assert restored.metadata == original.metadata
        assert restored.created_at == original.created_at

    def test_reliability_score_valid_range(self):
        """reliability_score 범위 검증 (0.0~1.0)."""
        # 경계값 정상
        schema_low = _make_metadata_schema(IndexType.CASE, reliability_score=0.0)
        assert schema_low.reliability_score == 0.0

        schema_high = _make_metadata_schema(IndexType.CASE, reliability_score=1.0)
        assert schema_high.reliability_score == 1.0

    def test_reliability_score_out_of_range_raises(self):
        """reliability_score 범위 밖 값은 ValidationError."""
        with pytest.raises(ValidationError):
            _make_metadata_schema(IndexType.CASE, reliability_score=1.5)

        with pytest.raises(ValidationError):
            _make_metadata_schema(IndexType.CASE, reliability_score=-0.1)

    def test_chunk_index_exceeds_total_raises(self):
        """chunk_index >= total_chunks이면 ValidationError."""
        with pytest.raises(ValidationError):
            _make_metadata_schema(IndexType.CASE, chunk_index=5, total_chunks=3)

        with pytest.raises(ValidationError):
            _make_metadata_schema(IndexType.CASE, chunk_index=1, total_chunks=1)

    def test_default_values(self):
        """기본값 확인 (chunk_index=0, total_chunks=1, reliability_score=1.0)."""
        schema = _make_metadata_schema(IndexType.MANUAL)
        assert schema.chunk_index == 0
        assert schema.total_chunks == 1
        assert schema.reliability_score == 1.0
        assert schema.metadata == {}
        assert schema.valid_from is None
        assert schema.valid_until is None


# ---------------------------------------------------------------------------
# 2. SearchResult 테스트
# ---------------------------------------------------------------------------


class TestSearchResult:
    """SearchResult 모델 필드 검증."""

    def test_required_fields(self):
        """source_type, reliability_score, metadata, score 필드 포함 확인."""
        result = SearchResult(
            doc_id="doc-001",
            source_type=IndexType.NOTICE,
            title="공시 정보",
            content="공시 본문",
            score=0.95,
        )
        assert result.source_type == IndexType.NOTICE
        assert result.score == 0.95
        assert result.reliability_score == 1.0
        assert result.metadata == {}

    def test_reliability_score_valid_range(self):
        """SearchResult reliability_score 범위 검증 (0.0~1.0)."""
        result = SearchResult(
            doc_id="doc-001",
            source_type=IndexType.NOTICE,
            title="공시 정보",
            content="공시 본문",
            score=0.95,
            reliability_score=0.0,
        )
        assert result.reliability_score == 0.0

    def test_reliability_score_out_of_range_raises(self):
        """SearchResult reliability_score 범위 밖 값은 ValidationError."""
        with pytest.raises(ValidationError):
            SearchResult(
                doc_id="doc-001",
                source_type=IndexType.NOTICE,
                title="공시",
                content="본문",
                score=0.9,
                reliability_score=1.5,
            )

        with pytest.raises(ValidationError):
            SearchResult(
                doc_id="doc-001",
                source_type=IndexType.NOTICE,
                title="공시",
                content="본문",
                score=0.9,
                reliability_score=-0.1,
            )

    def test_with_metadata(self):
        """metadata 필드에 임의 데이터 저장 확인."""
        result = SearchResult(
            doc_id="doc-002",
            source_type=IndexType.LAW,
            title="법령",
            content="법령 본문",
            score=0.88,
            metadata={"law_number": "제5678호"},
        )
        assert result.metadata["law_number"] == "제5678호"


# ---------------------------------------------------------------------------
# 3. 하위 호환성 테스트
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """기존 모델 하위 호환성 검증."""

    def test_retrieved_case_still_works(self):
        """RetrievedCase 모델이 여전히 동작하는지 확인."""
        case = RetrievedCase(
            id="case-001",
            category="도로/교통",
            complaint="도로 파손 민원",
            answer="보수 완료",
            score=0.92,
        )
        assert case.complaint == "도로 파손 민원"
        assert case.score == 0.92

    def test_generate_response_both_fields(self):
        """GenerateResponse에 retrieved_cases와 search_results 모두 사용 가능 확인."""
        case = RetrievedCase(
            complaint="민원 내용",
            answer="답변",
            score=0.9,
        )
        search = SearchResult(
            doc_id="doc-001",
            source_type=IndexType.CASE,
            title="유사 사례",
            content="사례 본문",
            score=0.88,
        )
        resp = GenerateResponse(
            request_id="req-001",
            text="생성된 응답",
            prompt_tokens=100,
            completion_tokens=50,
            retrieved_cases=[case],
            search_results=[search],
        )
        assert len(resp.retrieved_cases) == 1
        assert len(resp.search_results) == 1
        assert resp.search_results[0].source_type == IndexType.CASE

    def test_stream_response_search_results(self):
        """StreamResponse에 search_results 필드 사용 가능 확인."""
        search = SearchResult(
            doc_id="doc-002",
            source_type=IndexType.LAW,
            title="법령 결과",
            content="법령 본문",
            score=0.75,
        )
        resp = StreamResponse(
            request_id="req-002",
            text="스트리밍 응답",
            search_results=[search],
        )
        assert resp.search_results[0].doc_id == "doc-002"

    def test_generate_response_without_new_fields(self):
        """새 필드 없이 기존 방식으로도 GenerateResponse 생성 가능."""
        resp = GenerateResponse(
            request_id="req-003",
            text="기존 방식 응답",
            prompt_tokens=50,
            completion_tokens=30,
        )
        assert resp.search_results is None
        assert resp.retrieved_cases is None


class TestSeparatedGenerateSchemas:
    """공문서/민원답변 분리 스키마 검증."""

    def test_generate_public_doc_request_has_doc_type(self):
        req = GeneratePublicDocRequest(prompt="보도자료 초안 작성", doc_type="press_release")
        assert req.doc_type == "press_release"

    def test_generate_public_doc_response_supports_html(self):
        resp = GeneratePublicDocResponse(
            request_id="req-pub-001",
            text="공문서 초안",
            doc_type="official_document",
            formatted_html="<p>공문서 초안</p>",
            prompt_tokens=120,
            completion_tokens=80,
        )
        assert resp.formatted_html == "<p>공문서 초안</p>"

    def test_generate_civil_response_models_support_complaint_id(self):
        req = GenerateCivilResponseRequest(prompt="민원 회신 초안", complaint_id="cmp-001")
        resp = GenerateCivilResponseResponse(
            request_id="req-civ-001",
            complaint_id=req.complaint_id,
            text="민원 회신 초안",
            prompt_tokens=90,
            completion_tokens=60,
            retrieved_cases=[
                RetrievedCase(
                    complaint="도로 파손 민원",
                    answer="도로 보수 예정",
                    score=0.91,
                )
            ],
        )
        assert resp.complaint_id == "cmp-001"
        assert resp.retrieved_cases[0].complaint == "도로 파손 민원"


# ---------------------------------------------------------------------------
# 4. dataclass -> Pydantic 변환 테스트
# ---------------------------------------------------------------------------


class TestFromInternalMetadata:
    """index_manager.DocumentMetadata -> DocumentMetadataSchema 변환 헬퍼 테스트."""

    def test_basic_conversion(self):
        """기본 변환이 올바르게 동작하는지 확인."""
        internal = DocumentMetadata(
            doc_id="doc-100",
            doc_type="case",
            source="AI Hub",
            title="유사 민원 사례",
            category="환경/위생",
            reliability_score=0.85,
            created_at=_NOW_ISO,
            updated_at=_NOW_ISO,
            extras={"complaint_text": "악취 민원"},
        )
        result = from_internal_metadata(internal, content="민원 답변 본문")

        assert result.doc_id == "doc-100"
        assert result.source_type == IndexType.CASE
        assert result.source_id == "AI Hub"
        assert result.title == "유사 민원 사례"
        assert result.content == "민원 답변 본문"
        assert result.reliability_score == 0.85
        assert result.created_at == _NOW
        assert result.metadata == {"complaint_text": "악취 민원"}

    def test_conversion_with_valid_until(self):
        """valid_until이 있는 경우 변환 확인."""
        valid_until_dt = datetime(2027, 12, 31, 23, 59, 59)
        internal = DocumentMetadata(
            doc_id="doc-200",
            doc_type="law",
            source="법제처",
            title="환경보전법",
            category="법령",
            reliability_score=1.0,
            created_at=_NOW_ISO,
            updated_at=_NOW_ISO,
            valid_until=valid_until_dt.isoformat(),
        )
        result = from_internal_metadata(internal)

        assert result.source_type == IndexType.LAW
        assert result.valid_until == valid_until_dt

    def test_conversion_chunk_fields(self):
        """chunk_index, chunk_total -> total_chunks 매핑 확인."""
        internal = DocumentMetadata(
            doc_id="doc-300",
            doc_type="manual",
            source="기관 내부",
            title="업무 매뉴얼 3장",
            category="매뉴얼",
            reliability_score=0.9,
            created_at=_NOW_ISO,
            updated_at=_NOW_ISO,
            chunk_index=2,
            chunk_total=5,
        )
        result = from_internal_metadata(internal, content="3장 내용")

        assert result.chunk_index == 2
        assert result.total_chunks == 5

    def test_conversion_empty_extras(self):
        """extras가 None 또는 빈 dict인 경우 metadata가 빈 dict."""
        internal = DocumentMetadata(
            doc_id="doc-400",
            doc_type="notice",
            source="기관 공시",
            title="공시 문서",
            category="공시",
            reliability_score=0.7,
            created_at=_NOW_ISO,
            updated_at=_NOW_ISO,
            extras=None,
        )
        result = from_internal_metadata(internal)
        assert result.metadata == {}

    def test_conversion_with_valid_from(self):
        """valid_from이 있는 경우 변환 확인."""
        valid_from_dt = datetime(2026, 1, 1, 0, 0, 0)
        valid_until_dt = datetime(2027, 12, 31, 23, 59, 59)
        internal = DocumentMetadata(
            doc_id="doc-500",
            doc_type="law",
            source="법제처",
            title="환경보전법",
            category="법령",
            reliability_score=1.0,
            created_at=_NOW_ISO,
            updated_at=_NOW_ISO,
            valid_from=valid_from_dt.isoformat(),
            valid_until=valid_until_dt.isoformat(),
        )
        result = from_internal_metadata(internal)

        assert result.source_type == IndexType.LAW
        assert result.valid_from == valid_from_dt
        assert result.valid_until == valid_until_dt

    def test_conversion_invalid_doc_type_raises(self):
        """유효하지 않은 doc_type은 ValueError."""
        internal = DocumentMetadata(
            doc_id="doc-600",
            doc_type="unknown",
            source="테스트",
            title="테스트",
            category="테스트",
            reliability_score=0.5,
            created_at=_NOW_ISO,
            updated_at=_NOW_ISO,
        )
        with pytest.raises(ValueError):
            from_internal_metadata(internal)
