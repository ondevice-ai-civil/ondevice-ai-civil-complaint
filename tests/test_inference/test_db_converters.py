"""
변환 헬퍼 단위 테스트.

converters.py의 orm_to_dataclass, dataclass_to_orm, orm_to_pydantic 함수를
SQLite 인메모리 DB로 검증한다.
"""

import uuid
from datetime import date, datetime, timezone

import pytest

from src.inference.db.converters import (
    dataclass_to_orm,
    orm_to_dataclass,
    orm_to_pydantic,
)
from src.inference.db.crud import create_document_source
from src.inference.db.models import DocumentSource
from src.inference.index_manager import DocumentMetadata, IndexType
from src.inference.schemas import DocumentMetadataSchema


class TestOrmToDataclass:
    """orm_to_dataclass 테스트."""

    def test_basic_conversion(self, db_session, sample_doc):
        """ORM -> DocumentMetadata 기본 변환이 정확하다."""
        meta = orm_to_dataclass(sample_doc)

        assert isinstance(meta, DocumentMetadata)
        assert meta.doc_id == str(sample_doc.id)
        assert meta.doc_type == "case"
        assert meta.title == "테스트 민원 사례"
        assert meta.reliability_score == 0.8
        assert meta.chunk_index == 0
        assert meta.chunk_total == 1
        assert meta.created_at is not None
        assert meta.updated_at is not None

    def test_case_specific_fields_in_extras(self, db_session, sample_doc_kwargs):
        """CASE 전용 필드가 extras에 포함된다."""
        sample_doc_kwargs.update({
            "complaint_text": "민원 내용",
            "answer_text": "답변 내용",
        })
        doc = create_document_source(db_session, **sample_doc_kwargs)
        meta = orm_to_dataclass(doc)

        assert "complaint_text" in meta.extras
        assert meta.extras["complaint_text"] == "민원 내용"
        assert "answer_text" in meta.extras
        assert meta.extras["answer_text"] == "답변 내용"

    def test_law_specific_fields_in_extras(self, db_session, sample_doc_kwargs):
        """LAW 전용 필드가 extras에 포함된다."""
        sample_doc_kwargs.update({
            "source_type": "law",
            "law_number": "제1234호",
            "article_number": "제5조",
            "enforcement_date": date(2026, 1, 1),
        })
        doc = create_document_source(db_session, **sample_doc_kwargs)
        meta = orm_to_dataclass(doc)

        assert "law_number" in meta.extras
        assert meta.extras["law_number"] == "제1234호"
        assert "article_number" in meta.extras
        assert "enforcement_date" in meta.extras
        # date는 ISO 문자열로 변환
        assert meta.extras["enforcement_date"] == "2026-01-01"

    def test_metadata_json_merged_into_extras(self, db_session, sample_doc_kwargs):
        """metadata_ JSONB 필드 값이 extras에 병합된다."""
        sample_doc_kwargs["metadata_"] = {"custom_key": "custom_value"}
        doc = create_document_source(db_session, **sample_doc_kwargs)
        meta = orm_to_dataclass(doc)

        assert "custom_key" in meta.extras
        assert meta.extras["custom_key"] == "custom_value"

    def test_source_name_maps_to_source(self, db_session, sample_doc_kwargs):
        """source_name이 dataclass의 source 필드로 매핑된다."""
        sample_doc_kwargs["source_name"] = "AI Hub"
        doc = create_document_source(db_session, **sample_doc_kwargs)
        meta = orm_to_dataclass(doc)

        assert meta.source == "AI Hub"

    def test_source_name_none_maps_to_empty(self, db_session, sample_doc):
        """source_name이 None이면 source는 빈 문자열이다."""
        meta = orm_to_dataclass(sample_doc)
        assert meta.source == ""

    def test_valid_from_until_conversion(self, db_session, sample_doc_kwargs):
        """valid_from/valid_until이 ISO 문자열로 변환된다."""
        now = datetime(2026, 1, 1, 0, 0, 0)
        later = datetime(2027, 1, 1, 0, 0, 0)
        sample_doc_kwargs.update({
            "valid_from": now,
            "valid_until": later,
        })
        doc = create_document_source(db_session, **sample_doc_kwargs)
        meta = orm_to_dataclass(doc)

        assert meta.valid_from == now.isoformat()
        assert meta.valid_until == later.isoformat()


class TestDataclassToOrm:
    """dataclass_to_orm 테스트."""

    def _make_meta(self, **overrides):
        """헬퍼: DocumentMetadata 인스턴스를 생성한다."""
        defaults = {
            "doc_id": "DOC-001",
            "doc_type": "case",
            "source": "AI Hub",
            "title": "테스트 문서",
            "category": "도로/교통",
            "reliability_score": 0.85,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-15T00:00:00",
            "chunk_index": 0,
            "chunk_total": 1,
            "extras": {},
        }
        defaults.update(overrides)
        return DocumentMetadata(**defaults)

    def test_basic_conversion(self):
        """DocumentMetadata -> ORM kwargs 기본 변환이 정확하다."""
        meta = self._make_meta()
        kwargs = dataclass_to_orm(meta, content="문서 본문")

        assert kwargs["source_type"] == "case"
        assert kwargs["source_id"] == "DOC-001"
        assert kwargs["source_name"] == "AI Hub"
        assert kwargs["title"] == "테스트 문서"
        assert kwargs["content"] == "문서 본문"
        assert kwargs["category"] == "도로/교통"
        assert kwargs["chunk_index"] == 0
        assert kwargs["total_chunks"] == 1
        assert kwargs["reliability_score"] == 0.85

    def test_type_specific_fields_from_extras(self):
        """extras의 타입별 전용 필드가 kwargs 최상위로 분리된다."""
        meta = self._make_meta(extras={
            "complaint_text": "민원 텍스트",
            "answer_text": "답변 텍스트",
            "custom_key": "custom_value",
        })
        kwargs = dataclass_to_orm(meta, content="본문")

        # 타입별 필드는 최상위로
        assert kwargs["complaint_text"] == "민원 텍스트"
        assert kwargs["answer_text"] == "답변 텍스트"
        # 나머지는 metadata_로
        assert kwargs["metadata_"] == {"custom_key": "custom_value"}

    def test_valid_from_until_parsing(self):
        """valid_from/valid_until ISO 문자열이 datetime으로 변환된다."""
        meta = self._make_meta(
            valid_from="2026-01-01T00:00:00",
            valid_until="2027-01-01T00:00:00",
        )
        kwargs = dataclass_to_orm(meta, content="본문")

        assert isinstance(kwargs["valid_from"], datetime)
        assert isinstance(kwargs["valid_until"], datetime)
        assert kwargs["valid_from"].year == 2026
        assert kwargs["valid_until"].year == 2027

    def test_empty_extras(self):
        """extras가 비어있으면 metadata_도 비어있다."""
        meta = self._make_meta(extras={})
        kwargs = dataclass_to_orm(meta, content="본문")
        assert kwargs["metadata_"] == {}

    def test_none_extras(self):
        """extras가 None이면 metadata_는 비어있다."""
        meta = self._make_meta(extras=None)
        kwargs = dataclass_to_orm(meta, content="본문")
        assert kwargs["metadata_"] == {}

    def test_law_fields_from_extras(self):
        """LAW 전용 필드(law_number, article_number)가 분리된다."""
        meta = self._make_meta(
            doc_type="law",
            extras={
                "law_number": "제1234호",
                "article_number": "제5조",
                "enforcement_date": "2026-01-01",
            },
        )
        kwargs = dataclass_to_orm(meta, content="법령 본문")

        assert kwargs["law_number"] == "제1234호"
        assert kwargs["article_number"] == "제5조"
        assert kwargs["enforcement_date"] == "2026-01-01"
        assert kwargs["metadata_"] == {}


class TestOrmToPydantic:
    """orm_to_pydantic 테스트."""

    def test_basic_conversion(self, db_session, sample_doc):
        """ORM -> DocumentMetadataSchema 기본 변환이 정확하다."""
        schema = orm_to_pydantic(sample_doc)

        assert isinstance(schema, DocumentMetadataSchema)
        assert schema.doc_id == str(sample_doc.id)
        assert schema.source_type == IndexType.CASE
        assert schema.source_id == "CASE-001"
        assert schema.title == "테스트 민원 사례"
        assert schema.content == "민원 내용 본문 텍스트"
        assert schema.chunk_index == 0
        assert schema.total_chunks == 1
        assert schema.reliability_score == 0.8

    def test_type_specific_fields_in_metadata(self, db_session, sample_doc_kwargs):
        """타입별 전용 필드가 Pydantic 모델의 metadata에 포함된다."""
        sample_doc_kwargs.update({
            "complaint_text": "민원 내용",
            "answer_text": "답변 내용",
        })
        doc = create_document_source(db_session, **sample_doc_kwargs)
        schema = orm_to_pydantic(doc)

        assert "complaint_text" in schema.metadata
        assert schema.metadata["complaint_text"] == "민원 내용"
        assert "answer_text" in schema.metadata

    def test_timestamps_are_datetime(self, db_session, sample_doc):
        """created_at, updated_at이 datetime 타입이다."""
        schema = orm_to_pydantic(sample_doc)

        assert isinstance(schema.created_at, datetime)
        assert isinstance(schema.updated_at, datetime)

    def test_jsonb_metadata_merged(self, db_session, sample_doc_kwargs):
        """JSONB metadata_ 값이 Pydantic metadata에 병합된다."""
        sample_doc_kwargs["metadata_"] = {"region": "서울"}
        doc = create_document_source(db_session, **sample_doc_kwargs)
        schema = orm_to_pydantic(doc)

        assert "region" in schema.metadata
        assert schema.metadata["region"] == "서울"

    def test_pydantic_model_serialization(self, db_session, sample_doc):
        """Pydantic 모델이 JSON 직렬화 가능하다."""
        schema = orm_to_pydantic(sample_doc)
        data = schema.model_dump()

        assert isinstance(data, dict)
        assert "doc_id" in data
        assert "source_type" in data
        assert "metadata" in data
