"""
ORM 모델 단위 테스트.

DocumentSource, IndexingQueue, IndexVersion 모델의
생성/조회/관계/JSONB 필드를 SQLite 인메모리 DB로 검증한다.
"""

import uuid
from datetime import date, datetime, timezone

import pytest

from src.inference.db.models import DocumentSource, IndexingQueue, IndexVersion


class TestDocumentSource:
    """DocumentSource 모델 테스트."""

    def test_create_with_required_fields(self, db_session, sample_doc_kwargs):
        """필수 필드만으로 DocumentSource를 생성할 수 있다."""
        doc = DocumentSource(**sample_doc_kwargs)
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        assert doc.id == sample_doc_kwargs["id"]
        assert doc.source_type == "case"
        assert doc.source_id == "CASE-001"
        assert doc.title == "테스트 민원 사례"
        assert doc.content == "민원 내용 본문 텍스트"
        assert doc.chunk_index == 0
        assert doc.total_chunks == 1
        assert doc.reliability_score == 0.8

    def test_create_with_case_fields(self, db_session, sample_doc_kwargs):
        """CASE 전용 필드(complaint_text, answer_text) 포함 생성."""
        sample_doc_kwargs.update(
            {
                "complaint_text": "도로 포장이 파손되었습니다",
                "answer_text": "해당 구간 보수 공사를 진행하겠습니다",
            }
        )
        doc = DocumentSource(**sample_doc_kwargs)
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        assert doc.complaint_text == "도로 포장이 파손되었습니다"
        assert doc.answer_text == "해당 구간 보수 공사를 진행하겠습니다"
        # LAW 전용 필드는 None이어야 함
        assert doc.law_number is None
        assert doc.article_number is None

    def test_create_with_law_fields(self, db_session, sample_doc_kwargs):
        """LAW 전용 필드(law_number, article_number, enforcement_date) 포함 생성."""
        sample_doc_kwargs.update(
            {
                "source_type": "law",
                "source_id": "LAW-001",
                "law_number": "제1234호",
                "article_number": "제5조",
                "enforcement_date": date(2026, 1, 1),
            }
        )
        doc = DocumentSource(**sample_doc_kwargs)
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        assert doc.law_number == "제1234호"
        assert doc.article_number == "제5조"
        assert doc.enforcement_date == date(2026, 1, 1)
        # CASE 전용 필드는 None
        assert doc.complaint_text is None

    def test_metadata_json_field(self, db_session, sample_doc_kwargs):
        """metadata_ (JSONB) 필드에 딕셔너리를 저장하고 조회한다."""
        sample_doc_kwargs["metadata_"] = {
            "keywords": ["도로", "보수"],
            "priority": "high",
            "nested": {"key": "value"},
        }
        doc = DocumentSource(**sample_doc_kwargs)
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        assert doc.metadata_["keywords"] == ["도로", "보수"]
        assert doc.metadata_["priority"] == "high"
        assert doc.metadata_["nested"]["key"] == "value"

    def test_repr(self, sample_doc):
        """__repr__ 출력 형식 확인."""
        repr_str = repr(sample_doc)
        assert "DocumentSource" in repr_str
        assert "case" in repr_str

    def test_created_at_auto_set(self, sample_doc):
        """created_at이 자동으로 설정된다."""
        assert sample_doc.created_at is not None

    def test_updated_at_auto_set(self, sample_doc):
        """updated_at이 자동으로 설정된다."""
        assert sample_doc.updated_at is not None


class TestIndexingQueue:
    """IndexingQueue 모델 테스트."""

    def test_create_with_document_fk(self, db_session, sample_doc):
        """DocumentSource FK 관계로 IndexingQueue를 생성한다."""
        item = IndexingQueue(
            id=uuid.uuid4(),
            document_id=sample_doc.id,
            doc_type="CASE",
            complaint_text="민원 텍스트",
            answer_text="답변 텍스트",
            status="pending",
            priority=5,
        )
        db_session.add(item)
        db_session.commit()
        db_session.refresh(item)

        assert item.document_id == sample_doc.id
        assert item.status == "pending"
        assert item.priority == 5
        assert item.processed_at is None

    def test_relationship_bidirectional(self, db_session, sample_doc):
        """DocumentSource <-> IndexingQueue 양방향 관계를 검증한다."""
        item = IndexingQueue(
            id=uuid.uuid4(),
            document_id=sample_doc.id,
            doc_type="CASE",
            complaint_text="민원",
            answer_text="답변",
            status="pending",
            priority=0,
        )
        db_session.add(item)
        db_session.commit()

        # queue -> document 방향
        db_session.refresh(item)
        assert item.document is not None
        assert item.document.id == sample_doc.id

        # document -> queue_items 방향
        db_session.refresh(sample_doc)
        assert len(sample_doc.queue_items) == 1
        assert sample_doc.queue_items[0].id == item.id

    def test_create_without_document(self, db_session):
        """document_id 없이(NULL) IndexingQueue를 생성할 수 있다."""
        item = IndexingQueue(
            id=uuid.uuid4(),
            document_id=None,
            doc_type="CASE",
            complaint_text="민원",
            answer_text="답변",
            status="pending",
            priority=0,
        )
        db_session.add(item)
        db_session.commit()
        db_session.refresh(item)

        assert item.document_id is None
        assert item.document is None

    def test_repr(self, db_session, sample_doc):
        """__repr__ 출력 형식 확인."""
        item = IndexingQueue(
            id=uuid.uuid4(),
            document_id=sample_doc.id,
            doc_type="CASE",
            complaint_text="민원",
            answer_text="답변",
            status="pending",
            priority=0,
        )
        db_session.add(item)
        db_session.commit()

        repr_str = repr(item)
        assert "IndexingQueue" in repr_str
        assert "pending" in repr_str


class TestIndexVersion:
    """IndexVersion 모델 테스트."""

    def test_create_and_query(self, db_session, sample_index_version_kwargs):
        """IndexVersion을 생성하고 조회한다."""
        ver = IndexVersion(**sample_index_version_kwargs)
        db_session.add(ver)
        db_session.commit()
        db_session.refresh(ver)

        fetched = db_session.get(IndexVersion, sample_index_version_kwargs["id"])
        assert fetched is not None
        assert fetched.index_type == "case"
        assert fetched.version == "v1.0.0"
        assert fetched.total_documents == 100
        assert fetched.is_active is True
        assert fetched.index_file_path == "/data/faiss/case/index.faiss"
        assert fetched.meta_file_path == "/data/faiss/case/metadata.json"

    def test_optional_fields_default_null(self, db_session, sample_index_version_kwargs):
        """선택 필드(snapshot_path, build_duration_seconds, notes)는 기본 NULL이다."""
        ver = IndexVersion(**sample_index_version_kwargs)
        db_session.add(ver)
        db_session.commit()
        db_session.refresh(ver)

        assert ver.snapshot_path is None
        assert ver.build_duration_seconds is None
        assert ver.notes is None

    def test_with_optional_fields(self, db_session, sample_index_version_kwargs):
        """선택 필드를 포함하여 생성한다."""
        sample_index_version_kwargs.update(
            {
                "snapshot_path": "/snapshots/case_v1.tar.gz",
                "build_duration_seconds": 45.2,
                "notes": "초기 빌드",
            }
        )
        ver = IndexVersion(**sample_index_version_kwargs)
        db_session.add(ver)
        db_session.commit()
        db_session.refresh(ver)

        assert ver.snapshot_path == "/snapshots/case_v1.tar.gz"
        assert ver.build_duration_seconds == pytest.approx(45.2)
        assert ver.notes == "초기 빌드"

    def test_repr(self, db_session, sample_index_version_kwargs):
        """__repr__ 출력 형식 확인."""
        ver = IndexVersion(**sample_index_version_kwargs)
        db_session.add(ver)
        db_session.commit()

        repr_str = repr(ver)
        assert "IndexVersion" in repr_str
        assert "case" in repr_str
        assert "v1.0.0" in repr_str
