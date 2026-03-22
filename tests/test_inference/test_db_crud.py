"""
CRUD 레이어 단위 테스트.

crud.py의 14개 함수를 SQLite 인메모리 DB로 검증한다.
"""

import uuid
from datetime import datetime, timezone

import pytest

from src.inference.db.crud import (
    activate_version,
    create_document_source,
    create_index_version,
    create_indexing_queue_item,
    deactivate_versions,
    delete_document_source,
    get_active_version,
    get_by_source_type_and_id,
    get_document_source,
    get_document_sources,
    get_pending_items,
    get_queue_stats,
    update_document_source,
    update_queue_status,
)
from src.inference.db.models import DocumentSource, IndexVersion


# ============================================================================
# DocumentSource CRUD
# ============================================================================


class TestCreateDocumentSource:
    """create_document_source 테스트."""

    def test_create_success(self, db_session, sample_doc_kwargs):
        """정상적으로 문서를 생성한다."""
        doc = create_document_source(db_session, **sample_doc_kwargs)

        assert doc.id == sample_doc_kwargs["id"]
        assert doc.source_type == "case"
        assert doc.title == "테스트 민원 사례"
        assert doc.content == "민원 내용 본문 텍스트"

    def test_create_sets_timestamps(self, db_session, sample_doc_kwargs):
        """생성 시 created_at, updated_at이 자동 설정된다."""
        doc = create_document_source(db_session, **sample_doc_kwargs)
        assert doc.created_at is not None
        assert doc.updated_at is not None


class TestGetDocumentSource:
    """get_document_source 테스트."""

    def test_get_existing(self, db_session, sample_doc):
        """존재하는 ID로 조회하면 문서를 반환한다."""
        fetched = get_document_source(db_session, sample_doc.id)
        assert fetched is not None
        assert fetched.id == sample_doc.id
        assert fetched.title == sample_doc.title

    def test_get_nonexistent(self, db_session):
        """존재하지 않는 ID로 조회하면 None을 반환한다."""
        result = get_document_source(db_session, uuid.uuid4())
        assert result is None


class TestGetDocumentSources:
    """get_document_sources 테스트."""

    def _create_docs(self, db_session, count=5, source_type="case", status="active"):
        """헬퍼: 여러 문서를 생성한다."""
        docs = []
        for i in range(count):
            uid = uuid.uuid4()
            doc = create_document_source(
                db_session,
                id=uid,
                source_type=source_type,
                source_id=f"DOC-{uid.hex[:8]}",
                title=f"문서 {i}",
                content=f"내용 {i}",
                chunk_index=0,
                total_chunks=1,
                reliability_score=0.7,
                status=status,
                version="1.0",
                metadata_={},
                embedding_version="e5-large-v1",
            )
            docs.append(doc)
        return docs

    def test_filter_by_source_type(self, db_session):
        """source_type으로 필터링한다."""
        self._create_docs(db_session, count=3, source_type="case")
        self._create_docs(db_session, count=2, source_type="law")

        cases = get_document_sources(db_session, filters={"source_type": "case"})
        laws = get_document_sources(db_session, filters={"source_type": "law"})

        assert len(cases) == 3
        assert len(laws) == 2
        assert all(d.source_type == "case" for d in cases)

    def test_filter_by_status(self, db_session):
        """status로 필터링한다."""
        self._create_docs(db_session, count=3, status="active")
        self._create_docs(db_session, count=1, status="expired")

        active = get_document_sources(db_session, filters={"status": "active"})
        expired = get_document_sources(db_session, filters={"status": "expired"})

        assert len(active) == 3
        assert len(expired) == 1

    def test_pagination(self, db_session):
        """skip/limit 페이지네이션을 검증한다."""
        self._create_docs(db_session, count=10)

        page1 = get_document_sources(db_session, skip=0, limit=3)
        page2 = get_document_sources(db_session, skip=3, limit=3)
        all_docs = get_document_sources(db_session, skip=0, limit=100)

        assert len(page1) == 3
        assert len(page2) == 3
        assert len(all_docs) == 10

    def test_no_filters(self, db_session):
        """필터 없이 전체 조회한다."""
        self._create_docs(db_session, count=5)
        docs = get_document_sources(db_session)
        assert len(docs) == 5


class TestUpdateDocumentSource:
    """update_document_source 테스트."""

    def test_update_fields(self, db_session, sample_doc):
        """필드를 수정한다."""
        updated = update_document_source(
            db_session, sample_doc.id, title="수정된 제목", reliability_score=0.95
        )
        assert updated is not None
        assert updated.title == "수정된 제목"
        assert updated.reliability_score == 0.95

    def test_update_nonexistent(self, db_session):
        """존재하지 않는 ID로 수정 시 None을 반환한다."""
        result = update_document_source(db_session, uuid.uuid4(), title="없는 문서")
        assert result is None


class TestDeleteDocumentSource:
    """delete_document_source 테스트."""

    def test_delete_existing(self, db_session, sample_doc):
        """존재하는 문서를 삭제하면 True를 반환한다."""
        doc_id = sample_doc.id
        result = delete_document_source(db_session, doc_id)
        assert result is True

        # 삭제 확인
        fetched = get_document_source(db_session, doc_id)
        assert fetched is None

    def test_delete_nonexistent(self, db_session):
        """존재하지 않는 ID로 삭제 시 False를 반환한다."""
        result = delete_document_source(db_session, uuid.uuid4())
        assert result is False


class TestGetBySourceTypeAndId:
    """get_by_source_type_and_id 테스트."""

    def test_multiple_chunks(self, db_session):
        """동일 source_type + source_id의 여러 청크를 chunk_index 순서로 반환한다."""
        for i in range(3):
            create_document_source(
                db_session,
                id=uuid.uuid4(),
                source_type="law",
                source_id="LAW-100",
                title="법령 문서",
                content=f"청크 {i} 내용",
                chunk_index=i,
                total_chunks=3,
                reliability_score=0.9,
                status="active",
                version="1.0",
                metadata_={},
                embedding_version="e5-large-v1",
            )

        chunks = get_by_source_type_and_id(db_session, "law", "LAW-100")
        assert len(chunks) == 3
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2

    def test_no_match(self, db_session):
        """매칭되는 문서가 없으면 빈 리스트를 반환한다."""
        result = get_by_source_type_and_id(db_session, "manual", "NONEXISTENT")
        assert result == []


# ============================================================================
# IndexingQueue CRUD
# ============================================================================


class TestCreateIndexingQueueItem:
    """create_indexing_queue_item 테스트."""

    def test_create_success(self, db_session, sample_doc):
        """정상적으로 큐 항목을 생성한다."""
        item = create_indexing_queue_item(
            db_session,
            id=uuid.uuid4(),
            document_id=sample_doc.id,
            doc_type="CASE",
            complaint_text="민원 텍스트",
            answer_text="답변 텍스트",
            status="pending",
            priority=3,
        )
        assert item.status == "pending"
        assert item.priority == 3
        assert item.document_id == sample_doc.id


class TestGetPendingItems:
    """get_pending_items 테스트."""

    def test_priority_order(self, db_session, sample_doc):
        """pending 항목을 우선순위 내림차순으로 반환한다."""
        for priority in [1, 5, 3]:
            create_indexing_queue_item(
                db_session,
                id=uuid.uuid4(),
                document_id=sample_doc.id,
                doc_type="CASE",
                complaint_text="민원",
                answer_text="답변",
                status="pending",
                priority=priority,
            )
        # completed 상태는 포함되지 않아야 함
        create_indexing_queue_item(
            db_session,
            id=uuid.uuid4(),
            document_id=sample_doc.id,
            doc_type="CASE",
            complaint_text="완료된 민원",
            answer_text="완료된 답변",
            status="completed",
            priority=10,
        )

        pending = get_pending_items(db_session)
        assert len(pending) == 3
        assert pending[0].priority == 5
        assert pending[1].priority == 3
        assert pending[2].priority == 1


class TestUpdateQueueStatus:
    """update_queue_status 테스트."""

    def test_completed_sets_processed_at(self, db_session, sample_doc):
        """completed로 변경 시 processed_at이 자동 설정된다."""
        item = create_indexing_queue_item(
            db_session,
            id=uuid.uuid4(),
            document_id=sample_doc.id,
            doc_type="CASE",
            complaint_text="민원",
            answer_text="답변",
            status="pending",
            priority=0,
        )
        assert item.processed_at is None

        updated = update_queue_status(db_session, item.id, "completed")
        assert updated is not None
        assert updated.status == "completed"
        assert updated.processed_at is not None

    def test_skipped_sets_skip_reason(self, db_session, sample_doc):
        """skipped로 변경 시 skip_reason이 설정된다."""
        item = create_indexing_queue_item(
            db_session,
            id=uuid.uuid4(),
            document_id=sample_doc.id,
            doc_type="CASE",
            complaint_text="민원",
            answer_text="답변",
            status="pending",
            priority=0,
        )

        updated = update_queue_status(
            db_session, item.id, "skipped", skip_reason="중복 문서"
        )
        assert updated is not None
        assert updated.status == "skipped"
        assert updated.skip_reason == "중복 문서"
        assert updated.processed_at is not None

    def test_update_nonexistent(self, db_session):
        """존재하지 않는 항목은 None을 반환한다."""
        result = update_queue_status(db_session, uuid.uuid4(), "completed")
        assert result is None


class TestGetQueueStats:
    """get_queue_stats 테스트."""

    def test_stats_aggregation(self, db_session, sample_doc):
        """상태별 건수를 정확히 집계한다."""
        statuses = ["pending", "pending", "pending", "completed", "completed", "failed"]
        for i, status in enumerate(statuses):
            create_indexing_queue_item(
                db_session,
                id=uuid.uuid4(),
                document_id=sample_doc.id,
                doc_type="CASE",
                complaint_text=f"민원 {i}",
                answer_text=f"답변 {i}",
                status=status,
                priority=0,
            )

        stats = get_queue_stats(db_session)
        assert stats["pending"] == 3
        assert stats["completed"] == 2
        assert stats["failed"] == 1

    def test_empty_stats(self, db_session):
        """큐가 비어있으면 빈 딕셔너리를 반환한다."""
        stats = get_queue_stats(db_session)
        assert stats == {}


# ============================================================================
# IndexVersion CRUD
# ============================================================================


class TestCreateIndexVersion:
    """create_index_version 테스트."""

    def test_create_success(self, db_session, sample_index_version_kwargs):
        """정상적으로 인덱스 버전을 생성한다."""
        ver = create_index_version(db_session, **sample_index_version_kwargs)
        assert ver.index_type == "case"
        assert ver.version == "v1.0.0"
        assert ver.total_documents == 100
        assert ver.is_active is True


class TestGetActiveVersion:
    """get_active_version 테스트."""

    def test_get_active(self, db_session, sample_index_version_kwargs):
        """활성 버전을 조회한다."""
        create_index_version(db_session, **sample_index_version_kwargs)

        active = get_active_version(db_session, "case")
        assert active is not None
        assert active.is_active is True
        assert active.index_type == "case"

    def test_no_active(self, db_session):
        """활성 버전이 없으면 None을 반환한다."""
        result = get_active_version(db_session, "case")
        assert result is None


class TestDeactivateVersions:
    """deactivate_versions 테스트."""

    def test_deactivate(self, db_session):
        """특정 index_type의 모든 활성 버전을 비활성화한다."""
        for i in range(3):
            create_index_version(
                db_session,
                id=uuid.uuid4(),
                index_type="case",
                version=f"v{i}.0.0",
                total_documents=100 * (i + 1),
                index_file_path=f"/data/faiss/case/index_{i}.faiss",
                meta_file_path=f"/data/faiss/case/metadata_{i}.json",
                is_active=True,
            )

        count = deactivate_versions(db_session, "case")
        assert count == 3

        active = get_active_version(db_session, "case")
        assert active is None

    def test_deactivate_does_not_affect_other_type(self, db_session):
        """다른 index_type의 버전은 영향받지 않는다."""
        create_index_version(
            db_session,
            id=uuid.uuid4(),
            index_type="case",
            version="v1.0.0",
            total_documents=100,
            index_file_path="/data/case/index.faiss",
            meta_file_path="/data/case/metadata.json",
            is_active=True,
        )
        create_index_version(
            db_session,
            id=uuid.uuid4(),
            index_type="law",
            version="v1.0.0",
            total_documents=50,
            index_file_path="/data/law/index.faiss",
            meta_file_path="/data/law/metadata.json",
            is_active=True,
        )

        deactivate_versions(db_session, "case")

        law_active = get_active_version(db_session, "law")
        assert law_active is not None
        assert law_active.is_active is True


class TestActivateVersion:
    """activate_version 테스트."""

    def test_activate_deactivates_existing(self, db_session):
        """활성화 시 동일 타입의 기존 활성 버전이 자동 비활성화된다."""
        ver1 = create_index_version(
            db_session,
            id=uuid.uuid4(),
            index_type="case",
            version="v1.0.0",
            total_documents=100,
            index_file_path="/data/case/v1/index.faiss",
            meta_file_path="/data/case/v1/metadata.json",
            is_active=True,
        )
        ver2 = create_index_version(
            db_session,
            id=uuid.uuid4(),
            index_type="case",
            version="v2.0.0",
            total_documents=200,
            index_file_path="/data/case/v2/index.faiss",
            meta_file_path="/data/case/v2/metadata.json",
            is_active=False,
        )

        # ver2 활성화 -> ver1 자동 비활성화
        activated = activate_version(db_session, ver2.id)
        assert activated is not None
        assert activated.is_active is True
        assert activated.version == "v2.0.0"

        # ver1이 비활성화되었는지 확인
        db_session.refresh(ver1)
        assert ver1.is_active is False

        # 활성 버전이 ver2 하나뿐인지 확인
        active = get_active_version(db_session, "case")
        assert active.id == ver2.id

    def test_activate_nonexistent(self, db_session):
        """존재하지 않는 ID로 활성화 시 None을 반환한다."""
        result = activate_version(db_session, uuid.uuid4())
        assert result is None
