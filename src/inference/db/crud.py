"""
CRUD 레이어.

DocumentSource, IndexingQueue, IndexVersion 테이블에 대한
생성/조회/수정/삭제 함수를 제공한다.
모든 함수는 동기 Session을 인자로 받는다.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from src.inference.db.models import DocumentSource, IndexingQueue, IndexVersion


# ============================================================================
# DocumentSource CRUD
# ============================================================================


def create_document_source(db: Session, **kwargs: Any) -> DocumentSource:
    """새 문서 원본 레코드를 생성한다."""
    doc = DocumentSource(**kwargs)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document_source(
    db: Session, doc_id: uuid.UUID
) -> Optional[DocumentSource]:
    """ID로 문서 원본을 조회한다."""
    return db.get(DocumentSource, doc_id)


def get_document_sources(
    db: Session,
    filters: Optional[Dict[str, Any]] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[DocumentSource]:
    """필터 조건에 맞는 문서 원본 목록을 조회한다.

    Parameters
    ----------
    filters : dict, optional
        컬럼명-값 쌍의 필터 딕셔너리.
        예: {"source_type": "case", "status": "active"}
    skip : int
        건너뛸 행 수 (페이지네이션 오프셋).
    limit : int
        최대 반환 행 수.
    """
    stmt = select(DocumentSource)

    if filters:
        for col_name, value in filters.items():
            if hasattr(DocumentSource, col_name):
                stmt = stmt.where(
                    getattr(DocumentSource, col_name) == value
                )

    stmt = stmt.offset(skip).limit(limit).order_by(DocumentSource.created_at.desc())
    return list(db.scalars(stmt).all())


def update_document_source(
    db: Session, doc_id: uuid.UUID, **kwargs: Any
) -> Optional[DocumentSource]:
    """문서 원본 레코드를 수정한다.

    변경할 컬럼-값을 kwargs로 전달한다.
    """
    doc = db.get(DocumentSource, doc_id)
    if doc is None:
        return None

    for key, value in kwargs.items():
        if hasattr(doc, key):
            setattr(doc, key, value)

    db.commit()
    db.refresh(doc)
    return doc


def delete_document_source(db: Session, doc_id: uuid.UUID) -> bool:
    """문서 원본 레코드를 삭제한다. 성공 시 True 반환."""
    doc = db.get(DocumentSource, doc_id)
    if doc is None:
        return False

    db.delete(doc)
    db.commit()
    return True


def get_by_source_type_and_id(
    db: Session, source_type: str, source_id: str
) -> List[DocumentSource]:
    """source_type + source_id 조합으로 문서를 조회한다.

    동일 문서의 여러 청크가 반환될 수 있으므로 리스트를 반환한다.
    """
    stmt = (
        select(DocumentSource)
        .where(
            DocumentSource.source_type == source_type,
            DocumentSource.source_id == source_id,
        )
        .order_by(DocumentSource.chunk_index)
    )
    return list(db.scalars(stmt).all())


# ============================================================================
# IndexingQueue CRUD
# ============================================================================


def create_indexing_queue_item(db: Session, **kwargs: Any) -> IndexingQueue:
    """인덱싱 대기열에 새 항목을 추가한다."""
    item = IndexingQueue(**kwargs)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def get_pending_items(db: Session, limit: int = 50) -> List[IndexingQueue]:
    """pending 상태의 대기열 항목을 우선순위 내림차순으로 조회한다."""
    stmt = (
        select(IndexingQueue)
        .where(IndexingQueue.status == "pending")
        .order_by(IndexingQueue.priority.desc(), IndexingQueue.created_at)
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


def update_queue_status(
    db: Session,
    item_id: uuid.UUID,
    status: str,
    skip_reason: Optional[str] = None,
) -> Optional[IndexingQueue]:
    """대기열 항목의 상태를 변경한다.

    completed/failed 상태로 변경 시 processed_at을 자동 설정한다.
    """
    item = db.get(IndexingQueue, item_id)
    if item is None:
        return None

    item.status = status
    if skip_reason is not None:
        item.skip_reason = skip_reason

    if status in ("completed", "failed", "skipped"):
        item.processed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(item)
    return item


def get_queue_stats(db: Session) -> Dict[str, int]:
    """대기열 상태별 건수를 집계한다.

    Returns
    -------
    dict
        {"pending": 10, "processing": 2, "completed": 50, ...}
    """
    stmt = (
        select(IndexingQueue.status, func.count())
        .group_by(IndexingQueue.status)
    )
    rows = db.execute(stmt).all()
    return {status: count for status, count in rows}


# ============================================================================
# IndexVersion CRUD
# ============================================================================


def create_index_version(db: Session, **kwargs: Any) -> IndexVersion:
    """새 인덱스 버전 레코드를 생성한다."""
    ver = IndexVersion(**kwargs)
    db.add(ver)
    db.commit()
    db.refresh(ver)
    return ver


def get_active_version(
    db: Session, index_type: str
) -> Optional[IndexVersion]:
    """특정 index_type의 활성 버전을 조회한다.

    index_type별로 active 버전은 최대 1개여야 한다.
    """
    stmt = (
        select(IndexVersion)
        .where(
            IndexVersion.index_type == index_type,
            IndexVersion.is_active.is_(True),
        )
        .order_by(IndexVersion.built_at.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()


def deactivate_versions(db: Session, index_type: str) -> int:
    """특정 index_type의 모든 활성 버전을 비활성화한다.

    새 인덱스를 활성화하기 전에 호출하여 단일 활성 버전을 보장한다.

    Returns
    -------
    int
        비활성화된 레코드 수.
    """
    stmt = (
        update(IndexVersion)
        .where(
            IndexVersion.index_type == index_type,
            IndexVersion.is_active.is_(True),
        )
        .values(is_active=False)
    )
    result = db.execute(stmt)
    db.commit()
    return result.rowcount  # type: ignore[return-value]


def activate_version(
    db: Session, version_id: uuid.UUID
) -> Optional[IndexVersion]:
    """특정 인덱스 버전을 활성화한다.

    동일 index_type의 기존 활성 버전을 먼저 비활성화한 뒤 대상을 활성화한다.
    """
    ver = db.get(IndexVersion, version_id)
    if ver is None:
        return None

    # 동일 타입의 기존 활성 버전 비활성화
    deactivate_versions(db, ver.index_type)

    ver.is_active = True
    db.commit()
    db.refresh(ver)
    return ver
