"""
CRUD 레이어 (Unit of Work 패턴).

DocumentSource, IndexingQueue, IndexVersion 테이블에 대한
생성/조회/수정/삭제 함수를 제공한다.
모든 함수는 동기 Session을 인자로 받는다.

이 모듈의 함수들은 내부에서 commit을 수행하지 않는다.
트랜잭션의 commit/rollback 제어는 caller(서비스 계층)의 책임이다.
복합 작업의 원자성을 보장하기 위해 flush만 수행하여 DB에 SQL을 전송하되,
최종 확정은 caller가 결정한다.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from src.inference.db.models import DocumentSource, IndexingQueue, IndexVersion

# ---------------------------------------------------------------------------
# 상수 정의
# ---------------------------------------------------------------------------

MAX_LIMIT = 1000

_ALLOWED_FILTER_COLUMNS = frozenset(
    {
        "source_type",
        "source_id",
        "status",
        "category",
        "source_name",
        "embedding_version",
        "version",
    }
)

_IMMUTABLE_FIELDS = frozenset({"id", "created_at"})

_VALID_QUEUE_STATUSES = frozenset(
    {
        "pending",
        "processing",
        "completed",
        "skipped",
        "failed",
    }
)


# ============================================================================
# DocumentSource CRUD
# ============================================================================


def create_document_source(db: Session, **kwargs: Any) -> DocumentSource:
    """새 문서 원본 레코드를 생성한다."""
    doc = DocumentSource(**kwargs)
    db.add(doc)
    db.flush()
    db.refresh(doc)
    return doc


def get_document_source(db: Session, doc_id: uuid.UUID) -> Optional[DocumentSource]:
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
    limit = min(limit, MAX_LIMIT)
    stmt = select(DocumentSource)

    if filters:
        for col_name, value in filters.items():
            if col_name in _ALLOWED_FILTER_COLUMNS:
                stmt = stmt.where(getattr(DocumentSource, col_name) == value)

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
        if key in _IMMUTABLE_FIELDS:
            continue
        if hasattr(doc, key):
            setattr(doc, key, value)

    db.flush()
    db.refresh(doc)
    return doc


def delete_document_source(db: Session, doc_id: uuid.UUID) -> bool:
    """문서 원본 레코드를 삭제한다. 성공 시 True 반환."""
    doc = db.get(DocumentSource, doc_id)
    if doc is None:
        return False

    db.delete(doc)
    db.flush()
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
    db.flush()
    db.refresh(item)
    return item


def get_pending_items(db: Session, limit: int = 50) -> List[IndexingQueue]:
    """pending 상태의 대기열 항목을 우선순위 내림차순으로 조회한다."""
    limit = min(limit, MAX_LIMIT)
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
    if status not in _VALID_QUEUE_STATUSES:
        raise ValueError(
            f"유효하지 않은 상태: {status!r}. "
            f"허용 값: {', '.join(sorted(_VALID_QUEUE_STATUSES))}"
        )

    item = db.get(IndexingQueue, item_id)
    if item is None:
        return None

    item.status = status
    if skip_reason is not None:
        item.skip_reason = skip_reason

    if status in ("completed", "failed", "skipped"):
        item.processed_at = datetime.now(timezone.utc)

    db.flush()
    db.refresh(item)
    return item


def get_queue_stats(db: Session) -> Dict[str, int]:
    """대기열 상태별 건수를 집계한다.

    Returns
    -------
    dict
        {"pending": 10, "processing": 2, "completed": 50, ...}
    """
    stmt = select(IndexingQueue.status, func.count()).group_by(IndexingQueue.status)
    rows = db.execute(stmt).all()
    return {status: count for status, count in rows}


# ============================================================================
# IndexVersion CRUD
# ============================================================================


def create_index_version(db: Session, **kwargs: Any) -> IndexVersion:
    """새 인덱스 버전 레코드를 생성한다."""
    ver = IndexVersion(**kwargs)
    db.add(ver)
    db.flush()
    db.refresh(ver)
    return ver


def get_active_version(db: Session, index_type: str) -> Optional[IndexVersion]:
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
    db.flush()
    return result.rowcount  # type: ignore[return-value]


def activate_version(db: Session, version_id: uuid.UUID) -> Optional[IndexVersion]:
    """특정 인덱스 버전을 활성화한다.

    동일 index_type의 기존 활성 버전을 먼저 비활성화한 뒤 대상을 활성화한다.

    Race Condition 방지:
        SELECT ... FOR UPDATE로 동일 index_type의 모든 버전에 행 레벨 잠금을
        획득한 뒤 deactivate/activate를 수행한다. 동시 호출 시 후발 트랜잭션은
        잠금 해제까지 대기하므로 다중 active 버전이 생기는 문제를 방지한다.
        (PostgreSQL 전용 — SQLite는 FOR UPDATE를 지원하지 않는다.)
    """
    ver = db.get(IndexVersion, version_id)
    if ver is None:
        return None

    # 동일 index_type의 모든 버전에 대해 행 레벨 잠금 획득 (PostgreSQL 전용)
    lock_stmt = (
        select(IndexVersion).where(IndexVersion.index_type == ver.index_type).with_for_update()
    )
    db.execute(lock_stmt)

    # 잠금 획득 후 동일 타입의 기존 활성 버전 비활성화
    deactivate_versions(db, ver.index_type)

    ver.is_active = True
    db.flush()
    db.refresh(ver)
    return ver
