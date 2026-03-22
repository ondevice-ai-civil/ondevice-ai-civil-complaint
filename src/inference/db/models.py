"""
SQLAlchemy 2.0 ORM 모델.

ADR-004 + Issue #152 병합 스키마 기반 3개 테이블:
- DocumentSource  : 문서 원본 메타데이터
- IndexingQueue   : 인덱싱 대기열
- IndexVersion    : FAISS 인덱스 버전 관리

모든 모델은 SQLAlchemy 2.0 Mapped 스타일(mapped_column, Mapped)을 사용하며,
server_default로 DB 레벨 기본값을 지정한다.
"""

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# ---------------------------------------------------------------------------
# Base 클래스
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """모든 ORM 모델의 공통 기반 클래스."""
    pass


# ---------------------------------------------------------------------------
# DocumentSource
# ---------------------------------------------------------------------------


class DocumentSource(Base):
    """document_source 테이블 ORM 모델.

    모든 문서 타입(case, law, manual, notice)의 메타데이터를 통합 관리한다.
    타입별 전용 컬럼은 nullable로 처리하며, 해당하지 않는 타입에서는 NULL이다.
    """

    __tablename__ = "document_source"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "source_id", "chunk_index",
            name="uq_source_type_source_id_chunk",
        ),
        CheckConstraint(
            "source_type IN ('case', 'law', 'manual', 'notice')",
            name="ck_source_type_valid",
        ),
        CheckConstraint(
            "status IN ('active', 'expired', 'deprecated')",
            name="ck_status_valid",
        ),
        CheckConstraint(
            "reliability_score >= 0.0 AND reliability_score <= 1.0",
            name="ck_reliability_score_range",
        ),
        # 성능 인덱스
        Index("idx_docsource_source_type", "source_type"),
        Index("idx_docsource_status", "status"),
        Index("idx_docsource_category", "category"),
        Index("idx_docsource_valid_range", "valid_from", "valid_until"),
        Index("idx_docsource_metadata", "metadata", postgresql_using="gin"),
    )

    # -- 기본 키 --
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        comment="문서 고유 식별자",
    )

    # -- 공통 필드 --
    source_type: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="문서 타입: case, law, manual, notice",
    )
    source_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
        comment="원본 문서 식별자",
    )
    source_name: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True,
        comment="출처명 (AI Hub, 법제처 등)",
    )
    title: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="문서 제목",
    )
    content: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="문서 본문",
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
        comment="카테고리 (도로/교통, 환경/위생 등)",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, server_default=text("0"),
        comment="청크 인덱스",
    )
    total_chunks: Mapped[int] = mapped_column(
        Integer, server_default=text("1"),
        comment="전체 청크 수",
    )
    reliability_score: Mapped[float] = mapped_column(
        Float, server_default=text("0.6"),
        comment="신뢰도 점수 (0.0~1.0)",
    )
    valid_from: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="유효 시작일",
    )
    valid_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="유효 종료일",
    )
    status: Mapped[str] = mapped_column(
        String(20), server_default=text("'active'"),
        comment="문서 상태: active, expired, deprecated",
    )
    version: Mapped[str] = mapped_column(
        String(20), server_default=text("'1.0'"),
        comment="문서 버전",
    )
    # 'metadata'는 SQLAlchemy 내부 예약어이므로 Python 속성은 metadata_로 매핑
    metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb"),
        comment="추가 메타데이터 (JSONB)",
    )

    # -- CASE 전용 --
    complaint_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="민원 텍스트 (CASE 전용)",
    )
    answer_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="답변 텍스트 (CASE 전용)",
    )

    # -- LAW 전용 --
    law_number: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        comment="법률 번호 (LAW 전용)",
    )
    article_number: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
        comment="조항 번호 (LAW 전용)",
    )
    enforcement_date: Mapped[Optional[date]] = mapped_column(
        Date, nullable=True,
        comment="시행일 (LAW 전용)",
    )

    # -- MANUAL 전용 --
    department: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        comment="담당 부서 (MANUAL 전용)",
    )

    # -- NOTICE 전용 --
    notice_number: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        comment="공시 번호 (NOTICE 전용)",
    )
    effective_date: Mapped[Optional[date]] = mapped_column(
        Date, nullable=True,
        comment="시행일 (NOTICE 전용)",
    )

    # -- 인덱싱 관련 --
    faiss_index_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="FAISS 인덱스 내 ID",
    )
    embedding_version: Mapped[str] = mapped_column(
        String(50), server_default=text("'e5-large-v1'"),
        comment="임베딩 모델 버전",
    )

    # -- 타임스탬프 --
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="생성 시각",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(), onupdate=func.now(),
        comment="수정 시각",
    )

    # -- 관계 --
    queue_items: Mapped[List["IndexingQueue"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentSource(id={self.id}, "
            f"type={self.source_type}, title={self.title!r})>"
        )


# ---------------------------------------------------------------------------
# IndexingQueue
# ---------------------------------------------------------------------------


class IndexingQueue(Base):
    """indexing_queue 테이블 ORM 모델.

    새로운 민원 상담이 들어오면 인덱싱 대기열에 추가되며,
    배치 프로세스가 주기적으로 pending 항목을 소비한다.
    """

    __tablename__ = "indexing_queue"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'skipped', 'failed')",
            name="ck_queue_status_valid",
        ),
        # 성능 인덱스
        Index("idx_indexqueue_status", "status"),
        Index("idx_indexqueue_priority", "priority", "created_at"),
        Index("idx_indexqueue_document_id", "document_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        comment="큐 항목 고유 식별자",
    )
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_source.id", ondelete="SET NULL"),
        nullable=True,
        comment="연결된 문서 원본 ID",
    )
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True,
        comment="상담 세션 ID (FK 없음)",
    )
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True,
        comment="메시지 ID (FK 없음)",
    )
    doc_type: Mapped[str] = mapped_column(
        String(20), server_default=text("'CASE'"),
        comment="문서 타입",
    )
    complaint_text: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="민원 텍스트",
    )
    answer_text: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="답변 텍스트",
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
        comment="카테고리",
    )
    status: Mapped[str] = mapped_column(
        String(20), server_default=text("'pending'"),
        comment="처리 상태: pending, processing, completed, skipped, failed",
    )
    priority: Mapped[int] = mapped_column(
        Integer, server_default=text("0"),
        comment="우선순위 (높을수록 먼저)",
    )
    skip_reason: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True,
        comment="건너뛰기 사유",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="생성 시각",
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="처리 완료 시각",
    )

    # -- 관계 --
    document: Mapped[Optional["DocumentSource"]] = relationship(
        back_populates="queue_items", lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<IndexingQueue(id={self.id}, "
            f"status={self.status}, doc_type={self.doc_type})>"
        )


# ---------------------------------------------------------------------------
# IndexVersion
# ---------------------------------------------------------------------------


class IndexVersion(Base):
    """index_version 테이블 ORM 모델.

    FAISS 인덱스 빌드 이력을 관리하며,
    index_type별로 하나의 active 버전만 유지한다.
    """

    __tablename__ = "index_version"
    __table_args__ = (
        CheckConstraint(
            "index_type IN ('case', 'law', 'manual', 'notice')",
            name="ck_index_type_valid",
        ),
        Index("idx_indexversion_active", "index_type", "is_active"),
        Index(
            "uq_indexversion_one_active_per_type",
            "index_type",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        comment="인덱스 버전 고유 식별자",
    )
    index_type: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="인덱스 타입 (case, law, manual, notice)",
    )
    version: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="인덱스 버전 (예: v1.0.0)",
    )
    total_documents: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="포함 문서 수",
    )
    index_file_path: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="FAISS 인덱스 파일 경로",
    )
    meta_file_path: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="메타데이터 파일 경로",
    )
    snapshot_path: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="스냅샷 경로",
    )
    built_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="빌드 시각",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, server_default=text("true"),
        comment="활성 버전 여부",
    )
    build_duration_seconds: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="빌드 소요 시간 (초)",
    )
    notes: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="비고",
    )

    def __repr__(self) -> str:
        return (
            f"<IndexVersion(id={self.id}, "
            f"type={self.index_type}, version={self.version}, "
            f"active={self.is_active})>"
        )
