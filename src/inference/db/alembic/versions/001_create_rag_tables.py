"""RAG 핵심 테이블 생성: document_source, indexing_queue, index_version

ADR-004 Section D 스키마와 Issue #152 요구사항을 병합한 초기 마이그레이션.
ORM 모델(models.py)과 정합성을 유지한다.

Revision ID: 001
Revises: None
Create Date: 2026-03-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. document_source 테이블
    # ------------------------------------------------------------------
    op.create_table(
        "document_source",
        # PK
        sa.Column(
            "id",
            UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
            comment="문서 고유 식별자",
        ),
        # 공통 필드
        sa.Column(
            "source_type",
            sa.String(20),
            nullable=False,
            comment="문서 타입: case, law, manual, notice",
        ),
        sa.Column(
            "source_id",
            sa.String(255),
            nullable=False,
            comment="원본 문서 식별자",
        ),
        sa.Column(
            "source_name",
            sa.String(200),
            nullable=True,
            comment="출처명 (AI Hub, 법제처 등)",
        ),
        sa.Column("title", sa.String(500), nullable=False, comment="문서 제목"),
        sa.Column("content", sa.Text, nullable=False, comment="문서 본문"),
        sa.Column("category", sa.String(50), nullable=True, comment="카테고리"),
        sa.Column(
            "chunk_index",
            sa.Integer,
            server_default="0",
            comment="청크 인덱스",
        ),
        sa.Column(
            "total_chunks",
            sa.Integer,
            server_default="1",
            comment="전체 청크 수",
        ),
        sa.Column(
            "reliability_score",
            sa.Float,
            server_default="0.6",
            comment="신뢰도 점수 (0.0~1.0)",
        ),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True, comment="유효 시작일"),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True, comment="유효 종료일"),
        sa.Column(
            "status",
            sa.String(20),
            server_default=sa.text("'active'"),
            comment="문서 상태: active, expired, deprecated",
        ),
        sa.Column(
            "version",
            sa.String(20),
            server_default=sa.text("'1.0'"),
            comment="문서 버전",
        ),
        # 확장 메타데이터 (JSONB)
        sa.Column(
            "metadata",
            JSONB,
            server_default=sa.text("'{}'::jsonb"),
            comment="추가 메타데이터 (JSONB)",
        ),
        # CASE 타입 전용
        sa.Column("complaint_text", sa.Text, nullable=True, comment="민원 텍스트 (CASE 전용)"),
        sa.Column("answer_text", sa.Text, nullable=True, comment="답변 텍스트 (CASE 전용)"),
        # LAW 타입 전용
        sa.Column("law_number", sa.String(100), nullable=True, comment="법률 번호 (LAW 전용)"),
        sa.Column("article_number", sa.String(50), nullable=True, comment="조항 번호 (LAW 전용)"),
        sa.Column("enforcement_date", sa.Date, nullable=True, comment="시행일 (LAW 전용)"),
        # MANUAL 타입 전용
        sa.Column("department", sa.String(100), nullable=True, comment="담당 부서 (MANUAL 전용)"),
        # NOTICE 타입 전용
        sa.Column(
            "notice_number", sa.String(100), nullable=True, comment="공시 번호 (NOTICE 전용)"
        ),
        sa.Column("effective_date", sa.Date, nullable=True, comment="시행일 (NOTICE 전용)"),
        # 인덱싱 관련
        sa.Column("faiss_index_id", sa.Integer, nullable=True, comment="FAISS 인덱스 내 ID"),
        sa.Column(
            "embedding_version",
            sa.String(50),
            server_default=sa.text("'e5-large-v1'"),
            comment="임베딩 모델 버전",
        ),
        # 타임스탬프
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            comment="생성 시각",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            comment="수정 시각",
        ),
        # UNIQUE 제약조건
        sa.UniqueConstraint(
            "source_type",
            "source_id",
            "chunk_index",
            name="uq_source_type_source_id_chunk",
        ),
        # CHECK 제약조건
        sa.CheckConstraint(
            "source_type IN ('case', 'law', 'manual', 'notice')",
            name="ck_source_type_valid",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'expired', 'deprecated')",
            name="ck_status_valid",
        ),
        sa.CheckConstraint(
            "reliability_score >= 0.0 AND reliability_score <= 1.0",
            name="ck_reliability_score_range",
        ),
    )

    # document_source B-tree 인덱스
    op.create_index("idx_docsource_source_type", "document_source", ["source_type"])
    op.create_index("idx_docsource_status", "document_source", ["status"])
    op.create_index("idx_docsource_category", "document_source", ["category"])
    op.create_index("idx_docsource_valid_range", "document_source", ["valid_from", "valid_until"])

    # document_source GIN 인덱스 - JSONB 검색용
    op.create_index(
        "idx_docsource_metadata",
        "document_source",
        ["metadata"],
        postgresql_using="gin",
    )

    # ------------------------------------------------------------------
    # 2. indexing_queue 테이블
    # ------------------------------------------------------------------
    op.create_table(
        "indexing_queue",
        # PK
        sa.Column(
            "id",
            UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
            comment="큐 항목 고유 식별자",
        ),
        # document_source FK
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("document_source.id", ondelete="SET NULL"),
            nullable=True,
            comment="연결된 문서 원본 ID",
        ),
        # 참조 필드 (FK 없는 UUID - 참조 테이블 미존재)
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            nullable=True,
            comment="상담 세션 ID (FK 없음)",
        ),
        sa.Column(
            "message_id",
            UUID(as_uuid=True),
            nullable=True,
            comment="메시지 ID (FK 없음)",
        ),
        # 큐 데이터
        sa.Column(
            "doc_type",
            sa.String(20),
            server_default=sa.text("'CASE'"),
            comment="문서 타입",
        ),
        sa.Column("complaint_text", sa.Text, nullable=False, comment="민원 텍스트"),
        sa.Column("answer_text", sa.Text, nullable=False, comment="답변 텍스트"),
        sa.Column("category", sa.String(50), nullable=True, comment="카테고리"),
        sa.Column(
            "status",
            sa.String(20),
            server_default=sa.text("'pending'"),
            comment="처리 상태: pending, processing, completed, skipped, failed",
        ),
        sa.Column(
            "priority",
            sa.Integer,
            server_default="0",
            comment="우선순위 (높을수록 먼저)",
        ),
        sa.Column("skip_reason", sa.String(200), nullable=True, comment="건너뛰기 사유"),
        # 타임스탬프
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            comment="생성 시각",
        ),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="처리 완료 시각",
        ),
        # CHECK 제약조건
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'skipped', 'failed')",
            name="ck_queue_status_valid",
        ),
    )

    # indexing_queue 인덱스
    op.create_index("idx_indexqueue_status", "indexing_queue", ["status"])
    op.create_index("idx_indexqueue_priority", "indexing_queue", ["priority", "created_at"])
    op.create_index("idx_indexqueue_document_id", "indexing_queue", ["document_id"])

    # ------------------------------------------------------------------
    # 3. index_version 테이블
    # ------------------------------------------------------------------
    op.create_table(
        "index_version",
        # PK
        sa.Column(
            "id",
            UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
            comment="인덱스 버전 고유 식별자",
        ),
        # 인덱스 정보
        sa.Column(
            "index_type",
            sa.String(20),
            nullable=False,
            comment="인덱스 타입 (case, law, manual, notice)",
        ),
        sa.Column("version", sa.String(50), nullable=False, comment="인덱스 버전"),
        sa.Column("total_documents", sa.Integer, nullable=False, comment="포함 문서 수"),
        sa.Column(
            "index_file_path",
            sa.String(500),
            nullable=False,
            comment="FAISS 인덱스 파일 경로",
        ),
        sa.Column(
            "meta_file_path",
            sa.String(500),
            nullable=False,
            comment="메타데이터 파일 경로",
        ),
        sa.Column(
            "snapshot_path",
            sa.Text,
            nullable=True,
            comment="스냅샷 경로",
        ),
        # 상태
        sa.Column(
            "built_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            comment="빌드 시각",
        ),
        sa.Column(
            "is_active",
            sa.Boolean,
            server_default=sa.text("true"),
            comment="활성 버전 여부",
        ),
        sa.Column(
            "build_duration_seconds",
            sa.Float,
            nullable=True,
            comment="빌드 소요 시간 (초)",
        ),
        sa.Column("notes", sa.Text, nullable=True, comment="비고"),
    )

    # index_version 인덱스
    op.create_index("idx_indexversion_active", "index_version", ["index_type", "is_active"])

    # ------------------------------------------------------------------
    # 4. updated_at 자동 갱신 트리거 (document_source)
    # ------------------------------------------------------------------
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER trg_docsource_updated_at
        BEFORE UPDATE ON document_source
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    # 트리거 및 함수 제거
    op.execute("DROP TRIGGER IF EXISTS trg_docsource_updated_at ON document_source;")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")

    # 테이블 제거 (의존성 역순)
    op.drop_table("index_version")
    op.drop_table("indexing_queue")
    op.drop_table("document_source")
