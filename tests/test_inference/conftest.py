"""
테스트 공통 fixture.

SQLite 인메모리 DB를 사용하여 PostgreSQL 없이 ORM/CRUD를 검증한다.
- JSONB -> JSON 타입으로 폴백 (SQLite 호환)
- UUID -> CHAR(32) TypeDecorator (SQLite에서 UUID를 hex 문자열로 저장)
- gen_random_uuid() server_default -> 테스트에서 id 직접 전달

database.py가 모듈 레벨에서 PostgreSQL 엔진을 생성하므로,
테스트 환경에서는 해당 모듈을 mock으로 미리 등록하여 psycopg2 의존성을 우회한다.
"""

import sys
import types
import uuid as _uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

# database.py 모듈 레벨의 PostgreSQL engine 생성을 우회
_mock_database = types.ModuleType("src.inference.db.database")
_mock_database.engine = MagicMock()
_mock_database.SessionLocal = MagicMock()
_mock_database.get_db = MagicMock()
sys.modules["src.inference.db.database"] = _mock_database

# faiss 모듈이 설치되지 않은 환경에서도 DB 테스트가 동작하도록 mock 등록
# 이미 실제 faiss가 로드된 경우에는 mock하지 않는다
_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, MagicMock)
if not _faiss_is_real:
    _faiss_mock = MagicMock()
    # index_manager._maybe_upgrade_to_ivf()에서 isinstance 호출이 동작하도록 타입 설정
    _faiss_mock.IndexIVFFlat = type("IndexIVFFlat", (), {})
    _faiss_mock.IndexFlatIP = type("IndexFlatIP", (), {})
    sys.modules["faiss"] = _faiss_mock

import pytest
from sqlalchemy import CHAR, JSON, TypeDecorator, create_engine, event
from sqlalchemy.orm import sessionmaker

from src.inference.db.models import Base, DocumentSource, IndexingQueue, IndexVersion

# ---------------------------------------------------------------------------
# SQLite 호환 UUID TypeDecorator
# ---------------------------------------------------------------------------


class SQLiteUUID(TypeDecorator):
    """SQLite에서 UUID를 CHAR(32) hex 문자열로 저장하는 TypeDecorator."""

    impl = CHAR(32)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            if isinstance(value, _uuid.UUID):
                return value.hex
            return _uuid.UUID(value).hex
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return _uuid.UUID(value)
        return value


# ---------------------------------------------------------------------------
# PostgreSQL 전용 타입/옵션을 SQLite 호환으로 변환
# ---------------------------------------------------------------------------


def _patch_columns_for_sqlite():
    """PostgreSQL 전용 타입을 SQLite 호환 타입으로 변환한다."""
    from sqlalchemy.dialects.postgresql import JSONB, UUID

    for table in Base.metadata.tables.values():
        for column in table.columns:
            # JSONB -> JSON 변환
            if isinstance(column.type, JSONB):
                column.type = JSON()
            # UUID -> SQLiteUUID 변환
            if isinstance(column.type, UUID):
                column.type = SQLiteUUID()

            # SQLite에서 지원하지 않는 server_default 정리
            if column.server_default is not None:
                default_text = str(column.server_default.arg)
                if "gen_random_uuid()" in default_text:
                    column.server_default = None
                elif "::jsonb" in default_text:
                    column.server_default = None

        # PostgreSQL 전용 인덱스는 SQLite에서 지원하지 않으므로 제거
        # - GIN 인덱스 (postgresql_using="gin")
        # - Partial unique index (postgresql_where)
        indexes_to_remove = set()
        for idx in table.indexes:
            dialect_opts = getattr(idx, "dialect_options", {})
            pg_opts = dialect_opts.get("postgresql", {})
            if pg_opts.get("using") == "gin" or pg_opts.get("where") is not None:
                indexes_to_remove.add(idx)
        table.indexes -= indexes_to_remove


_patch_columns_for_sqlite()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine():
    """SQLite 인메모리 엔진."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """테스트용 세션. 각 테스트 후 롤백."""
    TestSession = sessionmaker(bind=db_engine)
    session = TestSession()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_doc_kwargs():
    """DocumentSource 생성에 필요한 최소 필수 kwargs."""
    return {
        "id": _uuid.uuid4(),
        "source_type": "case",
        "source_id": "CASE-001",
        "title": "테스트 민원 사례",
        "content": "민원 내용 본문 텍스트",
        "chunk_index": 0,
        "total_chunks": 1,
        "reliability_score": 0.8,
        "status": "active",
        "version": "1.0",
        "metadata_": {},
        "embedding_version": "e5-large-v1",
    }


@pytest.fixture
def sample_doc(db_session, sample_doc_kwargs):
    """DB에 저장된 DocumentSource 인스턴스."""
    doc = DocumentSource(**sample_doc_kwargs)
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


@pytest.fixture
def sample_index_version_kwargs():
    """IndexVersion 생성에 필요한 kwargs."""
    return {
        "id": _uuid.uuid4(),
        "index_type": "case",
        "version": "v1.0.0",
        "total_documents": 100,
        "index_file_path": "/data/faiss/case/index.faiss",
        "meta_file_path": "/data/faiss/case/metadata.json",
        "is_active": True,
    }
