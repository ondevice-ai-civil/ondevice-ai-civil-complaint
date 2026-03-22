"""
GovOn RAG 데이터베이스 모듈.

SQLAlchemy 2.0 기반 ORM 모델, CRUD 레이어, 변환 헬퍼를 제공한다.
"""

from src.inference.db.database import SessionLocal, engine, get_db
from src.inference.db.models import (
    Base,
    DocumentSource,
    IndexingQueue,
    IndexVersion,
)
from src.inference.db.crud import (
    # DocumentSource
    create_document_source,
    get_document_source,
    get_document_sources,
    update_document_source,
    delete_document_source,
    get_by_source_type_and_id,
    # IndexingQueue
    create_indexing_queue_item,
    get_pending_items,
    update_queue_status,
    get_queue_stats,
    # IndexVersion
    create_index_version,
    get_active_version,
    deactivate_versions,
    activate_version,
)
from src.inference.db.converters import (
    orm_to_dataclass,
    dataclass_to_orm,
    orm_to_pydantic,
)

__all__ = [
    # 데이터베이스 인프라
    "engine",
    "SessionLocal",
    "get_db",
    "Base",
    # ORM 모델
    "DocumentSource",
    "IndexingQueue",
    "IndexVersion",
    # DocumentSource CRUD
    "create_document_source",
    "get_document_source",
    "get_document_sources",
    "update_document_source",
    "delete_document_source",
    "get_by_source_type_and_id",
    # IndexingQueue CRUD
    "create_indexing_queue_item",
    "get_pending_items",
    "update_queue_status",
    "get_queue_stats",
    # IndexVersion CRUD
    "create_index_version",
    "get_active_version",
    "deactivate_versions",
    "activate_version",
    # 변환 헬퍼
    "orm_to_dataclass",
    "dataclass_to_orm",
    "orm_to_pydantic",
]
