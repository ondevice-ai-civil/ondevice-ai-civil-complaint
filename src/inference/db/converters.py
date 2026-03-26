"""
ORM <-> Dataclass / Pydantic 변환 헬퍼.

DocumentSource ORM 인스턴스를 기존 DocumentMetadata(dataclass) 또는
DocumentMetadataSchema(Pydantic) 모델로 상호 변환한다.
"""

from typing import Any, Dict

from src.inference.db.models import DocumentSource
from src.inference.index_manager import DocumentMetadata, IndexType
from src.inference.schemas import DocumentMetadataSchema

# 타입별 전용 필드 목록 (ORM <-> Dataclass/Pydantic 변환 시 공통 사용)
_TYPE_SPECIFIC_FIELDS: tuple = (
    "complaint_text",
    "answer_text",  # CASE
    "law_number",
    "article_number",  # LAW
    "enforcement_date",  # LAW
    "department",  # MANUAL
    "notice_number",
    "effective_date",  # NOTICE
)


# ---------------------------------------------------------------------------
# ORM -> Dataclass
# ---------------------------------------------------------------------------


def orm_to_dataclass(doc_source: DocumentSource) -> DocumentMetadata:
    """DocumentSource ORM -> DocumentMetadata dataclass 변환.

    ORM의 타입별 전용 필드(complaint_text, law_number 등)는
    extras dict에 모아서 전달한다.
    """
    # 타입별 추가 필드를 extras로 수집
    extras: Dict[str, Any] = {}
    if doc_source.metadata_:
        extras.update(doc_source.metadata_)

    for field_name in _TYPE_SPECIFIC_FIELDS:
        value = getattr(doc_source, field_name, None)
        if value is not None:
            # date/datetime 객체는 ISO 문자열로 직렬화
            extras[field_name] = value.isoformat() if hasattr(value, "isoformat") else value

    return DocumentMetadata(
        doc_id=str(doc_source.id),
        doc_type=doc_source.source_type,
        source=doc_source.source_name or "",
        title=doc_source.title,
        category=doc_source.category or "",
        reliability_score=doc_source.reliability_score,
        created_at=doc_source.created_at.isoformat(),
        updated_at=doc_source.updated_at.isoformat(),
        valid_from=(doc_source.valid_from.isoformat() if doc_source.valid_from else None),
        valid_until=(doc_source.valid_until.isoformat() if doc_source.valid_until else None),
        chunk_index=doc_source.chunk_index,
        chunk_total=doc_source.total_chunks,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Dataclass -> ORM create kwargs
# ---------------------------------------------------------------------------


def dataclass_to_orm(meta: DocumentMetadata, content: str) -> Dict[str, Any]:
    """DocumentMetadata dataclass -> DocumentSource 생성용 kwargs 딕셔너리.

    Parameters
    ----------
    meta : DocumentMetadata
        내부 dataclass 인스턴스.
    content : str
        문서 본문 텍스트 (dataclass에는 content가 없음).

    Returns
    -------
    dict
        crud.create_document_source()에 전달할 kwargs.
    """
    extras = dict(meta.extras) if meta.extras else {}

    kwargs: Dict[str, Any] = {
        "source_type": meta.doc_type,
        "source_id": meta.doc_id,
        "source_name": meta.source,
        "title": meta.title,
        "content": content,
        "category": meta.category,
        "chunk_index": meta.chunk_index,
        "total_chunks": meta.chunk_total,
        "reliability_score": meta.reliability_score,
        "metadata_": {},
    }

    # ISO 문자열 -> datetime 변환 (valid_from/valid_until)
    from datetime import datetime

    if meta.valid_from:
        kwargs["valid_from"] = datetime.fromisoformat(meta.valid_from)
    if meta.valid_until:
        kwargs["valid_until"] = datetime.fromisoformat(meta.valid_until)

    # extras에서 타입별 전용 필드 추출
    _type_field_map = {
        "complaint_text": str,
        "answer_text": str,
        "law_number": str,
        "article_number": str,
        "enforcement_date": str,  # DATE 컬럼이므로 문자열 그대로 전달
        "department": str,
        "notice_number": str,
        "effective_date": str,
    }
    remaining_extras: Dict[str, Any] = {}
    for key, value in extras.items():
        if key in _type_field_map:
            kwargs[key] = value
        else:
            remaining_extras[key] = value

    kwargs["metadata_"] = remaining_extras
    return kwargs


# ---------------------------------------------------------------------------
# ORM -> Pydantic
# ---------------------------------------------------------------------------


def orm_to_pydantic(doc_source: DocumentSource) -> DocumentMetadataSchema:
    """DocumentSource ORM -> DocumentMetadataSchema Pydantic 모델 변환."""
    # 타입별 추가 필드 + JSONB metadata를 합산
    extra_meta: Dict[str, Any] = {}
    if doc_source.metadata_:
        extra_meta.update(doc_source.metadata_)

    for field_name in _TYPE_SPECIFIC_FIELDS:
        value = getattr(doc_source, field_name, None)
        if value is not None:
            extra_meta[field_name] = value.isoformat() if hasattr(value, "isoformat") else value

    return DocumentMetadataSchema(
        doc_id=str(doc_source.id),
        source_type=IndexType(doc_source.source_type),
        source_id=doc_source.source_id,
        title=doc_source.title,
        content=doc_source.content,
        chunk_index=doc_source.chunk_index,
        total_chunks=doc_source.total_chunks,
        created_at=doc_source.created_at,
        updated_at=doc_source.updated_at,
        valid_from=doc_source.valid_from,
        valid_until=doc_source.valid_until,
        reliability_score=doc_source.reliability_score,
        metadata=extra_meta,
    )
