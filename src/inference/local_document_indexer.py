"""로컬 문서 폴더 스캔 및 증분 색인 서비스.

Issue #160 기준으로 지정 루트 폴더 아래의 문서를 스캔하고,
DocumentSource 스냅샷과 FAISS 인덱스를 동기화한다.

정책:
- 루트 폴더 하위의 ``case/``, ``law/``, ``manual/``, ``notice/`` 디렉터리를 canonical layout로 사용한다.
- 변경 감지는 파일 fingerprint(크기, mtime, sha256) 기준으로 수행한다.
- per-vector delete 대신, 영향받은 ``index_type`` 전체를 DB snapshot 기준으로 재빌드한다.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.inference.db.crud import create_document_source, create_index_version, deactivate_versions
from src.inference.db.models import DocumentSource
from src.inference.document_processor import DocumentProcessor
from src.inference.index_manager import DocumentMetadata, IndexType, MultiIndexManager

_LAYOUT_DIRS: Dict[IndexType, str] = {
    IndexType.CASE: "case",
    IndexType.LAW: "law",
    IndexType.MANUAL: "manual",
    IndexType.NOTICE: "notice",
}

if set(_LAYOUT_DIRS) != set(IndexType):
    raise ValueError("_LAYOUT_DIRS와 IndexType enum이 동기화되어야 합니다.")


@dataclass(frozen=True)
class ScannedDocument:
    index_type: IndexType
    path: Path
    relative_path: str
    source_id: str
    stat_signature: str


@dataclass(frozen=True)
class PreparedIndexSnapshot:
    index_type: IndexType
    vectors: np.ndarray
    metadata: List[DocumentMetadata]


@dataclass
class IndexSyncSummary:
    scanned_files: int = 0
    indexed_files: int = 0
    unchanged_files: int = 0
    removed_files: int = 0
    indexed_chunks: int = 0
    rebuilt_index_types: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    failed_files: List[Tuple[str, str]] = field(default_factory=list)


class LocalDocumentIndexer:
    """지정 폴더를 스캔하여 DB/FAISS 색인을 갱신한다."""

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        index_manager: MultiIndexManager,
        embed_model: Any,
        session_factory: sessionmaker[Session],
        processor: DocumentProcessor | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.index_manager = index_manager
        self.embed_model = embed_model
        self.session_factory = session_factory
        self.processor = processor or DocumentProcessor()
        self.ensure_layout()
        root_hash = hashlib.sha256(str(self.root_dir).encode("utf-8")).hexdigest()[:12]
        self.source_name = f"local-docs:{root_hash}"

    def ensure_layout(self) -> None:
        """canonical root layout을 준비한다."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for dirname in _LAYOUT_DIRS.values():
            (self.root_dir / dirname).mkdir(parents=True, exist_ok=True)

    def scan_files(self) -> Tuple[List[ScannedDocument], List[str]]:
        """canonical layout 아래의 지원 파일을 스캔한다."""
        scanned: List[ScannedDocument] = []
        skipped: List[str] = []
        seen_paths = set()

        for index_type, dirname in _LAYOUT_DIRS.items():
            type_root = self.root_dir / dirname
            if not type_root.exists():
                continue

            for path in sorted(p for p in type_root.rglob("*") if p.is_file()):
                if path.suffix.lower() not in self.processor.SUPPORTED_EXTENSIONS:
                    skipped.append(str(path))
                    continue

                resolved = path.resolve()
                if resolved in seen_paths:
                    skipped.append(str(path))
                    continue
                seen_paths.add(resolved)

                relative_path = path.relative_to(self.root_dir).as_posix()
                scanned.append(
                    ScannedDocument(
                        index_type=index_type,
                        path=path,
                        relative_path=relative_path,
                        source_id=self._build_source_id(index_type, relative_path),
                        stat_signature=self._build_stat_signature(path),
                    )
                )

        return scanned, skipped

    def sync(self) -> IndexSyncSummary:
        """scan-based incremental indexing을 수행한다."""
        scanned, skipped = self.scan_files()
        scanned_map = {(doc.index_type.value, doc.source_id): doc for doc in scanned}
        summary = IndexSyncSummary(
            scanned_files=len(scanned),
            skipped_files=sorted(skipped),
        )
        prepared_snapshots: List[PreparedIndexSnapshot] = []

        with self.session_factory() as db:
            try:
                existing_groups = self._load_existing_groups(db)
                affected_types: set[IndexType] = set()

                for key, rows in existing_groups.items():
                    if key in scanned_map:
                        continue
                    for row in rows:
                        db.delete(row)
                    summary.removed_files += 1
                    affected_types.add(IndexType(rows[0].source_type))

                for key, scanned_doc in scanned_map.items():
                    existing_rows = existing_groups.get(key)
                    existing_fingerprint = self._group_fingerprint(existing_rows)

                    try:
                        fingerprint = self._resolve_fingerprint(scanned_doc, existing_fingerprint)
                        if existing_rows and fingerprint is None:
                            summary.unchanged_files += 1
                            continue

                        processed = self._process_document(scanned_doc, fingerprint=fingerprint)
                    except Exception as exc:
                        logger.error(f"로컬 문서 처리 실패: {scanned_doc.path} — {exc}")
                        summary.failed_files.append((str(scanned_doc.path), str(exc)))
                        continue

                    if existing_rows:
                        for row in existing_rows:
                            db.delete(row)
                        db.flush()

                    for meta in processed:
                        content = meta.extras.get("chunk_text", "") if meta.extras else ""
                        kwargs = self._to_document_source_kwargs(meta, content)
                        create_document_source(db, **kwargs)

                    summary.indexed_files += 1
                    summary.indexed_chunks += len(processed)
                    affected_types.add(scanned_doc.index_type)

                for index_type in sorted(affected_types, key=lambda item: item.value):
                    prepared_snapshots.append(self._prepare_index_snapshot(db, index_type))
                    summary.rebuilt_index_types.append(index_type.value)

                db.commit()
            except Exception:
                db.rollback()
                raise

        for snapshot in prepared_snapshots:
            self.index_manager.replace_index(
                snapshot.index_type, snapshot.vectors, snapshot.metadata
            )
            self.index_manager.save_index(snapshot.index_type)

        return summary

    def _process_document(
        self,
        scanned_doc: ScannedDocument,
        *,
        fingerprint: str,
    ) -> List[DocumentMetadata]:
        return self.processor.process(
            str(scanned_doc.path),
            scanned_doc.index_type,
            source=self.source_name,
            title=scanned_doc.path.stem,
            document_id=scanned_doc.source_id,
            extras={
                "relative_path": scanned_doc.relative_path,
                "file_fingerprint": fingerprint,
            },
        )

    def _load_existing_groups(
        self,
        db: Session,
    ) -> Dict[Tuple[str, str], List[DocumentSource]]:
        stmt = (
            select(DocumentSource)
            .where(DocumentSource.source_name == self.source_name)
            .order_by(
                DocumentSource.source_type,
                DocumentSource.source_id,
                DocumentSource.chunk_index,
            )
        )

        grouped: Dict[Tuple[str, str], List[DocumentSource]] = {}
        for row in db.scalars(stmt).all():
            grouped.setdefault((row.source_type, row.source_id), []).append(row)
        return grouped

    @staticmethod
    def _group_fingerprint(rows: Sequence[DocumentSource] | None) -> str | None:
        if not rows:
            return None
        metadata = rows[0].metadata_ or {}
        return metadata.get("file_fingerprint")

    @staticmethod
    def _build_source_id(index_type: IndexType, relative_path: str) -> str:
        raw = f"{index_type.value}:{relative_path}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    @staticmethod
    def _build_stat_signature(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    @staticmethod
    def _fingerprint(path: Path, stat_signature: str | None = None) -> str:
        signature = stat_signature or LocalDocumentIndexer._build_stat_signature(path)
        digest = hashlib.sha256()
        digest.update(signature.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"{signature}:{digest.hexdigest()}"

    @staticmethod
    def _matches_stat_signature(
        existing_fingerprint: str | None,
        stat_signature: str,
    ) -> bool:
        return bool(existing_fingerprint and existing_fingerprint.startswith(f"{stat_signature}:"))

    def _resolve_fingerprint(
        self,
        scanned_doc: ScannedDocument,
        existing_fingerprint: str | None,
    ) -> str | None:
        if self._matches_stat_signature(existing_fingerprint, scanned_doc.stat_signature):
            return None

        fingerprint = self._fingerprint(scanned_doc.path, scanned_doc.stat_signature)
        if existing_fingerprint == fingerprint:
            return None
        return fingerprint

    def _prepare_index_snapshot(
        self,
        db: Session,
        index_type: IndexType,
    ) -> PreparedIndexSnapshot:
        """DB의 active snapshot을 기준으로 인덱스 재빌드용 데이터를 준비한다.

        index_type별 FAISS 인덱스는 source_name이 아니라 active DB snapshot 전체를
        source of truth로 사용한다. 로컬 문서 sync도 같은 type의 기존 문서를 보존한 채
        최신 스냅샷으로 재구성한다.
        """
        stmt = (
            select(DocumentSource)
            .where(
                DocumentSource.source_type == index_type.value,
                DocumentSource.status == "active",
            )
            .order_by(DocumentSource.source_id, DocumentSource.chunk_index)
        )
        rows = list(db.scalars(stmt).all())

        if rows:
            contents = [row.content for row in rows]
            vectors = self._encode_passages(contents)
            metadata = [self._row_to_metadata(row) for row in rows]
            for position, row in enumerate(rows):
                row.faiss_index_id = position
        else:
            vectors = np.empty((0, self.index_manager.embedding_dim), dtype=np.float32)
            metadata = []

        self._record_index_version(db, index_type, len(rows))
        return PreparedIndexSnapshot(index_type=index_type, vectors=vectors, metadata=metadata)

    def _row_to_metadata(self, row: DocumentSource) -> DocumentMetadata:
        extras = dict(row.metadata_ or {})
        extras.setdefault("chunk_text", row.content)
        extras.setdefault("chunk_id", f"{row.source_id}:{row.chunk_index}")
        return DocumentMetadata(
            doc_id=row.source_id,
            doc_type=row.source_type,
            source=row.source_name or "",
            title=row.title,
            category=row.category or "",
            reliability_score=row.reliability_score,
            created_at=row.created_at.isoformat(),
            updated_at=row.updated_at.isoformat(),
            valid_from=(row.valid_from.isoformat() if row.valid_from else None),
            valid_until=(row.valid_until.isoformat() if row.valid_until else None),
            chunk_index=row.chunk_index,
            chunk_total=row.total_chunks,
            extras=extras,
        )

    def _encode_passages(self, contents: Iterable[str]) -> np.ndarray:
        passages = [f"passage: {content}" for content in contents]
        vectors = self.embed_model.encode(passages, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def _record_index_version(
        self, db: Session, index_type: IndexType, total_documents: int
    ) -> None:
        deactivate_versions(db, index_type.value)
        index_dir = Path(self.index_manager.base_dir) / index_type.value
        version = datetime.now(timezone.utc).strftime("sync-%Y%m%d%H%M%S")
        create_index_version(
            db,
            id=uuid.uuid4(),
            index_type=index_type.value,
            version=version,
            total_documents=total_documents,
            index_file_path=str(index_dir / "index.faiss"),
            meta_file_path=str(index_dir / "metadata.json"),
            is_active=True,
            notes="local document sync snapshot",
        )

    @staticmethod
    def _to_document_source_kwargs(meta: DocumentMetadata, content: str) -> Dict[str, Any]:
        extras = dict(meta.extras) if meta.extras else {}
        kwargs: Dict[str, Any] = {
            "id": uuid.uuid4(),
            "source_type": meta.doc_type,
            "source_id": meta.doc_id,
            "source_name": meta.source,
            "title": meta.title,
            "content": content,
            "category": meta.category,
            "chunk_index": meta.chunk_index,
            "total_chunks": meta.chunk_total,
            "reliability_score": meta.reliability_score,
            "metadata_": extras,
        }
        if meta.valid_from:
            kwargs["valid_from"] = datetime.fromisoformat(meta.valid_from)
        if meta.valid_until:
            kwargs["valid_until"] = datetime.fromisoformat(meta.valid_until)
        return kwargs
