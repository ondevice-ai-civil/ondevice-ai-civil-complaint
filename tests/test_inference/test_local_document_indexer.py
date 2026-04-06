import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from src.inference.db.models import DocumentSource, IndexVersion
from src.inference.index_manager import IndexType, MultiIndexManager
from src.inference.local_document_indexer import LocalDocumentIndexer


@pytest.fixture(autouse=True)
def _no_tokenizer():
    import src.inference.document_processor as dp

    original = dp._tokenizer
    dp._tokenizer = dp._LOAD_FAILED
    with patch.object(dp, "_get_tokenizer", return_value=None):
        yield
    dp._tokenizer = original


class _FakeEmbedModel:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = np.array(
                [(digest[i % len(digest)] + 1) / 255.0 for i in range(self.dim)],
                dtype=np.float32,
            )
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.stack(vectors)


def _make_service(tmp_path: Path, db_engine, dim: int = 8):
    root_dir = tmp_path / "local_docs"
    manager = MultiIndexManager(base_dir=str(tmp_path / "faiss"), embedding_dim=dim)
    session_factory = sessionmaker(bind=db_engine)
    service = LocalDocumentIndexer(
        root_dir=root_dir,
        index_manager=manager,
        embed_model=_FakeEmbedModel(dim=dim),
        session_factory=session_factory,
    )
    return service, session_factory, manager, root_dir


class TestLocalDocumentIndexerScan:
    def test_scan_uses_canonical_layout_and_skips_unsupported(self, tmp_path, db_engine):
        service, _, _, root_dir = _make_service(tmp_path, db_engine)
        service.ensure_layout()

        (root_dir / "case" / "road.txt").write_text("도로 보수 지침", encoding="utf-8")
        (root_dir / "law" / "article.txt").write_text("제1조 도로 점검", encoding="utf-8")
        (root_dir / "manual" / "ignore.docx").write_text("지원 안 함", encoding="utf-8")

        scanned, skipped = service.scan_files()

        assert {(doc.index_type, doc.relative_path) for doc in scanned} == {
            (IndexType.CASE, "case/road.txt"),
            (IndexType.LAW, "law/article.txt"),
        }
        assert str(root_dir / "manual" / "ignore.docx") in skipped


class TestLocalDocumentIndexerSync:
    def test_full_sync_builds_db_snapshot_and_faiss_index(self, tmp_path, db_engine):
        service, SessionFactory, manager, root_dir = _make_service(tmp_path, db_engine)
        service.ensure_layout()

        (root_dir / "case" / "road.txt").write_text("도로 포장 파손 보수 절차", encoding="utf-8")
        (root_dir / "manual" / "guide.txt").write_text("민원 처리 가이드", encoding="utf-8")

        summary = service.sync()

        assert summary.scanned_files == 2
        assert summary.indexed_files == 2
        assert summary.unchanged_files == 0
        assert set(summary.rebuilt_index_types) == {"case", "manual"}

        with SessionFactory() as db:
            docs = list(
                db.scalars(
                    select(DocumentSource).order_by(
                        DocumentSource.source_type,
                        DocumentSource.source_id,
                        DocumentSource.chunk_index,
                    )
                ).all()
            )
            assert len(docs) == 2
            assert docs[0].metadata_["relative_path"] in {"case/road.txt", "manual/guide.txt"}
            assert "file_path" in docs[0].metadata_
            assert "chunk_id" in docs[0].metadata_

            versions = list(db.scalars(select(IndexVersion)).all())
            assert {version.index_type for version in versions} == {"case", "manual"}

        stats = manager.get_index_stats()
        assert stats["indexes"]["case"]["doc_count"] == 1
        assert stats["indexes"]["manual"]["doc_count"] == 1

    def test_incremental_sync_reindexes_only_changed_file(self, tmp_path, db_engine):
        service, SessionFactory, manager, root_dir = _make_service(tmp_path, db_engine)
        service.ensure_layout()

        case_file = root_dir / "case" / "road.txt"
        manual_file = root_dir / "manual" / "guide.txt"
        case_file.write_text("초기 도로 보수 절차", encoding="utf-8")
        manual_file.write_text("초기 민원 처리 가이드", encoding="utf-8")

        first = service.sync()
        assert first.indexed_files == 2

        case_file.write_text("갱신된 도로 보수 절차와 안전 점검", encoding="utf-8")
        second = service.sync()

        assert second.indexed_files == 1
        assert second.unchanged_files == 1
        assert second.removed_files == 0
        assert second.rebuilt_index_types == ["case"]

        with SessionFactory() as db:
            docs = list(db.scalars(select(DocumentSource)).all())
            road_rows = [
                doc for doc in docs if doc.metadata_.get("relative_path") == "case/road.txt"
            ]
            guide_rows = [
                doc for doc in docs if doc.metadata_.get("relative_path") == "manual/guide.txt"
            ]
            assert len(road_rows) == 1
            assert road_rows[0].content == "갱신된 도로 보수 절차와 안전 점검"
            assert len(guide_rows) == 1
            assert guide_rows[0].content == "초기 민원 처리 가이드"

        assert manager.get_index_stats()["indexes"]["case"]["doc_count"] == 1
        assert (
            manager.metadata[IndexType.CASE][0].extras["chunk_text"]
            == "갱신된 도로 보수 절차와 안전 점검"
        )

    def test_deleted_file_is_removed_and_index_rebuilt(self, tmp_path, db_engine):
        service, SessionFactory, manager, root_dir = _make_service(tmp_path, db_engine)
        service.ensure_layout()

        target = root_dir / "case" / "road.txt"
        target.write_text("삭제 예정 문서", encoding="utf-8")
        service.sync()

        target.unlink()
        summary = service.sync()

        assert summary.removed_files == 1
        assert summary.rebuilt_index_types == ["case"]

        with SessionFactory() as db:
            docs = list(db.scalars(select(DocumentSource)).all())
            assert docs == []

        assert manager.get_index_stats()["indexes"]["case"]["doc_count"] == 0

    def test_pdf_sync_preserves_page_metadata(self, tmp_path, db_engine):
        service, SessionFactory, manager, root_dir = _make_service(tmp_path, db_engine)
        service.ensure_layout()

        pdf_path = root_dir / "law" / "road-law.pdf"
        pdf_path.touch()

        import src.inference.document_processor as dp

        with patch.dict(
            dp._PAGE_PARSERS,
            {
                ".pdf": lambda fp: [
                    (1, "제1조 도로 점검 기준"),
                    (2, "제2조 도로 보수 절차"),
                ]
            },
            clear=False,
        ):
            summary = service.sync()

        assert summary.indexed_files == 1
        assert summary.rebuilt_index_types == ["law"]

        with SessionFactory() as db:
            docs = list(
                db.scalars(
                    select(DocumentSource)
                    .where(DocumentSource.source_type == "law")
                    .order_by(DocumentSource.chunk_index)
                ).all()
            )
            assert [doc.metadata_.get("page") for doc in docs] == [1, 2]

        assert [meta.extras.get("page") for meta in manager.metadata[IndexType.LAW]] == [1, 2]

    def test_commit_failure_does_not_replace_or_save_faiss(self, tmp_path, db_engine):
        service, SessionFactory, manager, root_dir = _make_service(tmp_path, db_engine)
        service.ensure_layout()
        (root_dir / "case" / "road.txt").write_text("도로 포장 파손 보수 절차", encoding="utf-8")

        class _FailingCommitSession:
            def __init__(self, session):
                self._session = session

            def __enter__(self):
                self._session.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                return self._session.__exit__(exc_type, exc, tb)

            def __getattr__(self, name):
                return getattr(self._session, name)

            def commit(self):
                raise RuntimeError("commit failed")

        service.session_factory = lambda: _FailingCommitSession(SessionFactory())

        with patch.object(manager, "replace_index", MagicMock()) as mock_replace:
            with patch.object(manager, "save_index", MagicMock()) as mock_save:
                with pytest.raises(RuntimeError, match="commit failed"):
                    service.sync()

        mock_replace.assert_not_called()
        mock_save.assert_not_called()
