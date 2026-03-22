import json
import os
from datetime import datetime, timedelta
from typing import List

import faiss
import numpy as np
import pytest

from src.inference.index_manager import (
    DocumentMetadata,
    IndexType,
    MultiIndexManager,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def make_vectors(n: int, dim: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim), dtype=np.float32)
    return _normalize(vecs)


def make_metadata(n: int, doc_type: str = "case") -> List[DocumentMetadata]:
    now = datetime.utcnow().isoformat()
    return [
        DocumentMetadata(
            doc_id=f"{doc_type}-{i:04d}",
            doc_type=doc_type,
            source=f"source-{doc_type}",
            title=f"Document {doc_type} {i}",
            category="general",
            reliability_score=0.95,
            created_at=now,
            updated_at=now,
            chunk_index=0,
            chunk_total=1,
        )
        for i in range(n)
    ]


@pytest.fixture
def manager(tmp_path):
    return MultiIndexManager(base_dir=str(tmp_path), embedding_dim=1024)


@pytest.fixture
def populated_manager(manager):
    vecs = make_vectors(20)
    meta = make_metadata(20, doc_type="case")
    manager.add_documents(IndexType.CASE, vecs, meta)
    return manager


# ---------------------------------------------------------------------------
# 1. IndexType enum
# ---------------------------------------------------------------------------

class TestIndexType:
    def test_values(self):
        assert IndexType.CASE.value == "case"
        assert IndexType.LAW.value == "law"
        assert IndexType.MANUAL.value == "manual"
        assert IndexType.NOTICE.value == "notice"

    def test_member_count(self):
        assert len(IndexType) == 4

    def test_is_str(self):
        for member in IndexType:
            assert isinstance(member, str)


# ---------------------------------------------------------------------------
# 2. DocumentMetadata
# ---------------------------------------------------------------------------

class TestDocumentMetadata:
    def test_required_fields(self):
        now = datetime.utcnow().isoformat()
        meta = DocumentMetadata(
            doc_id="d-001",
            doc_type="case",
            source="court",
            title="Test",
            category="civil",
            reliability_score=0.9,
            created_at=now,
            updated_at=now,
        )
        assert meta.doc_id == "d-001"
        assert meta.doc_type == "case"
        assert meta.title == "Test"

    def test_default_values(self):
        now = datetime.utcnow().isoformat()
        meta = DocumentMetadata(
            doc_id="d-002",
            doc_type="law",
            source="gazette",
            title="Default Test",
            category="admin",
            reliability_score=1.0,
            created_at=now,
            updated_at=now,
        )
        assert meta.valid_from is None
        assert meta.valid_until is None
        assert meta.chunk_index == 0
        assert meta.chunk_total == 1
        assert meta.extras == {}

    def test_optional_fields(self):
        now = datetime.utcnow()
        meta = DocumentMetadata(
            doc_id="d-003",
            doc_type="manual",
            source="ministry",
            title="Optional Test",
            category="procedure",
            reliability_score=0.85,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            valid_from=(now - timedelta(days=30)).isoformat(),
            valid_until=(now + timedelta(days=365)).isoformat(),
            chunk_index=2,
            chunk_total=5,
            extras={"lang": "ko"},
        )
        assert meta.valid_from is not None
        assert meta.valid_until is not None
        assert meta.chunk_index == 2
        assert meta.chunk_total == 5
        assert meta.extras == {"lang": "ko"}


# ---------------------------------------------------------------------------
# 3. MultiIndexManager initialization
# ---------------------------------------------------------------------------

class TestMultiIndexManagerInit:
    def test_creates_base_directory(self, tmp_path):
        target = str(tmp_path / "new_index_dir")
        assert not os.path.exists(target)
        MultiIndexManager(base_dir=target, embedding_dim=1024)
        assert os.path.isdir(target)

    def test_default_embedding_dim(self, tmp_path):
        mgr = MultiIndexManager(base_dir=str(tmp_path))
        assert mgr.embedding_dim == 1024

    def test_custom_embedding_dim(self, tmp_path):
        mgr = MultiIndexManager(base_dir=str(tmp_path), embedding_dim=768)
        assert mgr.embedding_dim == 768


# ---------------------------------------------------------------------------
# 4. Index CRUD
# ---------------------------------------------------------------------------

class TestIndexCRUD:
    def test_add_documents_increases_index_size(self, manager):
        vecs = make_vectors(10)
        meta = make_metadata(10)
        manager.add_documents(IndexType.CASE, vecs, meta)
        stats = manager.get_index_stats()
        assert stats["indexes"][IndexType.CASE.value]["doc_count"] == 10

    def test_add_documents_incremental(self, manager):
        vecs1 = make_vectors(5)
        meta1 = make_metadata(5)
        manager.add_documents(IndexType.CASE, vecs1, meta1)

        vecs2 = make_vectors(3)
        meta2 = make_metadata(3)
        manager.add_documents(IndexType.CASE, vecs2, meta2)

        stats = manager.get_index_stats()
        assert stats["indexes"][IndexType.CASE.value]["doc_count"] == 8

    def test_save_and_load_index(self, populated_manager, tmp_path):
        populated_manager.save_index(IndexType.CASE)

        new_mgr = MultiIndexManager(base_dir=str(tmp_path), embedding_dim=1024)
        new_mgr.load_index(IndexType.CASE)

        query = make_vectors(1)
        original_results = populated_manager.search(IndexType.CASE, query, top_k=5)
        loaded_results = new_mgr.search(IndexType.CASE, query, top_k=5)

        assert len(original_results) == len(loaded_results)
        for orig, loaded in zip(original_results, loaded_results):
            assert orig["doc_id"] == loaded["doc_id"]
            assert abs(orig["score"] - loaded["score"]) < 1e-5


# ---------------------------------------------------------------------------
# 5. Search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_top_k_count(self, populated_manager):
        query = make_vectors(1)
        results = populated_manager.search(IndexType.CASE, query, top_k=3)
        assert len(results) == 3

    def test_top_k_clamped_to_available(self, manager):
        vecs = make_vectors(3)
        meta = make_metadata(3)
        manager.add_documents(IndexType.CASE, vecs, meta)

        query = make_vectors(1)
        results = manager.search(IndexType.CASE, query, top_k=10)
        assert len(results) == 3

    def test_results_contain_metadata(self, populated_manager):
        query = make_vectors(1)
        results = populated_manager.search(IndexType.CASE, query, top_k=1)
        assert len(results) == 1
        result = results[0]
        assert "doc_id" in result
        assert "doc_type" in result
        assert "title" in result
        assert "source" in result

    def test_results_contain_score(self, populated_manager):
        query = make_vectors(1)
        results = populated_manager.search(IndexType.CASE, query, top_k=3)
        for result in results:
            assert "score" in result
            assert isinstance(result["score"], float)

    def test_search_empty_index_returns_empty(self, manager):
        query = make_vectors(1)
        results = manager.search(IndexType.LAW, query, top_k=5)
        assert results == []

    def test_search_returns_best_match_first(self, manager):
        dim = 1024
        target = np.zeros((1, dim), dtype=np.float32)
        target[0, 0] = 1.0

        vecs = make_vectors(10)
        vecs[7] = target[0]
        meta = make_metadata(10)
        manager.add_documents(IndexType.CASE, vecs, meta)

        results = manager.search(IndexType.CASE, target, top_k=1)
        assert results[0]["doc_id"] == "case-0007"


# ---------------------------------------------------------------------------
# 6. IVFFlat upgrade
# ---------------------------------------------------------------------------

class TestIVFFlatUpgrade:
    def test_maybe_upgrade_to_ivf(self, tmp_path):
        mgr = MultiIndexManager(base_dir=str(tmp_path), embedding_dim=1024)

        vecs = make_vectors(500)
        meta = make_metadata(500)
        manager_with_data = mgr
        manager_with_data.add_documents(IndexType.CASE, vecs, meta)

        if hasattr(mgr, "_maybe_upgrade_to_ivf"):
            mgr._maybe_upgrade_to_ivf(IndexType.CASE)

        stats = mgr.get_index_stats()
        assert stats["indexes"][IndexType.CASE.value]["doc_count"] == 500

    def test_search_works_after_ivf_upgrade(self, tmp_path):
        mgr = MultiIndexManager(base_dir=str(tmp_path), embedding_dim=1024)

        vecs = make_vectors(500)
        meta = make_metadata(500)
        mgr.add_documents(IndexType.CASE, vecs, meta)

        if hasattr(mgr, "_maybe_upgrade_to_ivf"):
            mgr._maybe_upgrade_to_ivf(IndexType.CASE)

        query = make_vectors(1)
        results = mgr.search(IndexType.CASE, query, top_k=5)
        assert len(results) > 0
        for r in results:
            assert "doc_id" in r
            assert "score" in r


# ---------------------------------------------------------------------------
# 7. index_registry.json
# ---------------------------------------------------------------------------

class TestIndexRegistry:
    def test_registry_created_after_save(self, populated_manager, tmp_path):
        populated_manager.save_index(IndexType.CASE)
        registry_path = os.path.join(str(tmp_path), "index_registry.json")
        assert os.path.isfile(registry_path)

    def test_registry_contains_stats(self, populated_manager, tmp_path):
        populated_manager.save_index(IndexType.CASE)
        registry_path = os.path.join(str(tmp_path), "index_registry.json")

        with open(registry_path) as f:
            registry = json.load(f)

        assert "indexes" in registry
        assert IndexType.CASE.value in registry["indexes"]
        entry = registry["indexes"][IndexType.CASE.value]
        assert "doc_count" in entry


# ---------------------------------------------------------------------------
# 8. Multi-index independence
# ---------------------------------------------------------------------------

class TestMultiIndexIndependence:
    def test_separate_indexes_are_independent(self, manager):
        case_vecs = make_vectors(10)
        case_meta = make_metadata(10, doc_type="case")
        manager.add_documents(IndexType.CASE, case_vecs, case_meta)

        law_vecs = make_vectors(7)
        law_meta = make_metadata(7, doc_type="law")
        manager.add_documents(IndexType.LAW, law_vecs, law_meta)

        stats = manager.get_index_stats()
        assert stats["indexes"][IndexType.CASE.value]["doc_count"] == 10
        assert stats["indexes"][IndexType.LAW.value]["doc_count"] == 7

    def test_search_does_not_leak_across_indexes(self, manager):
        case_vecs = make_vectors(5)
        case_meta = make_metadata(5, doc_type="case")
        manager.add_documents(IndexType.CASE, case_vecs, case_meta)

        law_vecs = make_vectors(5)
        law_meta = make_metadata(5, doc_type="law")
        manager.add_documents(IndexType.LAW, law_vecs, law_meta)

        query = make_vectors(1)

        case_results = manager.search(IndexType.CASE, query, top_k=5)
        law_results = manager.search(IndexType.LAW, query, top_k=5)

        for r in case_results:
            assert r["doc_type"] == "case"
        for r in law_results:
            assert r["doc_type"] == "law"

    def test_empty_index_unaffected_by_other(self, manager):
        vecs = make_vectors(10)
        meta = make_metadata(10, doc_type="case")
        manager.add_documents(IndexType.CASE, vecs, meta)

        query = make_vectors(1)
        manual_results = manager.search(IndexType.MANUAL, query, top_k=5)
        assert manual_results == []
