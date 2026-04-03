"""
MultiIndexManager 단위 테스트.

FAISS를 Mock으로 대체하여 GPU 없이 실행 가능.
인덱스 생성/로드/저장/검색/문서추가/IVF전환/통계를 검증한다.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# faiss mock 등록
# ---------------------------------------------------------------------------
_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, MagicMock)
if not _faiss_is_real:
    _faiss_mock = MagicMock()
    _faiss_mock.IndexIVFFlat = type("IndexIVFFlat", (), {})
    _faiss_mock.IndexFlatIP = type("IndexFlatIP", (), {})
    sys.modules.setdefault("faiss", _faiss_mock)

from src.inference.index_manager import (
    DocumentMetadata,
    IndexType,
    MultiIndexManager,
    _IVF_THRESHOLD,
)


# ---------------------------------------------------------------------------
# DocumentMetadata 테스트
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    def test_to_dict(self):
        """to_dict()가 올바른 딕셔너리를 반환한다."""
        meta = DocumentMetadata(
            doc_id="d1",
            doc_type="case",
            source="AI Hub",
            title="테스트",
            category="도로",
            reliability_score=0.9,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        d = meta.to_dict()
        assert d["doc_id"] == "d1"
        assert d["doc_type"] == "case"
        assert d["reliability_score"] == 0.9
        assert d["extras"] == {}

    def test_from_dict(self):
        """from_dict()가 올바른 인스턴스를 생성한다."""
        data = {
            "doc_id": "d2",
            "doc_type": "law",
            "source": "법제처",
            "title": "법령 테스트",
            "category": "법률",
            "reliability_score": 1.0,
            "created_at": "2026-01-01",
            "updated_at": "2026-01-01",
            "extras": {"law_text": "제1조"},
        }
        meta = DocumentMetadata.from_dict(data)
        assert meta.doc_id == "d2"
        assert meta.extras == {"law_text": "제1조"}

    def test_from_dict_ignores_unknown_fields(self):
        """알 수 없는 필드는 무시한다."""
        data = {
            "doc_id": "d3",
            "doc_type": "case",
            "source": "내부",
            "title": "제목",
            "category": "교통",
            "reliability_score": 0.5,
            "created_at": "2026-01-01",
            "updated_at": "2026-01-01",
            "unknown_field": "should be ignored",
        }
        meta = DocumentMetadata.from_dict(data)
        assert meta.doc_id == "d3"
        assert not hasattr(meta, "unknown_field")

    def test_roundtrip(self):
        """to_dict -> from_dict 라운드트립."""
        original = DocumentMetadata(
            doc_id="d4",
            doc_type="manual",
            source="기관",
            title="매뉴얼",
            category="업무",
            reliability_score=0.7,
            created_at="2026-01-01",
            updated_at="2026-01-01",
            chunk_index=2,
            chunk_total=5,
            extras={"manual_text": "절차"},
        )
        restored = DocumentMetadata.from_dict(original.to_dict())
        assert restored.doc_id == original.doc_id
        assert restored.chunk_index == 2
        assert restored.extras == {"manual_text": "절차"}


# ---------------------------------------------------------------------------
# MultiIndexManager 초기화 테스트
# ---------------------------------------------------------------------------


class TestMultiIndexManagerInit:
    def test_creates_base_dir(self, tmp_path):
        """초기화 시 base_dir을 생성한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mgr = MultiIndexManager(base_dir=base)
        assert os.path.isdir(base)

    def test_loads_empty_registry(self, tmp_path):
        """레지스트리 파일이 없으면 빈 구조를 사용한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss"):
            mgr = MultiIndexManager(base_dir=base)
        assert mgr._registry == {"indexes": {}, "updated_at": None}

    def test_loads_existing_registry(self, tmp_path):
        """기존 레지스트리 파일을 로드한다."""
        base = str(tmp_path / "faiss_index")
        os.makedirs(base, exist_ok=True)
        registry = {"indexes": {"case": {"doc_count": 100}}, "updated_at": "2026-01-01"}
        with open(os.path.join(base, "index_registry.json"), "w") as f:
            json.dump(registry, f)

        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_faiss.IndexFlatIP.return_value = MagicMock(ntotal=0)
            mgr = MultiIndexManager(base_dir=base)

        assert "case" in mgr._registry["indexes"]

    def test_handles_corrupt_registry(self, tmp_path):
        """손상된 레지스트리 파일은 빈 구조로 초기화한다."""
        base = str(tmp_path / "faiss_index")
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "index_registry.json"), "w") as f:
            f.write("not valid json{{{")

        with patch("src.inference.index_manager.faiss"):
            mgr = MultiIndexManager(base_dir=base)

        assert mgr._registry == {"indexes": {}, "updated_at": None}

    def test_custom_embedding_dim(self, tmp_path):
        """커스텀 임베딩 차원을 설정할 수 있다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss"):
            mgr = MultiIndexManager(base_dir=base, embedding_dim=768)
        assert mgr.embedding_dim == 768


# ---------------------------------------------------------------------------
# load_index 테스트
# ---------------------------------------------------------------------------


class TestLoadIndex:
    def test_load_creates_new_when_no_file(self, tmp_path):
        """인덱스 파일이 없으면 새 인덱스를 생성한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock(ntotal=0)
            mock_faiss.IndexFlatIP.return_value = mock_index
            mgr = MultiIndexManager(base_dir=base)
            result = mgr.load_index(IndexType.CASE)

        assert IndexType.CASE in mgr.indexes
        assert mgr.metadata[IndexType.CASE] == []

    def test_load_reads_existing_index(self, tmp_path):
        """기존 인덱스 파일을 읽는다."""
        base = str(tmp_path / "faiss_index")
        case_dir = os.path.join(base, "case")
        os.makedirs(case_dir, exist_ok=True)

        # 가짜 인덱스 파일 생성
        with open(os.path.join(case_dir, "index.faiss"), "wb") as f:
            f.write(b"fake")

        mock_index = MagicMock(ntotal=50)
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_faiss.read_index.return_value = mock_index
            mgr = MultiIndexManager(base_dir=base)
            mgr.load_index(IndexType.CASE)

        assert mgr.indexes[IndexType.CASE] == mock_index
        mock_faiss.read_index.assert_called()

    def test_load_reads_metadata(self, tmp_path):
        """메타데이터 JSON을 로드한다."""
        base = str(tmp_path / "faiss_index")
        case_dir = os.path.join(base, "case")
        os.makedirs(case_dir, exist_ok=True)

        meta = [
            {
                "doc_id": "d1",
                "doc_type": "case",
                "source": "test",
                "title": "제목",
                "category": "도로",
                "reliability_score": 0.9,
                "created_at": "2026-01-01",
                "updated_at": "2026-01-01",
                "extras": {},
            }
        ]
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_faiss.IndexFlatIP.return_value = MagicMock(ntotal=0)
            mgr = MultiIndexManager(base_dir=base)
            mgr.load_index(IndexType.CASE)

        assert len(mgr.metadata[IndexType.CASE]) == 1
        assert mgr.metadata[IndexType.CASE][0].doc_id == "d1"


# ---------------------------------------------------------------------------
# save_index 테스트
# ---------------------------------------------------------------------------


class TestSaveIndex:
    def test_save_writes_files(self, tmp_path):
        """인덱스와 메타데이터를 저장한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock(ntotal=1)
            mock_faiss.IndexFlatIP.return_value = mock_index
            mgr = MultiIndexManager(base_dir=base)
            mgr.indexes[IndexType.CASE] = mock_index
            mgr.metadata[IndexType.CASE] = [
                DocumentMetadata(
                    doc_id="d1",
                    doc_type="case",
                    source="test",
                    title="제목",
                    category="도로",
                    reliability_score=0.9,
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )
            ]
            mgr.save_index(IndexType.CASE)

        # write_index 호출 확인
        mock_faiss.write_index.assert_called_once()

        # 메타데이터 파일 확인
        meta_path = os.path.join(base, "case", "metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path, "r") as f:
            saved_meta = json.load(f)
        assert saved_meta[0]["doc_id"] == "d1"

        # 레지스트리 갱신 확인
        reg_path = os.path.join(base, "index_registry.json")
        assert os.path.exists(reg_path)

    def test_save_raises_when_not_loaded(self, tmp_path):
        """인덱스가 로드되지 않은 타입을 저장하면 ValueError."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss"):
            mgr = MultiIndexManager(base_dir=base)

        with pytest.raises(ValueError, match="로드되지 않았습니다"):
            mgr.save_index(IndexType.LAW)


# ---------------------------------------------------------------------------
# search 테스트
# ---------------------------------------------------------------------------


class TestSearch:
    def _make_manager(self, tmp_path):
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 3
            mock_index.search.return_value = (
                np.array([[0.95, 0.80, 0.60]], dtype=np.float32),
                np.array([[0, 1, 2]]),
            )
            mock_faiss.IndexFlatIP.return_value = mock_index
            mgr = MultiIndexManager(base_dir=base)
            mgr.indexes[IndexType.CASE] = mock_index
            mgr.metadata[IndexType.CASE] = [
                DocumentMetadata(
                    doc_id=f"d{i}",
                    doc_type="case",
                    source="test",
                    title=f"제목{i}",
                    category="도로",
                    reliability_score=0.9,
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )
                for i in range(3)
            ]
        return mgr

    def test_search_returns_results(self, tmp_path):
        """검색 결과를 반환한다."""
        mgr = self._make_manager(tmp_path)
        query_vec = np.random.rand(1024).astype(np.float32)
        results = mgr.search(IndexType.CASE, query_vec, top_k=3)
        assert len(results) == 3
        assert abs(results[0]["score"] - 0.95) < 1e-5
        assert results[0]["doc_id"] == "d0"

    def test_search_1d_query_vector(self, tmp_path):
        """1차원 쿼리 벡터를 자동으로 reshape한다."""
        mgr = self._make_manager(tmp_path)
        query_vec = np.random.rand(1024).astype(np.float32)
        results = mgr.search(IndexType.CASE, query_vec, top_k=3)
        # reshape되어 search가 정상 호출되어야 함
        mgr.indexes[IndexType.CASE].search.assert_called_once()
        call_args = mgr.indexes[IndexType.CASE].search.call_args[0]
        assert call_args[0].ndim == 2  # 2D로 reshape됨

    def test_search_empty_index(self, tmp_path):
        """빈 인덱스 검색 시 빈 결과를 반환한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatIP.return_value = mock_index
            mgr = MultiIndexManager(base_dir=base)
            mgr.indexes[IndexType.CASE] = mock_index

        query_vec = np.random.rand(1024).astype(np.float32)
        results = mgr.search(IndexType.CASE, query_vec)
        assert results == []

    def test_search_skips_negative_indices(self, tmp_path):
        """인덱스 -1은 건너뛴다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 2
            mock_index.search.return_value = (
                np.array([[0.9, -1.0]], dtype=np.float32),
                np.array([[0, -1]]),
            )
            mock_faiss.IndexFlatIP.return_value = mock_index
            mgr = MultiIndexManager(base_dir=base)
            mgr.indexes[IndexType.CASE] = mock_index
            mgr.metadata[IndexType.CASE] = [
                DocumentMetadata(
                    doc_id="d0",
                    doc_type="case",
                    source="test",
                    title="제목",
                    category="도로",
                    reliability_score=0.9,
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )
            ]

        query_vec = np.random.rand(1024).astype(np.float32)
        results = mgr.search(IndexType.CASE, query_vec, top_k=2)
        assert len(results) == 1

    def test_search_loads_unloaded_index(self, tmp_path):
        """로드되지 않은 인덱스는 자동 로드 시도한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatIP.return_value = mock_index
            mgr = MultiIndexManager(base_dir=base)
            # CASE 인덱스를 로드하지 않은 상태
            assert IndexType.CASE not in mgr.indexes

            query_vec = np.random.rand(1024).astype(np.float32)
            results = mgr.search(IndexType.CASE, query_vec)
            # 자동 로드 후 빈 결과 반환 (ntotal=0)
            assert results == []
            # 로드가 시도되었으므로 이제 indexes에 존재
            assert IndexType.CASE in mgr.indexes


# ---------------------------------------------------------------------------
# add_documents 테스트
# ---------------------------------------------------------------------------


class TestAddDocuments:
    def test_add_documents(self, tmp_path):
        """벡터와 메타데이터를 추가한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_index.is_trained = True
            mock_faiss.IndexFlatIP.return_value = mock_index
            # isinstance 체크 우회
            mock_faiss.IndexIVFFlat = type("IndexIVFFlat", (), {})
            mgr = MultiIndexManager(base_dir=base, embedding_dim=4)
            mgr.indexes[IndexType.CASE] = mock_index
            mgr.metadata[IndexType.CASE] = []

            vectors = np.random.rand(2, 4).astype(np.float32)
            metas = [
                DocumentMetadata(
                    doc_id=f"d{i}",
                    doc_type="case",
                    source="test",
                    title=f"제목{i}",
                    category="도로",
                    reliability_score=0.9,
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )
                for i in range(2)
            ]
            mgr.add_documents(IndexType.CASE, vectors, metas)

        mock_index.add.assert_called_once()
        assert len(mgr.metadata[IndexType.CASE]) == 2

    def test_add_documents_mismatched_count_raises(self, tmp_path):
        """벡터 수와 메타데이터 수 불일치 시 ValueError."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_faiss.IndexFlatIP.return_value = MagicMock(ntotal=0)
            mgr = MultiIndexManager(base_dir=base, embedding_dim=4)
            mgr.indexes[IndexType.CASE] = MagicMock(ntotal=0)
            mgr.metadata[IndexType.CASE] = []

        vectors = np.random.rand(3, 4).astype(np.float32)
        metas = [MagicMock()]  # 1개만

        with pytest.raises(ValueError, match="일치하지 않습니다"):
            mgr.add_documents(IndexType.CASE, vectors, metas)

    def test_add_documents_wrong_dim_raises(self, tmp_path):
        """벡터 차원 불일치 시 ValueError."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_faiss.IndexFlatIP.return_value = MagicMock(ntotal=0)
            mgr = MultiIndexManager(base_dir=base, embedding_dim=4)
            mgr.indexes[IndexType.CASE] = MagicMock(ntotal=0)
            mgr.metadata[IndexType.CASE] = []

        vectors = np.random.rand(1, 8).astype(np.float32)  # dim=8 != 4
        metas = [MagicMock()]

        with pytest.raises(ValueError, match="차원"):
            mgr.add_documents(IndexType.CASE, vectors, metas)

    def test_add_documents_creates_new_index(self, tmp_path):
        """인덱스가 없으면 새로 생성한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_index.is_trained = True
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.IndexIVFFlat = type("IndexIVFFlat", (), {})
            mgr = MultiIndexManager(base_dir=base, embedding_dim=4)
            # CASE 인덱스가 없는 상태

            vectors = np.random.rand(1, 4).astype(np.float32)
            metas = [
                DocumentMetadata(
                    doc_id="d1",
                    doc_type="case",
                    source="test",
                    title="제목",
                    category="도로",
                    reliability_score=0.9,
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )
            ]
            mgr.add_documents(IndexType.CASE, vectors, metas)

        assert IndexType.CASE in mgr.indexes


# ---------------------------------------------------------------------------
# _maybe_upgrade_to_ivf 테스트
# ---------------------------------------------------------------------------


class TestMaybeUpgradeToIvf:
    def test_skips_below_threshold(self, tmp_path):
        """문서 수가 임계값 미만이면 전환하지 않는다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 100  # < 100,000
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.IndexIVFFlat = type("IndexIVFFlat", (), {})
            mgr = MultiIndexManager(base_dir=base)
            mgr.indexes[IndexType.CASE] = mock_index

            mgr._maybe_upgrade_to_ivf(IndexType.CASE)

        # 인덱스가 변경되지 않아야 함
        assert mgr.indexes[IndexType.CASE] is mock_index

    def test_skips_already_ivf(self, tmp_path):
        """이미 IVFFlat이면 전환하지 않는다."""
        base = str(tmp_path / "faiss_index")

        import faiss as faiss_mod

        with patch("src.inference.index_manager.faiss") as mock_faiss:
            # IVFFlat 타입의 인덱스를 시뮬레이션
            mock_faiss.IndexIVFFlat = type("IndexIVFFlat", (), {})
            ivf_index = mock_faiss.IndexIVFFlat()
            ivf_index.ntotal = 200_000
            mock_faiss.IndexFlatIP.return_value = MagicMock(ntotal=0)
            mgr = MultiIndexManager(base_dir=base)
            mgr.indexes[IndexType.CASE] = ivf_index

            mgr._maybe_upgrade_to_ivf(IndexType.CASE)

        # 인덱스가 변경되지 않아야 함
        assert mgr.indexes[IndexType.CASE] is ivf_index

    def test_skips_when_no_index(self, tmp_path):
        """인덱스가 없으면 아무 것도 하지 않는다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss"):
            mgr = MultiIndexManager(base_dir=base)
            mgr._maybe_upgrade_to_ivf(IndexType.CASE)
        # 에러 없이 완료


# ---------------------------------------------------------------------------
# get_index_stats 테스트
# ---------------------------------------------------------------------------


class TestGetIndexStats:
    def test_returns_stats(self, tmp_path):
        """인덱스 통계를 반환한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 50
            mock_faiss.IndexFlatIP.return_value = mock_index
            mgr = MultiIndexManager(base_dir=base)
            mgr.indexes[IndexType.CASE] = mock_index
            mgr.metadata[IndexType.CASE] = [MagicMock()] * 50

        stats = mgr.get_index_stats()
        assert stats["base_dir"] == base
        assert stats["embedding_dim"] == 1024
        assert stats["indexes"]["case"]["loaded"] is True
        assert stats["indexes"]["case"]["doc_count"] == 50
        assert stats["indexes"]["case"]["metadata_count"] == 50

    def test_returns_unloaded_stats(self, tmp_path):
        """로드되지 않은 인덱스의 통계."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss"):
            mgr = MultiIndexManager(base_dir=base)

        stats = mgr.get_index_stats()
        assert stats["indexes"]["case"]["loaded"] is False
        assert stats["indexes"]["case"]["doc_count"] == 0

    def test_includes_registry_info(self, tmp_path):
        """레지스트리 정보를 포함한다."""
        base = str(tmp_path / "faiss_index")
        with patch("src.inference.index_manager.faiss") as mock_faiss:
            mock_faiss.IndexFlatIP.return_value = MagicMock(ntotal=0)
            mgr = MultiIndexManager(base_dir=base)
            mgr._registry = {
                "indexes": {"case": {"last_updated": "2026-01-01T00:00:00"}},
                "updated_at": None,
            }

        stats = mgr.get_index_stats()
        assert stats["indexes"]["case"]["last_updated"] == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# IndexType enum 테스트
# ---------------------------------------------------------------------------


class TestIndexType:
    def test_all_types(self):
        """모든 IndexType 값이 정의되어 있다."""
        assert IndexType.CASE.value == "case"
        assert IndexType.LAW.value == "law"
        assert IndexType.MANUAL.value == "manual"
        assert IndexType.NOTICE.value == "notice"

    def test_string_enum(self):
        """IndexType은 str Enum이다."""
        assert isinstance(IndexType.CASE, str)
        assert IndexType.CASE == "case"
