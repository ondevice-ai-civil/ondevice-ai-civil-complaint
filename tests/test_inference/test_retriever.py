"""
CivilComplaintRetriever 단위 테스트.

FAISS / SentenceTransformer를 Mock으로 대체하여 GPU 없이 실행 가능.
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 무거운 의존성 mock 등록 (import 전에 수행)
# ---------------------------------------------------------------------------
_st_mock = MagicMock()
sys.modules.setdefault("sentence_transformers", _st_mock)

_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, MagicMock)
if not _faiss_is_real:
    _faiss_mock = MagicMock()
    _faiss_mock.IndexFlatIP = MagicMock
    _faiss_mock.IndexIVFFlat = type("IndexIVFFlat", (), {})
    sys.modules.setdefault("faiss", _faiss_mock)

from src.inference.retriever import CivilComplaintRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sentence_model():
    """SentenceTransformer mock을 반환한다."""
    model = MagicMock()
    model.encode.return_value = np.random.rand(3, 1024).astype("float32")
    return model


@pytest.fixture
def retriever(mock_sentence_model):
    """기본 CivilComplaintRetriever 인스턴스 (인덱스 미로드)."""
    with patch("src.inference.retriever.SentenceTransformer", return_value=mock_sentence_model):
        ret = CivilComplaintRetriever(model_name="mock-model")
    return ret


# ---------------------------------------------------------------------------
# _parse_complaint 테스트
# ---------------------------------------------------------------------------


class TestParseComplaint:
    def test_parse_with_user_and_complaint(self, retriever):
        """[|user|] 블록에서 민원 내용을 추출한다."""
        text = "[|system|]시스템[|endofturn|][|user|]민원 내용: 도로가 파손되었습니다.[|endofturn|]"
        result = retriever._parse_complaint(text)
        assert result == "도로가 파손되었습니다."

    def test_parse_with_user_without_complaint_label(self, retriever):
        """민원 내용: 라벨이 없으면 user 블록 전체를 반환한다."""
        text = "[|user|]도로 파손 신고합니다.[|endofturn|]"
        result = retriever._parse_complaint(text)
        assert result == "도로 파손 신고합니다."

    def test_parse_without_user_tag(self, retriever):
        """[|user|] 태그가 없으면 원본 텍스트를 반환한다."""
        text = "일반 텍스트입니다."
        result = retriever._parse_complaint(text)
        assert result == "일반 텍스트입니다."

    def test_parse_empty_string(self, retriever):
        """빈 문자열은 그대로 반환한다."""
        assert retriever._parse_complaint("") == ""

    def test_parse_malformed_template(self, retriever):
        """파싱 에러 시 원본 텍스트를 반환한다."""
        text = "[|user|]"  # endofturn 없이 끝남
        result = retriever._parse_complaint(text)
        # 에러 없이 결과 반환
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_index 테스트
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_build_index_from_jsonl(self, retriever, tmp_path):
        """JSONL 데이터에서 인덱스를 빌드한다."""
        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps(
                {
                    "id": "1",
                    "category": "도로",
                    "text": "[|user|]민원 내용: 도로 파손[|endofturn|][|assistant|]복구 예정[|endofturn|]",
                }
            ),
            json.dumps(
                {
                    "id": "2",
                    "category": "환경",
                    "text": "[|user|]민원 내용: 악취 발생[|endofturn|][|assistant|]조사 예정[|endofturn|]",
                }
            ),
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        # encode가 적절한 shape 반환하도록 설정
        retriever.model.encode.return_value = np.random.rand(2, 1024).astype("float32")

        # faiss.IndexFlatIP mock 설정
        mock_index = MagicMock()
        with patch("src.inference.retriever.faiss") as faiss_mock:
            faiss_mock.IndexFlatIP.return_value = mock_index
            retriever.build_index(str(data_file))

        assert len(retriever.metadata) == 2
        assert retriever.metadata[0]["complaint"] == "도로 파손"
        assert retriever.metadata[1]["answer"] == "조사 예정"
        mock_index.add.assert_called_once()

    def test_build_index_nonexistent_path(self, retriever):
        """존재하지 않는 경로에서는 인덱스를 빌드하지 않는다."""
        retriever.build_index("/nonexistent/path.jsonl")
        assert retriever.index is None
        assert retriever.metadata == []

    def test_build_index_with_save(self, retriever, tmp_path):
        """save_path 전달 시 인덱스를 저장한다."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            json.dumps({"id": "1", "category": "도로", "text": "일반 텍스트"}),
            encoding="utf-8",
        )

        retriever.model.encode.return_value = np.random.rand(1, 1024).astype("float32")
        mock_index = MagicMock()
        save_path = str(tmp_path / "index.faiss")

        with patch("src.inference.retriever.faiss") as faiss_mock:
            faiss_mock.IndexFlatIP.return_value = mock_index
            retriever.build_index(str(data_file), save_path=save_path)

        # save_index가 호출되어야 함
        faiss_mock.write_index.assert_called_once()

    def test_build_index_skips_bad_lines(self, retriever, tmp_path):
        """잘못된 JSON 라인은 건너뛴다."""
        data_file = tmp_path / "train.jsonl"
        lines = [
            "not valid json",
            json.dumps({"id": "1", "category": "도로", "text": "정상 텍스트"}),
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        retriever.model.encode.return_value = np.random.rand(1, 1024).astype("float32")
        mock_index = MagicMock()

        with patch("src.inference.retriever.faiss") as faiss_mock:
            faiss_mock.IndexFlatIP.return_value = mock_index
            retriever.build_index(str(data_file))

        assert len(retriever.metadata) == 1


# ---------------------------------------------------------------------------
# save_index / load_index 테스트
# ---------------------------------------------------------------------------


class TestSaveLoadIndex:
    def test_save_index_raises_without_index(self, retriever):
        """인덱스가 없으면 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="Index not built"):
            retriever.save_index("/tmp/test.faiss")

    def test_save_index_creates_directory(self, retriever, tmp_path):
        """저장 시 디렉토리를 생성한다."""
        retriever.index = MagicMock()
        retriever.metadata = [{"id": "1", "complaint": "테스트"}]
        save_path = str(tmp_path / "subdir" / "index.faiss")

        with patch("src.inference.retriever.faiss") as faiss_mock:
            retriever.save_index(save_path)

        faiss_mock.write_index.assert_called_once()
        meta_path = save_path + ".meta.json"
        assert os.path.exists(meta_path)

    def test_load_index(self, retriever, tmp_path):
        """인덱스와 메타데이터를 로드한다."""
        index_path = str(tmp_path / "index.faiss")
        meta_path = index_path + ".meta.json"

        # 메타데이터 파일 생성
        meta = [{"id": "1", "complaint": "테스트", "answer": "답변"}]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        mock_index = MagicMock()
        with patch("src.inference.retriever.faiss") as faiss_mock:
            faiss_mock.read_index.return_value = mock_index
            retriever.load_index(index_path)

        assert retriever.index == mock_index
        assert len(retriever.metadata) == 1
        assert retriever.metadata[0]["complaint"] == "테스트"


# ---------------------------------------------------------------------------
# search 테스트
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_results(self, retriever):
        """검색 결과를 정상 반환한다."""
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.95, 0.80]]),
            np.array([[0, 1]]),
        )
        retriever.index = mock_index
        retriever.metadata = [
            {"id": "1", "category": "도로", "complaint": "도로 파손", "answer": "복구"},
            {"id": "2", "category": "환경", "complaint": "악취", "answer": "조사"},
        ]
        retriever.model.encode.return_value = np.random.rand(1, 1024).astype("float32")

        results = retriever.search("도로 파손", top_k=2)
        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert results[0]["id"] == "1"

    def test_search_without_index(self, retriever):
        """인덱스가 없으면 빈 리스트를 반환한다."""
        results = retriever.search("테스트 쿼리")
        assert results == []

    def test_search_skips_invalid_indices(self, retriever):
        """잘못된 인덱스(-1)는 건너뛴다."""
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.90, -1.0]]),
            np.array([[0, -1]]),
        )
        retriever.index = mock_index
        retriever.metadata = [{"id": "1", "complaint": "테스트", "answer": "답변"}]
        retriever.model.encode.return_value = np.random.rand(1, 1024).astype("float32")

        results = retriever.search("테스트", top_k=2)
        assert len(results) == 1

    def test_search_skips_out_of_range_index(self, retriever):
        """메타데이터 범위를 초과하는 인덱스는 건너뛴다."""
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.90, 0.80]]),
            np.array([[0, 999]]),  # 999는 범위 초과
        )
        retriever.index = mock_index
        retriever.metadata = [{"id": "1", "complaint": "테스트", "answer": "답변"}]
        retriever.model.encode.return_value = np.random.rand(1, 1024).astype("float32")

        results = retriever.search("테스트", top_k=2)
        assert len(results) == 1
