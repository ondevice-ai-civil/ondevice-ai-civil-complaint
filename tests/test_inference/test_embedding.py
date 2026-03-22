"""
EmbeddingPipeline 단위 테스트.

SentenceTransformer 모델은 mock으로 대체하여 실제 모델 로딩 없이 테스트한다.
"""

import json
import sys
import types
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# src.data_collection_preprocessing 패키지의 __init__.py가 dotenv, seoul_api_collector 등
# 테스트 환경에 없을 수 있는 모듈을 import하므로, 미리 mock 등록한다.
for _mod_name in (
    "dotenv",
    "sentence_transformers",
    "src.data_collection_preprocessing.seoul_api_collector",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()


# ---------------------------------------------------------------------------
# Mock 헬퍼
# ---------------------------------------------------------------------------


def _mock_encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True):
    """SentenceTransformer.encode() mock 구현."""
    n = len(texts)
    rng = np.random.default_rng(12345)
    vecs = rng.standard_normal((n, 1024)).astype(np.float32)
    if normalize_embeddings:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs = vecs / norms
    return vecs


@pytest.fixture
def pipeline():
    """SentenceTransformer를 mock한 EmbeddingPipeline 인스턴스."""
    with mock.patch("src.data_collection_preprocessing.embedding.SentenceTransformer") as MockST:
        mock_model = MockST.return_value
        mock_model.encode = _mock_encode
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.device = "cpu"

        from src.data_collection_preprocessing.embedding import EmbeddingPipeline
        p = EmbeddingPipeline()
        yield p


# ---------------------------------------------------------------------------
# 테스트용 JSONL 데이터
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "[|system|]당신은 민원 상담 AI입니다.[|endofturn|]\n"
    "[|user|][카테고리: 세금]\n"
    "민원 내용: 재산세 납부 기한이 언제인가요?[|endofturn|]\n"
    "[|assistant|]재산세 납부 기한은 매년 7월과 9월입니다.[|endofturn|]"
)

SAMPLE_TEXT_NO_COMPLAINT = (
    "[|system|]당신은 민원 상담 AI입니다.[|endofturn|]\n"
    "[|user|]안녕하세요[|endofturn|]\n"
    "[|assistant|]안녕하세요, 무엇을 도와드릴까요?[|endofturn|]"
)

SAMPLE_TEXT_NO_ASSISTANT = (
    "[|system|]시스템[|endofturn|]\n"
    "[|user|][카테고리: 교통]\n"
    "민원 내용: 도로 파손 신고[|endofturn|]"
)


def _make_jsonl_file(records, tmp_path):
    """tmp_path에 JSONL 파일을 생성하고 경로를 반환한다."""
    jsonl_path = tmp_path / "test_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(jsonl_path)


# ---------------------------------------------------------------------------
# 1. _parse_complaint_text 테스트
# ---------------------------------------------------------------------------


from src.data_collection_preprocessing.embedding import EmbeddingPipeline as _EP


class TestParseComplaintText:
    def test_normal_parse(self):
        result = _EP._parse_complaint_text(SAMPLE_TEXT)
        assert result is not None
        assert "재산세 납부 기한" in result

    def test_no_complaint_keyword(self):
        result = _EP._parse_complaint_text(SAMPLE_TEXT_NO_COMPLAINT)
        assert result is None

    def test_empty_string(self):
        result = _EP._parse_complaint_text("")
        assert result is None

    def test_no_user_block(self):
        result = _EP._parse_complaint_text("민원 내용: 테스트")
        assert result is None


# ---------------------------------------------------------------------------
# 2. _parse_answer_text 테스트
# ---------------------------------------------------------------------------


class TestParseAnswerText:
    def test_normal_parse(self):
        result = _EP._parse_answer_text(SAMPLE_TEXT)
        assert result is not None
        assert "재산세 납부 기한" in result
        assert "7월" in result

    def test_no_assistant_block(self):
        result = _EP._parse_answer_text(SAMPLE_TEXT_NO_ASSISTANT)
        assert result is None

    def test_empty_string(self):
        result = _EP._parse_answer_text("")
        assert result is None


# ---------------------------------------------------------------------------
# 3. _parse_category_from_text 테스트
# ---------------------------------------------------------------------------


class TestParseCategoryFromText:
    def test_normal_parse(self):
        result = _EP._parse_category_from_text(SAMPLE_TEXT)
        assert result == "세금"

    def test_no_category(self):
        result = _EP._parse_category_from_text("카테고리 없는 텍스트")
        assert result is None

    def test_category_with_spaces(self):
        text = "[카테고리:  도로/교통 ]"
        result = _EP._parse_category_from_text(text)
        assert result == "도로/교통"


# ---------------------------------------------------------------------------
# 4. load_jsonl 테스트
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_normal_load(self, pipeline, tmp_path):
        records = [
            {"text": SAMPLE_TEXT, "category": "세금", "id": "test-001"},
            {"text": SAMPLE_TEXT, "category": "세금", "id": "test-002"},
        ]
        jsonl_path = _make_jsonl_file(records, tmp_path)
        result = pipeline.load_jsonl(jsonl_path)
        assert len(result) == 2
        assert result[0]["id"] == "test-001"
        assert result[0]["category"] == "세금"
        assert "재산세" in result[0]["complaint_text"]
        assert result[0]["answer_text"] != ""

    def test_skip_unparseable(self, pipeline, tmp_path):
        records = [
            {"text": SAMPLE_TEXT, "category": "세금", "id": "good-001"},
            {"text": SAMPLE_TEXT_NO_COMPLAINT, "category": "기타", "id": "bad-001"},
        ]
        jsonl_path = _make_jsonl_file(records, tmp_path)
        result = pipeline.load_jsonl(jsonl_path)
        assert len(result) == 1
        assert result[0]["id"] == "good-001"

    def test_empty_file(self, pipeline, tmp_path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.touch()
        result = pipeline.load_jsonl(str(jsonl_path))
        assert result == []

    def test_category_fallback_to_text_parsing(self, pipeline, tmp_path):
        """JSONL에 category 필드가 없으면 텍스트에서 파싱한다."""
        records = [{"text": SAMPLE_TEXT, "id": "no-cat-001"}]
        jsonl_path = _make_jsonl_file(records, tmp_path)
        result = pipeline.load_jsonl(jsonl_path)
        assert len(result) == 1
        assert result[0]["category"] == "세금"


# ---------------------------------------------------------------------------
# 5. embed_documents 테스트
# ---------------------------------------------------------------------------


class TestEmbedDocuments:
    def test_output_shape(self, pipeline):
        texts = ["테스트 문장 1", "테스트 문장 2", "테스트 문장 3"]
        embeddings = pipeline.embed_documents(texts)
        assert embeddings.shape == (3, 1024)

    def test_output_dtype(self, pipeline):
        embeddings = pipeline.embed_documents(["테스트"])
        assert embeddings.dtype == np.float32

    def test_normalization(self, pipeline):
        embeddings = pipeline.embed_documents(["정규화 검증"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_single_document(self, pipeline):
        embeddings = pipeline.embed_documents(["단일 문서"])
        assert embeddings.shape == (1, 1024)

    def test_batch_consistency(self, pipeline):
        """배치 크기에 관계없이 결과 shape이 일관된다."""
        texts = [f"문장 {i}" for i in range(10)]
        embeddings = pipeline.embed_documents(texts, batch_size=3)
        assert embeddings.shape == (10, 1024)


# ---------------------------------------------------------------------------
# 6. process_jsonl 통합 테스트
# ---------------------------------------------------------------------------


class TestProcessJsonl:
    def test_full_pipeline(self, pipeline, tmp_path):
        records = [
            {"text": SAMPLE_TEXT, "category": "세금", "id": f"test-{i:03d}"}
            for i in range(5)
        ]
        jsonl_path = _make_jsonl_file(records, tmp_path)
        embeddings, metadata_list = pipeline.process_jsonl(jsonl_path)

        # 벡터 shape 검증
        assert embeddings.shape == (5, 1024)
        assert embeddings.dtype == np.float32

        # 메타데이터 수 일치
        assert len(metadata_list) == 5

        # 메타데이터 내용 검증
        meta = metadata_list[0]
        assert meta.doc_id == "test-000"
        assert meta.doc_type == "case"
        assert meta.category == "세금"
        assert meta.source == "AI Hub"
        assert meta.extras["complaint_text"] != ""
        assert meta.extras["answer_text"] != ""

    def test_empty_jsonl_raises(self, pipeline, tmp_path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.touch()
        with pytest.raises(ValueError, match="유효한 레코드가 없습니다"):
            pipeline.process_jsonl(str(jsonl_path))
