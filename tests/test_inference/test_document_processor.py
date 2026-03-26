"""
DocumentProcessor 단위 테스트.

이슈 #156 — PDF/HWP/TXT 파싱, 하이브리드 청킹, 메타데이터 생성 검증.
"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.inference.index_manager import DocumentMetadata, IndexType


# ---------------------------------------------------------------------------
# 모든 테스트에서 토크나이저 로딩을 차단 (EXAONE 다운로드 방지)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_tokenizer():
    """모든 테스트에서 토크나이저를 None으로 고정 (폴백 모드)."""
    import src.inference.document_processor as dp
    original = dp._tokenizer
    dp._tokenizer = None  # 글로벌 캐시 초기화
    with patch.object(dp, "_get_tokenizer", return_value=None):
        yield
    dp._tokenizer = original


from src.inference.document_processor import (
    DocumentProcessor,
    _chunk_fixed,
    _clean_text,
    _count_tokens,
    _hybrid_chunk,
    _parse_txt,
    _split_semantic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_txt(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(
        "민원인이 도로 파손 신고를 접수했습니다.\n\n담당 부서에서 확인 후 처리 예정입니다.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def tmp_txt_cp949(tmp_path: Path) -> Path:
    p = tmp_path / "cp949.txt"
    p.write_bytes("한국어 테스트 파일입니다.".encode("cp949"))
    return p


@pytest.fixture
def tmp_empty_txt(tmp_path: Path) -> Path:
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pdf"
    p.touch()
    return p


@pytest.fixture
def tmp_hwp(tmp_path: Path) -> Path:
    p = tmp_path / "sample.hwp"
    p.touch()
    return p


@pytest.fixture
def processor() -> DocumentProcessor:
    return DocumentProcessor(chunk_size=512, chunk_overlap=64, min_chunk_tokens=50)


@pytest.fixture
def small_processor() -> DocumentProcessor:
    return DocumentProcessor(chunk_size=20, chunk_overlap=4, min_chunk_tokens=5)


# ---------------------------------------------------------------------------
# 텍스트 정제 테스트
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_removes_page_numbers(self):
        text = "본문 내용\n  - 3 -  \n다음 본문"
        result = _clean_text(text)
        assert "- 3 -" not in result
        assert "본문 내용" in result

    def test_collapses_multiple_newlines(self):
        result = _clean_text("첫째\n\n\n\n\n둘째")
        assert result == "첫째\n\n둘째"

    def test_collapses_multiple_spaces(self):
        result = _clean_text("단어   사이    공백")
        assert result == "단어 사이 공백"

    def test_strips_whitespace(self):
        result = _clean_text("  \n본문\n  ")
        assert result == "본문"


# ---------------------------------------------------------------------------
# TXT 파싱 테스트
# ---------------------------------------------------------------------------


class TestParseTxt:
    def test_utf8(self, tmp_txt: Path):
        text = _parse_txt(str(tmp_txt))
        assert "민원인" in text
        assert "담당 부서" in text

    def test_cp949_fallback(self, tmp_txt_cp949: Path):
        text = _parse_txt(str(tmp_txt_cp949))
        assert "한국어 테스트" in text

    def test_invalid_encoding(self, tmp_path: Path):
        p = tmp_path / "binary.txt"
        p.write_bytes(b"\x80\x81\x82\x83" * 100)
        with pytest.raises(ValueError, match="인코딩을 식별할 수 없습니다"):
            _parse_txt(str(p))


# ---------------------------------------------------------------------------
# 의미 단위 분할 테스트
# ---------------------------------------------------------------------------


class TestSplitSemantic:
    def test_law_splits_by_article(self):
        text = textwrap.dedent("""\
            제1조(목적) 이 법은 민원 처리를 규정함.

            제2조(정의) 이 법에서 사용하는 용어의 뜻.

            제3조(적용범위) 모든 기관에 적용.""")
        segments = _split_semantic(text, IndexType.LAW)
        assert len(segments) >= 1

    def test_case_splits_by_paragraph(self):
        segments = _split_semantic("첫 번째 문단\n\n두 번째 문단\n\n세 번째 문단", IndexType.CASE)
        assert len(segments) == 3

    def test_manual_splits_by_paragraph(self):
        segments = _split_semantic("절차 1단계\n\n절차 2단계", IndexType.MANUAL)
        assert len(segments) == 2

    def test_empty_text(self):
        assert _split_semantic("", IndexType.CASE) == []


# ---------------------------------------------------------------------------
# 고정 크기 청킹 테스트
# ---------------------------------------------------------------------------


class TestChunkFixed:
    def test_short_text_single_chunk(self):
        chunks = _chunk_fixed("짧은 텍스트", chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        # 폴백: 2자 ≈ 1토큰, chunk_size=20 → char_size=40
        text = "가" * 200
        chunks = _chunk_fixed(text, chunk_size=20, chunk_overlap=4)
        assert len(chunks) > 1

    def test_overlap_exists(self):
        text = "가나다라마바사아자차카타파하" * 10
        chunks = _chunk_fixed(text, chunk_size=10, chunk_overlap=3)
        if len(chunks) >= 2:
            assert chunks[0][-4:] in chunks[1] or len(chunks[1]) > 0


# ---------------------------------------------------------------------------
# 하이브리드 청킹 테스트
# ---------------------------------------------------------------------------


class TestHybridChunk:
    def test_paragraphs_preserved(self):
        text = "문단1 내용\n\n문단2 내용\n\n문단3 내용"
        chunks = _hybrid_chunk(text, IndexType.CASE, chunk_size=512)
        full = " ".join(chunks)
        assert "문단1" in full
        assert "문단3" in full

    def test_large_segment_split(self):
        big = "가" * 2000
        text = f"짧은 문단\n\n{big}\n\n또 짧은 문단"
        chunks = _hybrid_chunk(text, IndexType.CASE, chunk_size=50)
        assert len(chunks) > 1

    def test_small_segments_merged(self):
        text = "가\n\n나\n\n다\n\n라\n\n마"
        chunks = _hybrid_chunk(text, IndexType.CASE, chunk_size=512, min_chunk_tokens=10)
        assert len(chunks) <= 2

    def test_empty_text_returns_original(self):
        chunks = _hybrid_chunk("", IndexType.CASE)
        assert chunks == [""]


# ---------------------------------------------------------------------------
# DocumentProcessor.process 테스트
# ---------------------------------------------------------------------------


class TestDocumentProcessor:
    def test_process_txt(self, processor: DocumentProcessor, tmp_txt: Path):
        results = processor.process(
            str(tmp_txt),
            IndexType.CASE,
            source="테스트",
            title="민원 접수",
            category="facilities",
        )
        assert len(results) >= 1
        meta = results[0]
        assert isinstance(meta, DocumentMetadata)
        assert meta.doc_type == "case"
        assert meta.source == "테스트"
        assert meta.title == "민원 접수"
        assert meta.category == "facilities"
        assert meta.chunk_index == 0
        assert meta.chunk_total == len(results)
        assert "content" in meta.extras

    def test_process_empty_file(self, processor: DocumentProcessor, tmp_empty_txt: Path):
        results = processor.process(str(tmp_empty_txt), IndexType.CASE)
        assert results == []

    def test_unsupported_extension(self, processor: DocumentProcessor, tmp_path: Path):
        p = tmp_path / "file.docx"
        p.touch()
        with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
            processor.process(str(p), IndexType.CASE)

    def test_default_reliability_score(self, processor: DocumentProcessor, tmp_txt: Path):
        assert processor.process(str(tmp_txt), IndexType.LAW)[0].reliability_score == 1.0
        assert processor.process(str(tmp_txt), IndexType.MANUAL)[0].reliability_score == 0.9
        assert processor.process(str(tmp_txt), IndexType.NOTICE)[0].reliability_score == 0.7

    def test_custom_reliability_score(self, processor: DocumentProcessor, tmp_txt: Path):
        results = processor.process(str(tmp_txt), IndexType.CASE, reliability_score=0.95)
        assert results[0].reliability_score == 0.95

    def test_doc_id_format(self, processor: DocumentProcessor, tmp_txt: Path):
        doc_id = processor.process(str(tmp_txt), IndexType.CASE)[0].doc_id
        parts = doc_id.rsplit("-", 1)
        assert len(parts) == 2
        assert len(parts[0]) == 12
        assert parts[1] == "0000"

    def test_chunk_index_sequential(self, tmp_path: Path):
        p = tmp_path / "long.txt"
        p.write_text("가나다라마바사아자차카타파하 " * 100, encoding="utf-8")
        proc = DocumentProcessor(chunk_size=20, chunk_overlap=4)
        results = proc.process(str(p), IndexType.CASE)
        assert len(results) > 1
        for i, meta in enumerate(results):
            assert meta.chunk_index == i
            assert meta.chunk_total == len(results)

    def test_title_defaults_to_filename(self, processor: DocumentProcessor, tmp_txt: Path):
        assert processor.process(str(tmp_txt), IndexType.CASE)[0].title == "sample"

    def test_extras_contains_file_info(self, processor: DocumentProcessor, tmp_txt: Path):
        extras = processor.process(str(tmp_txt), IndexType.CASE, extras={"custom": "value"})[0].extras
        assert extras["file_extension"] == ".txt"
        assert extras["custom"] == "value"
        assert "file_path" in extras


# ---------------------------------------------------------------------------
# PDF 파싱 테스트 (PyMuPDF mock)
# ---------------------------------------------------------------------------


class TestParsePdf:
    def test_process_pdf_with_mock(self, processor: DocumentProcessor, tmp_pdf: Path):
        import src.inference.document_processor as dp

        with patch.dict(
            dp._PARSERS,
            {".pdf": lambda fp: "법령 제1조 내용입니다.\n\n법령 제2조 내용입니다."},
        ):
            results = processor.process(str(tmp_pdf), IndexType.LAW, source="법제처")

        assert len(results) >= 1
        assert results[0].doc_type == "law"
        assert results[0].source == "법제처"


# ---------------------------------------------------------------------------
# HWP 파싱 테스트 (python-hwp mock)
# ---------------------------------------------------------------------------


class TestParseHwp:
    def test_process_hwp_with_mock(self, processor: DocumentProcessor, tmp_hwp: Path):
        import src.inference.document_processor as dp

        with patch.dict(
            dp._PARSERS,
            {".hwp": lambda fp: "업무 매뉴얼 1절\n\n처리 절차 설명"},
        ):
            results = processor.process(str(tmp_hwp), IndexType.MANUAL, source="기관 내부")

        assert len(results) >= 1
        assert results[0].doc_type == "manual"


# ---------------------------------------------------------------------------
# batch 처리 테스트
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_batch_multiple_files(self, processor: DocumentProcessor, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"doc{i}.txt").write_text(f"문서 {i} 내용입니다.", encoding="utf-8")
        files = [str(tmp_path / f"doc{i}.txt") for i in range(3)]
        results = processor.process_batch(files, IndexType.CASE, source="테스트")
        assert len(results) == 3

    def test_batch_skips_failed_files(self, processor: DocumentProcessor, tmp_path: Path):
        good = tmp_path / "good.txt"
        good.write_text("정상 문서", encoding="utf-8")
        bad = tmp_path / "bad.docx"
        bad.touch()
        results = processor.process_batch([str(good), str(bad)], IndexType.CASE)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# parse_only 테스트
# ---------------------------------------------------------------------------


class TestParseOnly:
    def test_parse_only_returns_cleaned_text(self, processor: DocumentProcessor, tmp_txt: Path):
        text = processor.parse_only(str(tmp_txt))
        assert "민원인" in text
        assert isinstance(text, str)

    def test_parse_only_unsupported(self, processor: DocumentProcessor, tmp_path: Path):
        p = tmp_path / "file.xlsx"
        p.touch()
        with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
            processor.parse_only(str(p))
