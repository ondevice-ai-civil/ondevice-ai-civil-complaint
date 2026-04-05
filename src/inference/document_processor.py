"""
DocumentProcessor: 다형식 문서 파싱 및 하이브리드 청킹 모듈.

이슈 #156 — PDF(PyMuPDF), HWP, TXT 파서를 통합하고,
의미 단위(조/항/호, 문단) + 고정 크기(512토큰, 128토큰 오버랩) 하이브리드 청킹을 수행한다.

ADR-004 Section B.3 참조.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.inference.index_manager import DocumentMetadata, IndexType

# ---------------------------------------------------------------------------
# 토크나이저 (토큰 기반 청킹용)
# ---------------------------------------------------------------------------

_LOAD_FAILED = object()  # 센티널: 로드 실패 확정
_tokenizer = None  # None=미시도, _LOAD_FAILED=실패확정


def _get_tokenizer():
    """transformers 토크나이저를 lazy-load한다.

    EXAONE 토크나이저가 없으면 단순 문자 기반 근사로 폴백.
    로드 실패 시 센티널을 설정하여 재시도를 방지한다.
    """
    global _tokenizer
    if _tokenizer is _LOAD_FAILED:
        return None
    if _tokenizer is not None:
        return _tokenizer
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            "LGAI-EXAONE/EXAONE-Deep-7.8B",
            trust_remote_code=True,
        )
        logger.info("EXAONE 토크나이저 로드 완료")
    except Exception:
        logger.warning("EXAONE 토크나이저 로드 실패 — 문자 기반 폴백 사용")
        _tokenizer = _LOAD_FAILED
    return None if _tokenizer is _LOAD_FAILED else _tokenizer


def _count_tokens(text: str) -> int:
    """텍스트의 토큰 수를 반환한다."""
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    # 폴백: 한국어 평균 1.5자 ≈ 1토큰 근사
    return max(1, len(text) // 2)


# ---------------------------------------------------------------------------
# 파서 (PDF / HWP / TXT)
# ---------------------------------------------------------------------------


def _parse_pdf_pages(file_path: str) -> List[Tuple[int, str]]:
    """PyMuPDF로 PDF의 페이지별 텍스트를 추출한다."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("PyMuPDF가 설치되지 않았습니다: pip install PyMuPDF") from e

    pages: List[Tuple[int, str]] = []
    with fitz.open(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((page_number, text))
    return pages


def _parse_pdf(file_path: str) -> str:
    """PyMuPDF로 PDF 텍스트를 추출한다."""
    pages = _parse_pdf_pages(file_path)
    return "\n\n".join(text for _, text in pages)


def _parse_hwp(file_path: str) -> str:
    """HWP 텍스트를 추출한다.

    pyhwp 또는 호환 라이브러리가 필요하다. PyPI에 안정적인 HWP 파서가
    없으므로 런타임 ImportError로 안내한다.
    """
    try:
        import hwp
    except ImportError as e:
        raise ImportError(
            "HWP 파서가 설치되지 않았습니다. " "pyhwp 또는 호환 라이브러리를 설치해 주세요."
        ) from e

    doc = hwp.open(file_path)
    try:
        paragraphs: List[str] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs)
    finally:
        if hasattr(doc, "close"):
            doc.close()


def _parse_txt(file_path: str) -> str:
    """TXT 파일을 UTF-8로 읽는다. 실패 시 cp949 폴백."""
    path = Path(file_path)
    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"텍스트 파일 인코딩을 식별할 수 없습니다: {file_path}")


_PARSERS = {
    ".pdf": _parse_pdf,
    ".hwp": _parse_hwp,
    ".txt": _parse_txt,
}

_PAGE_PARSERS = {
    ".pdf": _parse_pdf_pages,
}


# ---------------------------------------------------------------------------
# 텍스트 정제
# ---------------------------------------------------------------------------

# 페이지 번호, 머리글/바닥글 패턴
_HEADER_FOOTER_RE = re.compile(
    r"^[\s]*[-–—]?\s*\d+\s*[-–—]?\s*$",  # 페이지 번호만 있는 줄
    re.MULTILINE,
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _clean_text(text: str) -> str:
    """추출된 원시 텍스트를 정제한다."""
    text = _HEADER_FOOTER_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 의미 단위 분할
# ---------------------------------------------------------------------------

# 법령: 제N조, 제N항, 제N호
_LAW_ARTICLE_RE = re.compile(r"(?=\n\s*제\s*\d+\s*조(?:의\d+)?\s*[\(（])")
# 문단 분할 (빈 줄 기준)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _split_semantic(text: str, doc_type: IndexType) -> List[str]:
    """문서 타입에 따라 의미 단위로 분할한다.

    - LAW: 조/항 단위
    - MANUAL/NOTICE: 문단(빈 줄) 단위
    - CASE: 문단 단위
    """
    if doc_type == IndexType.LAW:
        segments = _LAW_ARTICLE_RE.split(text)
    else:
        segments = _PARAGRAPH_RE.split(text)

    return [s.strip() for s in segments if s.strip()]


# ---------------------------------------------------------------------------
# 고정 크기 청킹 (토큰 기반)
# ---------------------------------------------------------------------------


def _chunk_fixed(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[str]:
    """토큰 기반 고정 크기 청킹.

    토크나이저가 로드된 경우 정확한 토큰 분할,
    그렇지 않으면 문자 기반 근사 분할을 수행한다.
    """
    # overlap이 chunk_size 이상이면 보정 (무한루프 방지)
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    tok = _get_tokenizer()

    if tok is not None:
        token_ids = tok.encode(text, add_special_tokens=False)
        if len(token_ids) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        step = max(1, chunk_size - chunk_overlap)
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            chunk_text = tok.decode(token_ids[start:end], skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            if end >= len(token_ids):
                break
            start += step
        return chunks

    # 폴백: 문자 기반 근사 (한국어 ~2자 ≈ 1토큰)
    char_size = chunk_size * 2
    char_overlap = chunk_overlap * 2
    if len(text) <= char_size:
        return [text]

    chunks = []
    start = 0
    step = max(1, char_size - char_overlap)
    while start < len(text):
        end = min(start + char_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(text):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# 하이브리드 청킹
# ---------------------------------------------------------------------------


def _hybrid_chunk(
    text: str,
    doc_type: IndexType,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    min_chunk_tokens: int = 50,
) -> List[str]:
    """의미 단위 + 고정 크기 하이브리드 청킹.

    1단계: 의미 단위 분할 (조/항, 문단)
    2단계: 큰 세그먼트는 고정 크기로 재분할
    3단계: 작은 세그먼트는 인접 세그먼트와 병합
    """
    if not text.strip():
        return []

    segments = _split_semantic(text, doc_type)

    if not segments:
        return _chunk_fixed(text, chunk_size, chunk_overlap)

    chunks: List[str] = []
    buffer = ""

    for segment in segments:
        seg_tokens = _count_tokens(segment)

        if seg_tokens > chunk_size:
            # 버퍼에 쌓인 것 먼저 처리
            if buffer.strip():
                if _count_tokens(buffer) > chunk_size:
                    chunks.extend(_chunk_fixed(buffer, chunk_size, chunk_overlap))
                else:
                    chunks.append(buffer.strip())
                buffer = ""
            # 큰 세그먼트는 고정 크기로 분할
            chunks.extend(_chunk_fixed(segment, chunk_size, chunk_overlap))
        elif _count_tokens(buffer + "\n\n" + segment if buffer else segment) > chunk_size:
            # 버퍼 + 현재 세그먼트가 chunk_size를 초과하면 버퍼 flush
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = segment
        else:
            # 버퍼에 추가
            buffer = buffer + "\n\n" + segment if buffer else segment

    # 남은 버퍼 처리
    if buffer.strip():
        if _count_tokens(buffer) > chunk_size:
            chunks.extend(_chunk_fixed(buffer, chunk_size, chunk_overlap))
        else:
            chunks.append(buffer.strip())

    # 최소 토큰 미만 청크 병합
    merged: List[str] = []
    for chunk in chunks:
        if merged and _count_tokens(chunk) < min_chunk_tokens:
            candidate = merged[-1] + "\n\n" + chunk
            if _count_tokens(candidate) <= chunk_size:
                merged[-1] = candidate
                continue
        merged.append(chunk)

    return merged if merged else []


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------


@dataclass
class BatchResult:
    """process_batch 반환 타입. 성공/실패 정보를 모두 포함한다."""

    succeeded: List[DocumentMetadata] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)  # [(file_path, error)]

    @property
    def total_chunks(self) -> int:
        return len(self.succeeded)

    @property
    def success_count(self) -> int:
        return self.total_chunks - len(self.failed) if not self.failed else self._count_files()

    def _count_files(self) -> int:
        seen = set()
        for m in self.succeeded:
            seen.add(m.extras.get("file_path", ""))
        return len(seen)


# ---------------------------------------------------------------------------
# DocumentProcessor
# ---------------------------------------------------------------------------

# 문서 타입별 기본 신뢰도 (ADR-004 Table)
_DEFAULT_RELIABILITY: Dict[IndexType, float] = {
    IndexType.CASE: 0.6,
    IndexType.LAW: 1.0,
    IndexType.MANUAL: 0.9,
    IndexType.NOTICE: 0.7,
}


class DocumentProcessor:
    """다형식 문서를 파싱하고 청크 분할하여 DocumentMetadata 리스트를 반환한다.

    Parameters
    ----------
    chunk_size : int
        청크당 최대 토큰 수 (기본 512).
    chunk_overlap : int
        청크 간 오버랩 토큰 수 (기본 128, ADR-004).
    min_chunk_tokens : int
        최소 청크 크기. 이보다 작으면 인접 청크와 병합 (기본 50).
    """

    SUPPORTED_EXTENSIONS = frozenset(_PARSERS.keys())

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_tokens: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        file_path: str,
        doc_type: IndexType,
        *,
        source: str = "",
        title: Optional[str] = None,
        category: str = "",
        reliability_score: Optional[float] = None,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> List[DocumentMetadata]:
        """파일을 파싱 → 정제 → 청킹하여 DocumentMetadata 리스트를 반환한다.

        Parameters
        ----------
        file_path : str
            파싱할 원본 문서 경로.
        doc_type : IndexType
            문서의 semantic type.
        document_id : Optional[str]
            원본 문서 단위의 안정 ID. 지정되면 생성되는 모든 chunk가 같은 doc_id를 공유한다.

        Returns
        -------
        List[DocumentMetadata]
            청크별 메타데이터 리스트. doc_id는 원본 문서 단위로 동일하며,
            청크는 chunk_index로 구분한다.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in _PARSERS:
            raise ValueError(
                f"지원하지 않는 파일 형식: {ext} "
                f"(지원: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))})"
            )

        logger.info(f"문서 파싱 시작: {file_path} (type={doc_type.value})")

        units: List[Tuple[Optional[int], str]] = []
        page_parser = _PAGE_PARSERS.get(ext)
        if page_parser is not None:
            for page_number, page_text in page_parser(file_path):
                cleaned_page = _clean_text(page_text)
                if cleaned_page:
                    units.append((page_number, cleaned_page))
        else:
            raw_text = _PARSERS[ext](file_path)
            if not raw_text.strip():
                logger.warning(f"빈 문서: {file_path}")
                return []

            cleaned = _clean_text(raw_text)
            if not cleaned:
                logger.warning(f"정제 후 빈 문서: {file_path}")
                return []
            units.append((None, cleaned))

        if not units:
            logger.warning(f"정제 후 빈 문서: {file_path}")
            return []

        chunk_entries: List[Tuple[str, Optional[int]]] = []
        for page_number, cleaned_text in units:
            chunks = _hybrid_chunk(
                cleaned_text,
                doc_type,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_chunk_tokens=self.min_chunk_tokens,
            )
            for chunk in chunks:
                chunk_entries.append((chunk, page_number))

        if not chunk_entries:
            logger.warning(f"청킹 결과 없음: {file_path}")
            return []

        logger.info(f"청킹 완료: {len(chunk_entries)}개 청크 생성 ({file_path})")

        # 4. 메타데이터 생성
        now_iso = datetime.now(timezone.utc).isoformat()
        doc_title = title or path.stem
        score = (
            reliability_score
            if reliability_score is not None
            else _DEFAULT_RELIABILITY.get(doc_type, 0.5)
        )
        # doc_id: 원본 문서 단위 안정 ID (모든 청크가 동일)
        doc_id = (
            document_id or hashlib.sha256(f"{file_path}:{doc_type.value}".encode()).hexdigest()[:12]
        )

        results: List[DocumentMetadata] = []
        for idx, (chunk, page_number) in enumerate(chunk_entries):
            chunk_extras = dict(extras or {})
            chunk_extras.update(
                {
                    "chunk_text": chunk,
                    "file_path": str(path),
                    "file_extension": ext,
                    "chunk_id": f"{doc_id}:{idx}",
                }
            )
            if page_number is not None:
                chunk_extras["page"] = page_number

            meta = DocumentMetadata(
                doc_id=doc_id,
                doc_type=doc_type.value,
                source=source,
                title=doc_title,
                category=category,
                reliability_score=score,
                created_at=now_iso,
                updated_at=now_iso,
                valid_from=valid_from,
                valid_until=valid_until,
                chunk_index=idx,
                chunk_total=len(chunk_entries),
                extras=chunk_extras,
            )
            results.append(meta)

        return results

    def process_batch(
        self,
        file_paths: List[str],
        doc_type: IndexType,
        **kwargs: Any,
    ) -> BatchResult:
        """여러 파일을 일괄 처리한다.

        Returns
        -------
        BatchResult
            성공한 청크 리스트와 실패한 파일 정보를 모두 포함.
        """
        result = BatchResult()
        for fp in file_paths:
            try:
                chunks = self.process(fp, doc_type, **kwargs)
                result.succeeded.extend(chunks)
            except Exception as e:
                logger.error(f"문서 처리 실패: {fp} — {e}")
                result.failed.append((fp, str(e)))
        logger.info(
            f"배치 처리 완료: {len(file_paths)}개 파일 → "
            f"{result.total_chunks}개 청크, {len(result.failed)}개 실패"
        )
        return result

    def parse_only(self, file_path: str) -> str:
        """파싱 + 정제만 수행하고 텍스트를 반환한다 (청킹 없음)."""
        ext = Path(file_path).suffix.lower()
        if ext not in _PARSERS:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")
        raw = _PARSERS[ext](file_path)
        return _clean_text(raw)
