from datetime import datetime
from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.inference.index_manager import DocumentMetadata, IndexType

class RetrievedCase(BaseModel):
    id: Optional[str] = None
    category: Optional[str] = None
    complaint: str
    answer: str
    score: float

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="The input prompt for generation.")
    max_tokens: int = Field(default=512, gt=0, le=4096, description="Maximum number of tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter.")
    stream: bool = Field(default=False, description="Whether to stream the output using SSE.")
    stop: Optional[List[str]] = Field(default=None, description="List of stop sequences.")
    use_rag: bool = Field(default=True, description="Whether to use RAG (Retrieval-Augmented Generation).")

class GenerateResponse(BaseModel):
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    retrieved_cases: Optional[List[RetrievedCase]] = None
    search_results: Optional[List["SearchResult"]] = None

class StreamResponse(BaseModel):
    request_id: str
    text: str
    finished: bool = False
    retrieved_cases: Optional[List[RetrievedCase]] = None
    search_results: Optional[List["SearchResult"]] = None


# ---------------------------------------------------------------------------
# 분류 스키마 (이슈 #56)
# ---------------------------------------------------------------------------

VALID_CATEGORIES = Literal[
    "environment", "traffic", "facilities", "civil_service", "welfare", "other"
]


class ClassificationResult(BaseModel):
    """분류 결과 모델. classifier 에이전트의 JSON 출력을 검증한다."""

    category: VALID_CATEGORIES
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ClassifyRequest(BaseModel):
    """민원 분류 전용 요청 모델."""

    prompt: str = Field(..., min_length=1, max_length=10000, description="분류할 민원 텍스트")


class ClassifyResponse(BaseModel):
    """민원 분류 응답 모델."""

    request_id: str
    classification: Optional[ClassificationResult] = None
    classification_error: Optional[str] = None
    prompt_tokens: int
    completion_tokens: int


# ---------------------------------------------------------------------------
# 확장 스키마 (이슈 #151)
# ---------------------------------------------------------------------------


class DocumentMetadataSchema(BaseModel):
    """Pydantic v2 기반 문서 메타데이터 스키마.

    index_manager.DocumentMetadata(dataclass)의 API 표현 모델로,
    직렬화/역직렬화 및 유효성 검사를 제공한다.
    """

    doc_id: str
    source_type: IndexType  # CASE, LAW, MANUAL, NOTICE
    source_id: str          # 원본 문서 식별자
    title: str
    content: str
    chunk_index: int = 0
    total_chunks: int = 1
    created_at: datetime
    updated_at: datetime
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def _check_chunk_bounds(self) -> "DocumentMetadataSchema":
        if self.chunk_index >= self.total_chunks:
            raise ValueError(
                f"chunk_index({self.chunk_index})는 total_chunks({self.total_chunks})보다 작아야 합니다"
            )
        return self


class SearchRequest(BaseModel):
    """확장 검색 요청 모델."""

    query: str = Field(..., min_length=1, max_length=2000, description="검색 쿼리 텍스트")
    doc_type: IndexType = Field(default=IndexType.CASE, description="검색 대상 문서 타입")
    top_k: int = Field(default=5, gt=0, le=50, description="반환할 최대 결과 수")


class SearchResponse(BaseModel):
    """확장 검색 응답 모델."""

    query: str
    doc_type: IndexType
    results: List["SearchResult"]
    total: int


class SearchResult(BaseModel):
    """확장된 검색 결과 모델.

    기존 RetrievedCase 대비 source_type, reliability_score, metadata 등
    풍부한 컨텍스트를 포함한다.
    """

    doc_id: str
    source_type: IndexType
    title: str
    content: str
    score: float
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0
    total_chunks: int = 1


# ---------------------------------------------------------------------------
# 변환 헬퍼
# ---------------------------------------------------------------------------


def from_internal_metadata(
    meta: "DocumentMetadata",
    content: str = "",
) -> DocumentMetadataSchema:
    """index_manager.DocumentMetadata (dataclass) -> DocumentMetadataSchema (Pydantic) 변환.

    Parameters
    ----------
    meta : DocumentMetadata
        내부 dataclass 메타데이터 인스턴스.
    content : str
        문서 본문 텍스트. dataclass에는 content 필드가 없으므로
        별도로 전달받는다.

    Returns
    -------
    DocumentMetadataSchema
        Pydantic 모델 인스턴스.
    """
    return DocumentMetadataSchema(
        doc_id=meta.doc_id,
        source_type=IndexType(meta.doc_type),
        source_id=meta.source,
        title=meta.title,
        content=content,
        chunk_index=meta.chunk_index,
        total_chunks=meta.chunk_total,
        created_at=datetime.fromisoformat(meta.created_at),
        updated_at=datetime.fromisoformat(meta.updated_at),
        valid_from=(
            datetime.fromisoformat(meta.valid_from)
            if meta.valid_from
            else None
        ),
        valid_until=(
            datetime.fromisoformat(meta.valid_until)
            if meta.valid_until
            else None
        ),
        reliability_score=meta.reliability_score,
        metadata=meta.extras if meta.extras else {},
    )


# forward-ref 해소
GenerateResponse.model_rebuild()
StreamResponse.model_rebuild()
SearchResponse.model_rebuild()
