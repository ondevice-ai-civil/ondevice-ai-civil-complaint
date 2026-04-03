from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.inference.hybrid_search import SearchMode
from src.inference.index_manager import DocumentMetadata, IndexType


class RetrievedCase(BaseModel):
    id: Optional[str] = None
    category: Optional[str] = None
    complaint: str
    answer: str
    score: float


class BaseGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(default=512, gt=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = Field(default=None)
    use_rag: bool = Field(default=True)


class BaseGenerateResponse(BaseModel):
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    search_results: Optional[List["SearchResult"]] = None


class GenerateCivilResponseRequest(BaseGenerateRequest):
    complaint_id: Optional[str] = None


class GenerateCivilResponseResponse(BaseGenerateResponse):
    complaint_id: Optional[str] = None
    retrieved_cases: Optional[List[RetrievedCase]] = None


class GenerateRequest(GenerateCivilResponseRequest):
    """레거시 /v1/generate 호환 요청 모델."""


class GenerateResponse(GenerateCivilResponseResponse):
    """레거시 /v1/generate 호환 응답 모델."""


class StreamResponse(BaseModel):
    request_id: str
    text: str
    finished: bool = False
    retrieved_cases: Optional[List[RetrievedCase]] = None
    search_results: Optional[List["SearchResult"]] = None


class DocumentMetadataSchema(BaseModel):
    doc_id: str
    source_type: IndexType
    source_id: str
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
    query: str = Field(..., min_length=1, max_length=2000)
    doc_type: IndexType = Field(default=IndexType.CASE)
    top_k: int = Field(default=5, gt=0, le=50)
    search_mode: SearchMode = Field(default=SearchMode.HYBRID)


class SearchResponse(BaseModel):
    query: str
    doc_type: IndexType
    search_mode: SearchMode = SearchMode.HYBRID
    actual_search_mode: Optional[SearchMode] = None
    results: List["SearchResult"]
    total: int
    search_time_ms: Optional[float] = None


class SearchResult(BaseModel):
    doc_id: str
    source_type: IndexType
    title: str
    content: str
    score: float
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0
    total_chunks: int = 1


def from_internal_metadata(
    meta: "DocumentMetadata",
    content: str = "",
) -> DocumentMetadataSchema:
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
        valid_from=(datetime.fromisoformat(meta.valid_from) if meta.valid_from else None),
        valid_until=(datetime.fromisoformat(meta.valid_until) if meta.valid_until else None),
        reliability_score=meta.reliability_score,
        metadata=meta.extras if meta.extras else {},
    )


class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    stream: bool = Field(default=False)
    force_tools: Optional[List[str]] = None
    max_tokens: int = Field(default=512, gt=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    use_rag: bool = Field(default=True)


class ToolResultSchema(BaseModel):
    tool: str
    success: bool
    latency_ms: float
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentTraceSchema(BaseModel):
    request_id: str
    session_id: str
    plan: List[str] = Field(default_factory=list)
    plan_reason: str = ""
    tool_results: List[ToolResultSchema] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    error: Optional[str] = None


class AgentRunResponse(BaseModel):
    request_id: str
    session_id: str
    text: str
    trace: AgentTraceSchema
    search_results: Optional[List[Dict[str, Any]]] = None


GenerateCivilResponseResponse.model_rebuild()
GenerateResponse.model_rebuild()
StreamResponse.model_rebuild()
SearchResponse.model_rebuild()
