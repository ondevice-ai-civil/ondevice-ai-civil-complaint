from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class BaseGenerateResponse(BaseModel):
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int


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


class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    stream: bool = Field(default=False)
    force_tools: Optional[List[str]] = None
    max_tokens: int = Field(default=512, gt=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


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
