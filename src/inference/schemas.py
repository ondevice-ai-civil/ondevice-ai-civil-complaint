from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.inference.runtime_config import govon_config

# Expose validation limits from unified config for Field constraints.
_MAX_PROMPT_LEN = govon_config.validation.max_prompt_length
_MAX_TOKENS_CEILING = govon_config.validation.max_tokens_ceiling
_DEFAULT_MAX_TOKENS = govon_config.generation.max_tokens
_DEFAULT_TEMPERATURE = govon_config.generation.temperature
_DEFAULT_TOP_P = govon_config.generation.top_p
_DEFAULT_MAX_ITERATIONS = govon_config.context.max_iterations


class RetrievedCase(BaseModel):
    id: Optional[str] = None
    category: Optional[str] = None
    complaint: str
    answer: str
    score: float


class BaseGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=_MAX_PROMPT_LEN)
    max_tokens: int = Field(default=_DEFAULT_MAX_TOKENS, gt=0, le=_MAX_TOKENS_CEILING)
    temperature: float = Field(default=_DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=_DEFAULT_TOP_P, ge=0.0, le=1.0)
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


class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX_PROMPT_LEN)
    session_id: Optional[str] = None
    stream: bool = Field(default=False)
    force_tools: Optional[List[str]] = None
    max_tokens: int = Field(default=_DEFAULT_MAX_TOKENS, gt=0, le=_MAX_TOKENS_CEILING)
    temperature: float = Field(default=_DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_iterations: int = Field(default=_DEFAULT_MAX_ITERATIONS, ge=1, le=20)


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
