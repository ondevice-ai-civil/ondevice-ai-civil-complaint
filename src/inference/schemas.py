from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for generation.")
    max_tokens: int = Field(default=512, gt=0, description="Maximum number of tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter.")
    stream: bool = Field(default=False, description="Whether to stream the output using SSE.")
    stop: Optional[List[str]] = Field(default=None, description="List of stop sequences.")

class GenerateResponse(BaseModel):
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int

class StreamResponse(BaseModel):
    request_id: str
    text: str
    finished: bool = False
