import json
import os
import re
import uuid
from typing import AsyncGenerator, List, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from contextlib import asynccontextmanager

from .vllm_stabilizer import apply_transformers_patch
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StreamResponse,
    RetrievedCase,
)
from .retriever import CivilComplaintRetriever
from .agent_manager import AgentManager

# --- Rate Limiting (optional) ---
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    limiter = Limiter(key_func=get_remote_address)
    _RATE_LIMIT_AVAILABLE = True
except ImportError:
    limiter = None
    _RATE_LIMIT_AVAILABLE = False

# --- API Key Authentication ---
_API_KEY = os.getenv("API_KEY") or os.getenv("GOVON_API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    """API Key 인증. API_KEY 환경변수 미설정 시 인증을 건너뜁니다 (개발 편의)."""
    if _API_KEY is None:
        return
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


# --- M3 Optimized Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "umyunsang/GovOn-EXAONE-LoRA-v2")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/v2_train.jsonl")
INDEX_PATH = os.getenv("INDEX_PATH", "models/faiss_index/complaints.index")

GPU_UTILIZATION = float(os.getenv("GPU_UTILIZATION", "0.8"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
TRUST_REMOTE_CODE = True

# Apply EXAONE-specific runtime patches
apply_transformers_patch()


class vLLMEngineManager:
    """Manages the global AsyncLLMEngine, Retriever, and AgentManager for M3 Phase."""
    def __init__(self):
        self.engine: AsyncLLMEngine = None
        self.retriever: CivilComplaintRetriever = None
        self.agent_manager: AgentManager = None

    async def initialize(self):
        # 1. Initialize Optimized vLLM Engine
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            trust_remote_code=TRUST_REMOTE_CODE,
            gpu_memory_utilization=GPU_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            dtype="half",
            enforce_eager=True
        )
        logger.info(f"Initializing vLLM M3 engine with model: {MODEL_PATH}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 2. Initialize RAG Retriever
        logger.info(f"Initializing RAG Retriever with index: {INDEX_PATH}")
        self.retriever = CivilComplaintRetriever(
            index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
            data_path=DATA_PATH if not os.path.exists(INDEX_PATH) else None
        )
        if self.retriever.index is not None and not os.path.exists(INDEX_PATH):
            self.retriever.save_index(INDEX_PATH)

        # 3. Initialize AgentManager (Multi-Agent Persona)
        self.agent_manager = AgentManager()
        if not self.agent_manager.personas:
            logger.warning("No agent personas loaded. Agent personalization might be limited.")

    def _escape_special_tokens(self, text: str) -> str:
        """Escape EXAONE chat template tokens to prevent prompt injection."""
        tokens = ["[|user|]", "[|assistant|]", "[|system|]", "[|endofturn|]", "<thought>", "</thought>"]
        for token in tokens:
            text = text.replace(token, token.replace("[", "\\[").replace("]", "\\]").replace("<", "\\<").replace(">", "\\>"))
        return text

    def _augment_prompt(self, prompt: str, retrieved_cases: List[dict]) -> str:
        """Augment the prompt with retrieved similar cases (RAG)."""
        if not retrieved_cases:
            return prompt

        rag_context = "\n\n### 참고 사례 (유사 민원 및 답변):\n"
        for i, case in enumerate(retrieved_cases):
            safe_complaint = self._escape_special_tokens(case.get('complaint', ''))
            safe_answer = self._escape_special_tokens(case.get('answer', ''))
            rag_context += f"{i+1}. [민원]: {safe_complaint}\n   [답변]: {safe_answer}\n\n"

        if "[|user|]" in prompt:
            parts = prompt.split("[|user|]", 1)
            return f"{parts[0]}[|user|]{rag_context}위 참고 사례를 바탕으로 다음 민원에 대해 답변해 주세요.\n\n{parts[1]}"
        return f"{rag_context}\n\n{prompt}"

    def _extract_query(self, prompt: str) -> str:
        """Regex-based query extraction."""
        user_match = re.search(r"\[\|user\|\](.*?)\[\|endofturn\|\]", prompt, re.DOTALL)
        if user_match:
            user_block = user_match.group(1)
            complaint_match = re.search(r"민원\s*내용\s*:\s*(.+)", user_block, re.DOTALL)
            if complaint_match:
                return complaint_match.group(1).strip()
            return user_block.strip()
        return prompt

    async def generate(self, request: GenerateRequest, request_id: str, agent_type: str = "generator") -> Tuple[AsyncGenerator, List[dict]]:
        # 1. RAG: Retrieve similar cases if enabled
        retrieved_cases = []
        # Pre-escape user input
        safe_prompt = self._escape_special_tokens(request.prompt)
        augmented_prompt = safe_prompt

        if request.use_rag and self.retriever:
            query = self._extract_query(safe_prompt)
            retrieved_cases = self.retriever.search(query, top_k=3)
            augmented_prompt = self._augment_prompt(safe_prompt, retrieved_cases)

        # 2. Agent Personalization
        persona = self.agent_manager.get_persona(agent_type)
        if not persona:
            logger.warning(f"Persona '{agent_type}' not found. Using default.")
            persona = "당신은 지자체 민원 공무원을 돕는 친절한 AI 어시스턴트입니다."

        # Final EXAONE Prompt Assembly
        final_prompt = f"[|system|]{persona}[|endofturn|][|user|]{augmented_prompt}[|assistant|]"

        # 3. vLLM Generation
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            repetition_penalty=1.1
        )

        return self.engine.generate(final_prompt, sampling_params, request_id), retrieved_cases


manager = vLLMEngineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    await manager.initialize()
    yield

app = FastAPI(
    title="GovOn AI Serving API (M3 Optimized)",
    description="High-performance FastAPI + vLLM with RAG and Multi-Agent Persona.",
    lifespan=lifespan
)

# CORS Middleware
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
if _RATE_LIMIT_AVAILABLE and limiter is not None:
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

def _rate_limit(limit_string: str):
    if _RATE_LIMIT_AVAILABLE and limiter is not None:
        return limiter.limit(limit_string)
    def _noop(func):
        return func
    return _noop


@app.get("/health")
async def health():
    """Information disclosure minimized health check."""
    return {
        "status": "healthy",
        "agents_loaded": len(manager.agent_manager.personas) if manager.agent_manager else 0,
        "rag_status": "ready" if manager.retriever and manager.retriever.index else "disabled"
    }


@app.post("/v1/classify")
@_rate_limit("60/minute")
async def classify(request: GenerateRequest, _: None = Depends(verify_api_key)):
    """Classify complaint using the Classifier Agent."""
    request_id = str(uuid.uuid4())
    # Deterministic classification
    request.use_rag = False
    request.max_tokens = 32
    request.temperature = 0.0
    
    results_generator, _ = await manager.generate(request, request_id, agent_type="classifier")
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    if final_output is None:
        raise HTTPException(status_code=500, detail="Classification failed.")

    return {"request_id": request_id, "category": final_output.outputs[0].text.strip()}


@app.post("/v1/generate", response_model=GenerateResponse)
@_rate_limit("30/minute")
async def generate(request: GenerateRequest, _: None = Depends(verify_api_key)):
    """Non-streaming text generation using Generator Agent."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Use /v1/stream for streaming.")

    request_id = str(uuid.uuid4())
    results_generator, retrieved_cases = await manager.generate(request, request_id, agent_type="generator")

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise HTTPException(status_code=500, detail="Generation failed.")

    return GenerateResponse(
        request_id=request_id,
        text=final_output.outputs[0].text,
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.outputs[0].token_ids),
        retrieved_cases=[RetrievedCase(**c) for c in retrieved_cases]
    )


@app.post("/v1/stream")
@_rate_limit("30/minute")
async def stream_generate(request: GenerateRequest, _: None = Depends(verify_api_key)):
    """Streaming text generation using SSE."""
    if not request.stream:
        request.stream = True

    request_id = str(uuid.uuid4())
    results_generator, retrieved_cases = await manager.generate(request, request_id, agent_type="generator")

    async def stream_results() -> AsyncGenerator[str, None]:
        cases_data = [RetrievedCase(**c).model_dump() for c in retrieved_cases]

        async for request_output in results_generator:
            text = request_output.outputs[0].text
            finished = request_output.finished

            response_obj = {
                "request_id": request_id,
                "text": text,
                "finished": finished
            }
            if finished:
                response_obj["retrieved_cases"] = cases_data

            yield f"data: {json.dumps(response_obj)}\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")


@app.post("/search", response_model=SearchResponse)
@_rate_limit("60/minute")
async def search(request: SearchRequest, _: None = Depends(verify_api_key)):
    """Enhanced search endpoint."""
    if not manager.retriever:
        raise HTTPException(status_code=503, detail="Search index not initialized.")
    
    # Escape user input
    safe_query = manager._escape_special_tokens(request.query)
    raw_results = manager.retriever.search(safe_query, top_k=request.top_k)
    
    results = [
        SearchResult(
            doc_id=raw.get("doc_id", ""),
            source_type=request.doc_type,
            title=raw.get("category", ""),
            content=raw.get("complaint", "") + "\n" + raw.get("answer", ""),
            score=raw.get("score", 0.0),
            reliability_score=raw.get("reliability_score", 0.6),
        )
        for raw in raw_results
    ]

    return SearchResponse(
        query=request.query,
        doc_type=request.doc_type,
        results=results,
        total=len(results),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
