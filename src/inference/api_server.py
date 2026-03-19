import json
import uuid
import os
from typing import AsyncGenerator, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from contextlib import asynccontextmanager

from .vllm_stabilizer import apply_transformers_patch
from .schemas import GenerateRequest, GenerateResponse, StreamResponse, RetrievedCase
from .retriever import CivilComplaintRetriever

# --- M3 Optimized Configuration ---
MODEL_PATH = "umyunsang/civil-complaint-exaone-awq"
DATA_PATH = "GovOn/data/processed/v2_train.jsonl"
INDEX_PATH = "GovOn/models/faiss_index/complaints.index"

# Optimized for 16GB VRAM with AWQ INT4 model
GPU_UTILIZATION = 0.8  # Increased for AWQ efficiency
MAX_MODEL_LEN = 8192    # Optimized for EXAONE 32K but capped for stability
TRUST_REMOTE_CODE = True

# Apply EXAONE-specific runtime patches
apply_transformers_patch()

class vLLMEngineManager:
    """Manages the global AsyncLLMEngine and Retriever lifecycle for M3 Phase."""
    def __init__(self):
        self.engine: AsyncLLMEngine = None
        self.retriever: CivilComplaintRetriever = None

    async def initialize(self):
        # 1. Initialize Optimized vLLM Engine
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            trust_remote_code=TRUST_REMOTE_CODE,
            gpu_memory_utilization=GPU_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            dtype="half",
            enforce_eager=True # More stable for patched EXAONE
        )
        print(f"Initializing vLLM M3 engine with model: {MODEL_PATH}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 2. Initialize RAG Retriever
        print(f"Initializing RAG Retriever with index: {INDEX_PATH}")
        self.retriever = CivilComplaintRetriever(
            index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
            data_path=DATA_PATH if not os.path.exists(INDEX_PATH) else None
        )
        if not os.path.exists(INDEX_PATH) and self.retriever.index is not None:
            self.retriever.save_index(INDEX_PATH)

    def _augment_prompt(self, prompt: str, retrieved_cases: List[dict]) -> str:
        """Augment the prompt with retrieved similar cases (RAG)."""
        if not retrieved_cases:
            return prompt
            
        rag_context = "\n\n### 참고 사례 (유사 민원 및 답변):\n"
        for i, case in enumerate(retrieved_cases):
            rag_context += f"{i+1}. [민원]: {case['complaint']}\n   [답변]: {case['answer']}\n\n"
        
        # Structure the prompt for EXAONE Chat Template
        if "[|user|]" in prompt:
            parts = prompt.split("[|user|]")
            return f"{parts[0]}[|user|]{rag_context}위 참고 사례를 바탕으로 다음 민원에 대해 답변해 주세요.\n\n{parts[1]}"
        return f"{rag_context}\n\n{prompt}"

    async def generate(self, request: GenerateRequest, request_id: str):
        # 1. RAG: Retrieve similar cases if enabled
        retrieved_cases = []
        augmented_prompt = request.prompt
        
        if request.use_rag and self.retriever:
            # Simple query extraction
            query = request.prompt
            if "민원 내용:" in query:
                query = query.split("민원 내용:")[1].split("[|endofturn|]")[0].strip()
            elif "[|user|]" in query:
                query = query.split("[|user|]")[1].split("[|endofturn|]")[0].strip()
                
            retrieved_cases = self.retriever.search(query, top_k=3)
            augmented_prompt = self._augment_prompt(request.prompt, retrieved_cases)

        # 2. vLLM Generation
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            repetition_penalty=1.1 # Added for EXAONE stability
        )
        
        return self.engine.generate(augmented_prompt, sampling_params, request_id), retrieved_cases

manager = vLLMEngineManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    await manager.initialize()
    yield

app = FastAPI(
    title="GovOn AI Serving API (M3 Optimized)",
    description="High-performance FastAPI + vLLM with RAG support for GovOn project.",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_PATH, "rag_enabled": manager.retriever is not None}

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Non-streaming text generation."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Use /v1/stream for streaming.")
    
    request_id = str(uuid.uuid4())
    results_generator, retrieved_cases = await manager.generate(request, request_id)
    
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
async def stream_generate(request: GenerateRequest):
    """Streaming text generation using SSE."""
    if not request.stream:
        request.stream = True
    
    request_id = str(uuid.uuid4())
    results_generator, retrieved_cases = await manager.generate(request, request_id)
    
    async def stream_results() -> AsyncGenerator[str, None]:
        cases_data = [RetrievedCase(**c).dict() for c in retrieved_cases]
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
