import json
import uuid
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from contextlib import asynccontextmanager

from .schemas import GenerateRequest, GenerateResponse, StreamResponse

# --- Configuration (Based on Serving Specialist Rules) ---
MODEL_PATH = "LG-EXAONE/EXAONE-3.0-7.8B-Instruct" # Change as needed
GPU_UTILIZATION = 0.8
MAX_MODEL_LEN = 4096

class vLLMEngineManager:
    """Manages the global AsyncLLMEngine lifecycle."""
    def __init__(self):
        self.engine: AsyncLLMEngine = None

    async def initialize(self):
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            dtype="half"
        )
        print(f"Initializing vLLM engine with model: {MODEL_PATH}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, request: GenerateRequest, request_id: str) -> AsyncGenerator:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop
        )
        return self.engine.generate(request.prompt, sampling_params, request_id)

manager = vLLMEngineManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for vLLM engine."""
    await manager.initialize()
    yield
    # Cleanup logic can be added here if needed

app = FastAPI(
    title="GovOn AI Serving API",
    description="High-performance FastAPI + vLLM serving for GovOn project.",
    lifespan=lifespan
)

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text in a single request (Non-streaming)."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Use streaming endpoint or set stream=False.")
    
    request_id = str(uuid.uuid4())
    results_generator = await manager.generate(request, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    if final_output is None:
        raise HTTPException(status_code=500, detail="Generation failed.")

    return GenerateResponse(
        request_id=request_id,
        text=final_output.outputs[0].text,
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.outputs[0].token_ids)
    )

@app.post("/v1/stream")
async def stream_generate(request: GenerateRequest):
    """Stream text generation results using SSE."""
    if not request.stream:
        request.stream = True # Force stream for this endpoint
    
    request_id = str(uuid.uuid4())
    
    async def stream_results() -> AsyncGenerator[str, None]:
        results_generator = await manager.generate(request, request_id)
        async for request_output in results_generator:
            text = request_output.outputs[0].text
            finished = request_output.finished
            yield f"data: {json.dumps({'request_id': request_id, 'text': text, 'finished': finished})}\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
