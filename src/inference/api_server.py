import json
import uuid
import os
from typing import AsyncGenerator, List, Tuple
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from contextlib import asynccontextmanager

from .vllm_stabilizer import apply_transformers_patch
from .schemas import GenerateRequest, GenerateResponse, StreamResponse, RetrievedCase
from .retriever import CivilComplaintRetriever
from .agent_manager import AgentManager

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
        try:
            # 1. Initialize Optimized vLLM Engine
            engine_args = AsyncEngineArgs(
                model=MODEL_PATH,
                trust_remote_code=TRUST_REMOTE_CODE,
                gpu_memory_utilization=GPU_UTILIZATION,
                max_model_len=MAX_MODEL_LEN,
                dtype="half",
                enforce_eager=True
            )
            print(f"Initializing vLLM M3 engine with model: {MODEL_PATH}")
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # 2. Initialize RAG Retriever
            print(f"Initializing RAG Retriever with index: {INDEX_PATH}")
            self.retriever = CivilComplaintRetriever(
                index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
                data_path=DATA_PATH if not os.path.exists(INDEX_PATH) else None
            )
            if self.retriever.index is not None and not os.path.exists(INDEX_PATH):
                self.retriever.save_index(INDEX_PATH)
                
            # 3. Initialize AgentManager (Multi-Agent Persona)
            self.agent_manager = AgentManager()
            if not self.agent_manager.personas:
                raise RuntimeError("No agent personas loaded. Check agents directory.")
                
        except Exception as e:
            print(f"Critical failure during vLLM Manager initialization: {e}")
            raise

    def _escape_special_tokens(self, text: str) -> str:
        """Escape EXAONE chat template tokens to prevent prompt injection."""
        # More comprehensive escape for M3 stability
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
            safe_complaint = self._escape_special_tokens(case['complaint'])
            safe_answer = self._escape_special_tokens(case['answer'])
            rag_context += f"{i+1}. [민원]: {safe_complaint}\n   [답변]: {safe_answer}\n\n"
        
        # Structure the prompt for EXAONE Chat Template
        if "[|user|]" in prompt:
            parts = prompt.split("[|user|]", 1)
            return f"{parts[0]}[|user|]{rag_context}위 참고 사례를 바탕으로 다음 민원에 대해 답변해 주세요.\n\n{parts[1]}"
        return f"{rag_context}\n\n{prompt}"

    async def generate(self, request: GenerateRequest, request_id: str, agent_type: str = "generator") -> Tuple[AsyncGenerator, List[dict]]:
        # 1. RAG: Retrieve similar cases if enabled
        retrieved_cases = []
        augmented_prompt = request.prompt
        
        if request.use_rag and self.retriever:
            # Extract actual complaint for search
            query = request.prompt
            if "민원 내용:" in query:
                query = query.split("민원 내용:")[1].split("[|endofturn|]")[0].strip()
            elif "[|user|]" in query:
                query = query.split("[|user|]")[1].split("[|endofturn|]")[0].strip()
                
            retrieved_cases = self.retriever.search(query, top_k=3)
            augmented_prompt = self._augment_prompt(request.prompt, retrieved_cases)

        # 2. Agent Personalization
        persona = self.agent_manager.get_persona(agent_type)
        if not persona:
            print(f"Warning: Persona '{agent_type}' not found. Using default.")
            persona = "You are a helpful assistant."

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
    title="GovOn AI Integrated Backend (M3)",
    description="Unified API for Text Generation, RAG, and Classification using Multi-Agent Persona.",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_PATH, "agents": list(manager.agent_manager.personas.keys())}

@app.post("/v1/classify")
async def classify(request: GenerateRequest):
    """Classify civil complaint into categories using the Classifier Agent."""
    request_id = str(uuid.uuid4())
    # Optimization: Classification usually needs fewer tokens and no RAG
    request.use_rag = False
    request.max_tokens = 32
    request.temperature = 0.0 # Deterministic
    
    results_generator, _ = await manager.generate(request, request_id, agent_type="classifier")
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    
    if final_output is None:
        raise HTTPException(status_code=500, detail="Classification failed.")

    return {"request_id": request_id, "category": final_output.outputs[0].text.strip()}

@app.post("/v1/search")
async def search(query: str, top_k: int = 3):
    """Directly search for similar cases using the Retriever Agent logic."""
    if not manager.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized.")
    
    results = manager.retriever.search(query, top_k=top_k)
    return {"results": [RetrievedCase(**c).model_dump() for c in results]}

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Non-streaming text generation using the Generator Agent."""
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
async def stream_generate(request: GenerateRequest):
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
            
            response_obj = {"request_id": request_id, "text": text, "finished": finished}
            if finished:
                response_obj["retrieved_cases"] = cases_data
                
            yield f"data: {json.dumps(response_obj)}\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
