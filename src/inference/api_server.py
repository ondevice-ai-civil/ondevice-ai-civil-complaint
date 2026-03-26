import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

from .agent_manager import AgentManager
from .bm25_indexer import BM25Indexer
from .hybrid_search import HybridSearchEngine, SearchMode
from .index_manager import IndexType, MultiIndexManager
from .retriever import CivilComplaintRetriever
from .schemas import (
    ClassificationResult,
    ClassifyRequest,
    ClassifyResponse,
    GenerateRequest,
    GenerateResponse,
    RetrievedCase,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StreamResponse,
)
from .feature_flags import FeatureFlags
from .vllm_stabilizer import apply_transformers_patch

# --- Rate Limiting (optional) ---
try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    _RATE_LIMIT_AVAILABLE = True
except ImportError:
    limiter = None
    _RATE_LIMIT_AVAILABLE = False

# --- API Key Authentication ---
_API_KEY = os.getenv("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    """API Key 인증. API_KEY 환경변수 미설정 시 인증을 건너뛴다."""
    if _API_KEY is None:
        return  # API_KEY 미설정 시 인증 건너뜀 (개발 환경 호환)
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


# --- M3 Optimized Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "umyunsang/GovOn-EXAONE-LoRA-v2")
DATA_PATH = os.getenv("DATA_PATH", "data/processed/v2_train.jsonl")
INDEX_PATH = os.getenv("INDEX_PATH", "models/faiss_index/complaints.index")

# Optimized for 16GB VRAM with AWQ INT4 model
GPU_UTILIZATION = float(os.getenv("GPU_UTILIZATION", "0.8"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
TRUST_REMOTE_CODE = True
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
AGENTS_DIR = os.getenv("AGENTS_DIR", os.path.join(_PROJECT_ROOT, "agents"))

# Apply EXAONE-specific runtime patches
apply_transformers_patch()


class vLLMEngineManager:
    """Manages the global AsyncLLMEngine and Retriever lifecycle for M3 Phase."""

    def __init__(self):
        self.engine: AsyncLLMEngine = None
        self.retriever: CivilComplaintRetriever = None
        self.index_manager = None
        self.agent_manager: AgentManager = None
        self.hybrid_engine: Optional[HybridSearchEngine] = None
        self.bm25_indexers: dict = {}
        self.embed_model = None
        self.feature_flags: FeatureFlags = FeatureFlags.from_env()
        self.pii_masker = None

    async def initialize(self):
        # 1. Initialize Optimized vLLM Engine
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            trust_remote_code=TRUST_REMOTE_CODE,
            gpu_memory_utilization=GPU_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            dtype="half",
            enforce_eager=True,  # More stable for patched EXAONE
        )
        logger.info(f"Initializing vLLM M3 engine with model: {MODEL_PATH}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 2. Initialize RAG Retriever
        logger.info(f"Initializing RAG Retriever with index: {INDEX_PATH}")
        self.retriever = CivilComplaintRetriever(
            index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
            data_path=DATA_PATH if not os.path.exists(INDEX_PATH) else None,
        )
        if self.retriever.index is not None and not os.path.exists(INDEX_PATH):
            self.retriever.save_index(INDEX_PATH)

        # 3. Initialize Agent Manager
        logger.info(f"Loading agent personas from: {AGENTS_DIR}")
        self.agent_manager = AgentManager(AGENTS_DIR)
        logger.info(f"Loaded agents: {self.agent_manager.list_agents()}")

        # 4. Initialize MultiIndexManager
        faiss_index_dir = os.getenv("FAISS_INDEX_DIR", "models/faiss_index")
        if os.path.isdir(faiss_index_dir):
            self.index_manager = MultiIndexManager(base_dir=faiss_index_dir)
            logger.info(f"MultiIndexManager 초기화 완료: {faiss_index_dir}")
        else:
            logger.warning(
                f"FAISS 인덱스 디렉토리 미존재: {faiss_index_dir} — MultiIndexManager 미초기화"
            )

        # 5. Initialize BM25 Indexers & HybridSearchEngine
        bm25_index_dir = os.getenv("BM25_INDEX_DIR", "models/bm25_index")
        if os.path.isdir(bm25_index_dir):
            if not os.getenv("BM25_INDEX_HMAC_KEY"):
                logger.warning(
                    "BM25_INDEX_HMAC_KEY 미설정: BM25 인덱스 무결성 검증이 비활성화되어 있습니다. "
                    "프로덕션 환경에서는 HMAC 키를 설정하세요."
                )
            for idx_type in IndexType:
                bm25_path = os.path.join(bm25_index_dir, f"{idx_type.value}.pkl")
                if os.path.exists(bm25_path):
                    try:
                        indexer = BM25Indexer()
                        indexer.load(bm25_path)
                        self.bm25_indexers[idx_type] = indexer
                        logger.info(
                            f"BM25 인덱스 로드 완료: {idx_type.value} ({indexer.doc_count}건)"
                        )
                    except Exception as e:
                        logger.warning(f"BM25 인덱스 로드 실패 ({idx_type.value}): {e}")

        # Embed model 추출 (retriever에서 공유)
        if self.retriever and hasattr(self.retriever, "model"):
            self.embed_model = self.retriever.model

        # HybridSearchEngine 초기화
        if self.index_manager and self.embed_model:
            self.hybrid_engine = HybridSearchEngine(
                index_manager=self.index_manager,
                bm25_indexers=self.bm25_indexers,
                embed_model=self.embed_model,
            )
            logger.info(
                f"HybridSearchEngine 초기화 완료 (BM25 인덱스: "
                f"{list(self.bm25_indexers.keys())})"
            )
        else:
            logger.warning("HybridSearchEngine 미초기화: index_manager 또는 embed_model 없음")

        # 6. Initialize PII Masker (검색 결과 개인정보 마스킹용)
        try:
            from src.data_collection_preprocessing.pii_masking import PIIMasker

            self.pii_masker = PIIMasker.create_strict_masker()
            logger.info("PIIMasker 초기화 완료 (검색 결과 PII 마스킹 활성)")
        except Exception as e:
            logger.warning(f"PIIMasker 초기화 실패 — 검색 결과 PII 마스킹이 비활성화됩니다: {e}")
            self.pii_masker = None

    def _escape_special_tokens(self, text: str) -> str:
        """Escape EXAONE chat template tokens to prevent prompt injection."""
        tokens = [
            "[|user|]",
            "[|assistant|]",
            "[|system|]",
            "[|endofturn|]",
            "<thought>",
            "</thought>",
        ]
        for token in tokens:
            text = text.replace(
                token,
                token.replace("[", "\\[")
                .replace("]", "\\]")
                .replace("<", "\\<")
                .replace(">", "\\>"),
            )
        return text

    @staticmethod
    def _strip_thought_blocks(text: str) -> str:
        """LLM 출력에서 내부 추론 블록(<thought>...</thought>)을 제거한다."""
        return re.sub(r"<thought>.*?</thought>\s*", "", text, flags=re.DOTALL).strip()

    def _build_rag_context(self, retrieved_cases: List[dict]) -> str:
        """RAG 참고 사례 컨텍스트 문자열을 생성한다."""
        if not retrieved_cases:
            return ""
        rag_context = "### 참고 사례 (유사 민원 및 답변):\n"
        for i, case in enumerate(retrieved_cases):
            safe_complaint = self._escape_special_tokens(case.get("complaint", ""))
            safe_answer = self._escape_special_tokens(case.get("answer", ""))
            rag_context += f"{i+1}. [민원]: {safe_complaint}\n   [답변]: {safe_answer}\n\n"
        return rag_context

    def _augment_prompt(self, prompt: str, retrieved_cases: List[dict]) -> str:
        """Augment the prompt with retrieved similar cases (RAG). (레거시 폴백용)"""
        if not retrieved_cases:
            return prompt

        rag_context = "\n\n" + self._build_rag_context(retrieved_cases)

        # Structure the prompt for EXAONE Chat Template
        if "[|user|]" in prompt:
            parts = prompt.split("[|user|]")
            return f"{parts[0]}[|user|]{rag_context}위 참고 사례를 바탕으로 다음 민원에 대해 답변해 주세요.\n\n{parts[1]}"
        return f"{rag_context}\n\n{prompt}"

    def _extract_query(self, prompt: str) -> str:
        """m-5: 정규식 기반 쿼리 추출."""
        user_match = re.search(r"\[\|user\|\](.*?)\[\|endofturn\|\]", prompt, re.DOTALL)
        if user_match:
            user_block = user_match.group(1)
            complaint_match = re.search(r"민원\s*내용\s*:\s*(.+)", user_block, re.DOTALL)
            if complaint_match:
                return complaint_match.group(1).strip()
            return user_block.strip()
        return prompt

    async def generate(
        self, request: GenerateRequest, request_id: str, flags: "FeatureFlags | None" = None
    ) -> tuple:
        # 1. RAG: Retrieve similar cases if enabled
        effective_flags = flags or self.feature_flags
        retrieved_cases = []

        if request.use_rag and effective_flags.use_rag_pipeline and self.retriever:
            query = self._escape_special_tokens(self._extract_query(request.prompt))
            retrieved_cases = self.retriever.search(query, top_k=3)

        # 2. Build prompt: generator 페르소나가 있으면 사용, 없으면 레거시 폴백
        if self.agent_manager and self.agent_manager.get_agent("generator"):
            safe_message = self._escape_special_tokens(request.prompt)
            if retrieved_cases:
                rag_context = self._build_rag_context(retrieved_cases)
                safe_message = (
                    f"{rag_context}"
                    f"위 참고 사례를 바탕으로 다음 민원에 대해 답변해 주세요.\n\n"
                    f"{safe_message}"
                )
            augmented_prompt = self.agent_manager.build_prompt("generator", safe_message)
        else:
            augmented_prompt = request.prompt
            if retrieved_cases:
                augmented_prompt = self._augment_prompt(request.prompt, retrieved_cases)

        # 3. vLLM Generation
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            repetition_penalty=1.1,  # Added for EXAONE stability
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
    lifespan=lifespan,
)

# --- m-6: CORS 미들웨어 ---
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
if ALLOWED_ORIGINS and ALLOWED_ORIGINS[0]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- C-2: Rate Limiting 미들웨어 ---
if _RATE_LIMIT_AVAILABLE and limiter is not None:
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)


# --- H-1: /health 정보 노출 최소화 ---
@app.get("/health")
async def health():
    index_summary = None
    if manager.index_manager:
        stats = manager.index_manager.get_index_stats()
        index_summary = {
            idx_type: {
                "loaded": info.get("loaded", False),
                "doc_count": info.get("doc_count", 0),
            }
            for idx_type, info in stats.get("indexes", {}).items()
        }
    # BM25 인덱스 상태
    bm25_summary = {}
    for idx_type in IndexType:
        indexer = manager.bm25_indexers.get(idx_type)
        if indexer and indexer.is_ready():
            bm25_summary[idx_type.value] = {"loaded": True, "doc_count": indexer.doc_count}
        else:
            bm25_summary[idx_type.value] = {"loaded": False}

    return {
        "status": "healthy",
        "rag_enabled": manager.index_manager is not None or manager.retriever is not None,
        "agents_loaded": manager.agent_manager.list_agents() if manager.agent_manager else [],
        "indexes": index_summary,
        "bm25_indexes": bm25_summary,
        "hybrid_search_enabled": manager.hybrid_engine is not None,
        "pii_masking_enabled": manager.pii_masker is not None,
        "feature_flags": {
            "use_rag_pipeline": manager.feature_flags.use_rag_pipeline,
            "model_version": manager.feature_flags.model_version,
        },
    }


# --- C-2: Rate limit decorator helper ---
def _rate_limit(limit_string: str):
    """slowapi가 사용 가능할 때만 rate limit 데코레이터를 반환한다."""
    if _RATE_LIMIT_AVAILABLE and limiter is not None:
        return limiter.limit(limit_string)

    # slowapi 미설치 시 아무 동작도 하지 않는 패스스루 데코레이터
    def _noop(func):
        return func

    return _noop


def get_feature_flags(request: Request) -> FeatureFlags:
    """요청 헤더에서 Feature Flag 오버라이드를 적용하여 반환한다."""
    header = request.headers.get("X-Feature-Flag")
    return manager.feature_flags.override_from_header(header)


@app.post("/v1/classify", response_model=ClassifyResponse)
@_rate_limit("60/minute")
async def classify(
    request: ClassifyRequest,
    _: None = Depends(verify_api_key),
    flags: FeatureFlags = Depends(get_feature_flags),
):
    """민원 분류 엔드포인트. classifier 에이전트 페르소나로 카테고리를 결정한다."""
    if not manager.agent_manager or not manager.agent_manager.get_agent("classifier"):
        raise HTTPException(status_code=503, detail="분류 에이전트가 로드되지 않았습니다.")

    classifier = manager.agent_manager.get_agent("classifier")
    safe_prompt = manager._escape_special_tokens(request.prompt)
    classify_prompt = manager.agent_manager.build_prompt("classifier", safe_prompt)

    request_id = str(uuid.uuid4())
    sampling_params = SamplingParams(
        temperature=classifier.temperature,
        top_p=0.9,
        max_tokens=classifier.max_tokens,
    )

    results_generator = manager.engine.generate(classify_prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise HTTPException(status_code=500, detail="분류 처리에 실패했습니다.")

    response_text = final_output.outputs[0].text

    classification = None
    classification_error = None
    try:
        json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if json_match:
            classification = ClassificationResult.model_validate_json(json_match.group())
        else:
            classification_error = "LLM 응답에서 JSON 객체를 찾을 수 없습니다."
            logger.warning(f"분류 JSON 파싱 실패 (request_id={request_id}): JSON 미발견")
    except Exception as e:
        classification_error = f"분류 결과 검증 실패: {e}"
        logger.warning(f"분류 JSON 파싱 실패 (request_id={request_id}): {e}")

    return ClassifyResponse(
        request_id=request_id,
        classification=classification,
        classification_error=classification_error,
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.outputs[0].token_ids),
    )


@app.post("/v1/generate", response_model=GenerateResponse)
@_rate_limit("30/minute")
async def generate(
    request: GenerateRequest,
    _: None = Depends(verify_api_key),
    flags: FeatureFlags = Depends(get_feature_flags),
):
    """Non-streaming text generation."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Use /v1/stream for streaming.")

    request_id = str(uuid.uuid4())
    results_generator, retrieved_cases = await manager.generate(request, request_id, flags)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise HTTPException(status_code=500, detail="Generation failed.")

    return GenerateResponse(
        request_id=request_id,
        text=manager._strip_thought_blocks(final_output.outputs[0].text),
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.outputs[0].token_ids),
        retrieved_cases=[RetrievedCase(**c) for c in retrieved_cases],
    )


@app.post("/v1/stream")
@_rate_limit("30/minute")
async def stream_generate(
    request: GenerateRequest,
    _: None = Depends(verify_api_key),
    flags: FeatureFlags = Depends(get_feature_flags),
):
    """Streaming text generation using SSE."""
    if not request.stream:
        request.stream = True

    request_id = str(uuid.uuid4())
    results_generator, retrieved_cases = await manager.generate(request, request_id, flags)

    async def stream_results() -> AsyncGenerator[str, None]:
        cases_data = [RetrievedCase(**c).model_dump() for c in retrieved_cases]

        async for request_output in results_generator:
            text = request_output.outputs[0].text
            finished = request_output.finished
            if finished:
                text = manager._strip_thought_blocks(text)

            response_obj = {"request_id": request_id, "text": text, "finished": finished}
            if finished:
                response_obj["retrieved_cases"] = cases_data

            yield f"data: {json.dumps(response_obj)}\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")


def _extract_content_by_type(result: dict, index_type: IndexType) -> str:
    """인덱스 타입별로 적절한 content 텍스트를 추출한다.

    Parameters
    ----------
    result : dict
        검색 결과 딕셔너리 (extras 포함).
    index_type : IndexType
        검색 대상 문서 타입.

    Returns
    -------
    str
        추출된 content 텍스트. 비어있으면 title을 폴백으로 반환한다.
    """
    extras = result.get("extras", {})
    if index_type == IndexType.CASE:
        text = (extras.get("complaint_text", "") + "\n" + extras.get("answer_text", "")).strip()
    elif index_type == IndexType.LAW:
        text = extras.get("law_text", "") or extras.get("content", "")
    elif index_type == IndexType.MANUAL:
        text = extras.get("manual_text", "") or extras.get("content", "")
    elif index_type == IndexType.NOTICE:
        text = extras.get("notice_text", "") or extras.get("content", "")
    else:
        text = ""
    return text or result.get("title", "")


def _mask_search_results(
    results: List[SearchResult], masker: Optional[object]
) -> List[SearchResult]:
    """검색 결과 내 PII(개인식별정보)를 마스킹한다."""
    if masker is None:
        return results
    for r in results:
        r.content = masker.mask_all(r.content)
        # metadata 내 텍스트 필드도 마스킹
        for key in ("complaint_text", "answer_text", "complaint", "answer"):
            if key in r.metadata and isinstance(r.metadata[key], str):
                r.metadata[key] = masker.mask_all(r.metadata[key])
    return results


# --- S-1: /search 엔드포인트 ---
# NOTE: slowapi는 경로별 별도 버킷으로 rate limit을 추적합니다.
# /v1/search와 /search 각각 60/minute으로 제한됩니다.
@app.post("/v1/search", response_model=SearchResponse)
@app.post("/search", response_model=SearchResponse)
@_rate_limit("60/minute")
async def search(request: SearchRequest, req: Request, _: None = Depends(verify_api_key)):
    """확장 검색 엔드포인트. search_mode로 검색 방식을 선택한다."""
    start_time = time.monotonic()
    try:
        # HybridSearchEngine 사용
        if manager.hybrid_engine:
            results_raw, actual_mode = await manager.hybrid_engine.search(
                query=request.query,
                index_type=request.doc_type,
                top_k=request.top_k,
                mode=request.search_mode,
            )
            results = [
                SearchResult(
                    doc_id=r.get("doc_id", ""),
                    source_type=IndexType(r.get("doc_type", request.doc_type.value)),
                    title=r.get("title", ""),
                    content=_extract_content_by_type(r, request.doc_type),
                    score=r.get("score", 0.0),
                    reliability_score=r.get("reliability_score", 1.0),
                    metadata=r.get("extras", {}),
                    chunk_index=r.get("chunk_index", 0),
                    total_chunks=r.get("chunk_total", 1),
                )
                for r in results_raw
            ]
        # 레거시 폴백: retriever 사용
        elif manager.retriever:
            raw_results = manager.retriever.search(request.query, top_k=request.top_k)
            results = [
                SearchResult(
                    doc_id=raw.get("id", raw.get("doc_id", "")),
                    source_type=request.doc_type,
                    title=raw.get("category", ""),
                    content=raw.get("complaint", "") + "\n" + raw.get("answer", ""),
                    score=raw.get("score", 0.0),
                    reliability_score=raw.get("reliability_score", 1.0),
                )
                for raw in raw_results
            ]
            actual_mode = SearchMode.DENSE
        else:
            raise HTTPException(
                status_code=503,
                detail="검색 엔진이 아직 초기화되지 않았습니다.",
            )

        # PII 마스킹: 검색 결과 내 개인정보를 마스킹하여 반환
        results = _mask_search_results(results, manager.pii_masker)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        # 폴백이 발생한 경우에만 actual_search_mode를 설정
        actual_search_mode = actual_mode if actual_mode != request.search_mode else None
        return SearchResponse(
            query=request.query,
            doc_type=request.doc_type,
            search_mode=actual_mode,
            actual_search_mode=actual_search_mode,
            results=results,
            total=len(results),
            search_time_ms=round(elapsed_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        # C-3: 내부 예외 정보 노출 방지
        logger.error(f"검색 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="검색 처리 중 내부 오류가 발생했습니다.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
