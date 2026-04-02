import asyncio
import html
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

try:
    from vllm import AsyncLLM, SamplingParams
except ImportError:
    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLM
        from vllm.sampling_params import SamplingParams
    except ImportError:
        AsyncLLM = None
        SamplingParams = None

# SKIP_MODEL_LOAD: E2E 테스트 등 모델 없이 서버만 기동할 때 사용
# RuntimeConfig 로드 전 조기 참조용 (import 순서 의존성)
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "false").lower() in ("true", "1", "yes")

from .agent_loop import AgentLoop, AgentTrace, ToolResult
from .agent_manager import AgentManager
from .bm25_indexer import BM25Indexer
from .feature_flags import FeatureFlags
from .hybrid_search import HybridSearchEngine, SearchMode
from .index_manager import IndexType, MultiIndexManager
from .retriever import CivilComplaintRetriever
from .runtime_config import RuntimeConfig, ServingProfile
from .schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentTraceSchema,
    ClassificationResult,
    ClassifyRequest,
    ClassifyResponse,
    GenerateCivilResponseRequest,
    GenerateCivilResponseResponse,
    GeneratePublicDocRequest,
    GeneratePublicDocResponse,
    GenerateRequest,
    GenerateResponse,
    RetrievedCase,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StreamResponse,
    ToolResultSchema,
)
from .session_context import SessionStore
from .tool_router import ToolType, tool_name

if not SKIP_MODEL_LOAD:
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams

        from .vllm_stabilizer import apply_transformers_patch
    except ImportError:
        logger.warning("vllm modules not found. Model loading will fail if attempted.")
        AsyncEngineArgs = object
        AsyncLLMEngine = object
        apply_transformers_patch = lambda: None

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


# --- Runtime Configuration ---
runtime_config = RuntimeConfig.from_env()
runtime_config.log_summary()

# 하위 호환을 위해 기존 모듈 레벨 변수 유지 (RuntimeConfig에서 참조)
MODEL_PATH = runtime_config.model.model_path
DATA_PATH = runtime_config.paths.data_path
INDEX_PATH = runtime_config.paths.index_path
GPU_UTILIZATION = runtime_config.gpu_utilization
MAX_MODEL_LEN = runtime_config.max_model_len
TRUST_REMOTE_CODE = runtime_config.model.trust_remote_code
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
AGENTS_DIR = runtime_config.paths.agents_dir


@dataclass
class PreparedGeneration:
    """생성 실행 전 프롬프트, 검색 컨텍스트, 샘플링 설정을 묶는다."""

    prompt: str
    sampling_params: SamplingParams
    retrieved_cases: List[dict]
    search_results: List[SearchResult]

# Apply EXAONE-specific runtime patches (모델 로드 시에만)
if not SKIP_MODEL_LOAD:
    apply_transformers_patch()


class vLLMEngineManager:
    """Manages the global AsyncLLMEngine and Retriever lifecycle for M3 Phase."""

    def __init__(self):
        self.engine: AsyncLLM = None
        self.retriever: CivilComplaintRetriever = None
        self.index_manager = None
        self.agent_manager: AgentManager = None
        self.hybrid_engine: Optional[HybridSearchEngine] = None
        self.bm25_indexers: dict = {}
        self.embed_model = None
        self.feature_flags: FeatureFlags = FeatureFlags.from_env()
        self.pii_masker = None
        self.session_store: SessionStore = SessionStore()
        self.agent_loop: Optional[AgentLoop] = None
        self.langgraph_runtime: Optional[Any] = None

    async def initialize(self):
        if SKIP_MODEL_LOAD:
            logger.info("SKIP_MODEL_LOAD=true: 모델 및 인덱스 로딩을 건너뜁니다 (E2E 테스트 모드)")
            return

        # 1. Initialize Optimized vLLM Engine
        logger.info(f"Initializing vLLM M3 engine with model: {MODEL_PATH}")

        # vllm.v1.engine.async_llm.AsyncLLM 또는 vllm.engine.async_llm_engine.AsyncLLMEngine 모두
        # from_engine_args를 지원하므로 이를 통해 초기화한다.
        try:
            engine_args = AsyncEngineArgs(
                model=MODEL_PATH,
                trust_remote_code=TRUST_REMOTE_CODE,
                gpu_memory_utilization=GPU_UTILIZATION,
                max_model_len=MAX_MODEL_LEN,
                dtype=runtime_config.model.dtype,
                enforce_eager=runtime_config.model.enforce_eager,
            )

            if hasattr(AsyncLLM, "from_engine_args"):
                self.engine = AsyncLLM.from_engine_args(engine_args)
            else:
                # Fallback for direct instantiation if from_engine_args is missing
                self.engine = AsyncLLM(engine_args)
        except Exception as e:
            logger.error(f"vLLM 엔진 초기화 실패: {e}")
            raise e

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

        # 7. Initialize Agent Loop (세션 기반 에이전트 루프)
        self._init_agent_loop()
        logger.info("AgentLoop 초기화 완료")

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
        """민원 사례 기반 RAG 컨텍스트 문자열을 생성한다."""
        if not retrieved_cases:
            return ""
        rag_context = "### 참고 사례 (유사 민원 및 답변):\n"
        for i, case in enumerate(retrieved_cases):
            safe_complaint = self._escape_special_tokens(case.get("complaint", ""))
            safe_answer = self._escape_special_tokens(case.get("answer", ""))
            rag_context += f"{i+1}. [민원]: {safe_complaint}\n   [답변]: {safe_answer}\n\n"
        return rag_context

    def _augment_prompt(self, prompt: str, retrieved_cases: List[dict]) -> str:
        """기존 민원 답변 프롬프트 증강 규약을 유지하는 호환 메서드."""
        rag_context = self._build_rag_context(retrieved_cases)
        if not rag_context:
            return prompt

        user_tag = "[|user|]"
        if user_tag in prompt:
            return prompt.replace(user_tag, f"{user_tag}{rag_context}\n", 1)
        return f"{rag_context}\n{prompt}"

    def _build_search_result_context(
        self,
        search_results: List[SearchResult],
        heading: str,
    ) -> str:
        """SearchResult 목록을 프롬프트용 컨텍스트 문자열로 변환한다."""
        if not search_results:
            return ""

        lines = [heading]
        for index, result in enumerate(search_results, start=1):
            safe_title = self._escape_special_tokens(result.title)
            safe_content = self._escape_special_tokens(result.content[:300])
            lines.append(f"{index}. [{result.source_type.value}] {safe_title}")
            lines.append(f"   근거: {safe_content}")
        return "\n".join(lines)

    def _build_persona_prompt(
        self,
        primary_agent: str,
        user_message: str,
        fallback_agent: Optional[str] = "generator",
    ) -> str:
        """지정된 페르소나 또는 호환용 fallback 페르소나로 프롬프트를 구성한다."""
        if self.agent_manager:
            for agent_name in (primary_agent, fallback_agent):
                if agent_name and self.agent_manager.get_agent(agent_name):
                    return self.agent_manager.build_prompt(agent_name, user_message)
        return user_message

    @staticmethod
    def _render_public_doc_html(text: str) -> str:
        """공문서 본문을 단순 HTML 단락으로 변환한다."""
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)

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

    def _search_results_to_cases(self, search_results: List[SearchResult]) -> List[dict]:
        """CASE 검색 결과를 RetrievedCase 호환 dict 목록으로 변환한다."""
        retrieved_cases: List[dict] = []
        for result in search_results:
            if result.source_type != IndexType.CASE:
                continue

            metadata = result.metadata or {}
            complaint = metadata.get("complaint_text") or metadata.get("complaint") or result.content
            answer = metadata.get("answer_text") or metadata.get("answer") or result.content
            retrieved_cases.append(
                {
                    "id": result.doc_id,
                    "category": metadata.get("category", ""),
                    "complaint": complaint,
                    "answer": answer,
                    "score": result.score,
                }
            )
        return retrieved_cases

    async def _retrieve_search_results(
        self,
        query: str,
        index_types: List[IndexType],
        top_k_per_type: int = 2,
    ) -> List[SearchResult]:
        """생성 도구용 로컬 RAG 검색 결과를 수집한다."""
        if not query.strip():
            return []

        collected: List[SearchResult] = []

        if self.hybrid_engine:
            async def _search_index(index_type: IndexType) -> List[SearchResult]:
                results_raw, _ = await self.hybrid_engine.search(
                    query=query,
                    index_type=index_type,
                    top_k=top_k_per_type,
                    mode=SearchMode.HYBRID,
                )
                return [
                    SearchResult(
                        doc_id=item.get("doc_id", ""),
                        source_type=IndexType(item.get("doc_type", index_type.value)),
                        title=item.get("title", ""),
                        content=_extract_content_by_type(item, index_type),
                        score=item.get("score", 0.0),
                        reliability_score=item.get("reliability_score", 1.0),
                        metadata=item.get("extras", {}),
                        chunk_index=item.get("chunk_index", 0),
                        total_chunks=item.get("chunk_total", 1),
                    )
                    for item in results_raw
                ]

            grouped_results = await asyncio.gather(
                *[_search_index(index_type) for index_type in index_types],
                return_exceptions=True,
            )

            for result in grouped_results:
                if isinstance(result, BaseException):
                    logger.warning(f"생성용 로컬 검색 실패: {result}")
                    continue
                collected.extend(result)

        elif self.retriever and IndexType.CASE in index_types:
            for raw in self.retriever.search(query, top_k=max(3, top_k_per_type)):
                collected.append(
                    SearchResult(
                        doc_id=raw.get("id", raw.get("doc_id", "")),
                        source_type=IndexType.CASE,
                        title=raw.get("category", "유사 민원 사례"),
                        content=(raw.get("complaint", "") + "\n" + raw.get("answer", "")).strip(),
                        score=raw.get("score", 0.0),
                        reliability_score=raw.get("reliability_score", 1.0),
                        metadata={
                            "complaint": raw.get("complaint", ""),
                            "answer": raw.get("answer", ""),
                            "category": raw.get("category", ""),
                        },
                    )
                )

        return _mask_search_results(collected, self.pii_masker)

    async def _prepare_public_doc_generation(
        self,
        request: GeneratePublicDocRequest,
        flags: "FeatureFlags | None" = None,
    ) -> PreparedGeneration:
        """공문서 생성용 프롬프트와 RAG 컨텍스트를 구성한다."""
        effective_flags = flags or self.feature_flags
        query = self._escape_special_tokens(self._extract_query(request.prompt))
        search_results: List[SearchResult] = []

        if request.use_rag and effective_flags.use_rag_pipeline:
            search_results = await self._retrieve_search_results(
                query,
                [IndexType.LAW, IndexType.MANUAL, IndexType.NOTICE],
            )

        safe_message = self._escape_special_tokens(request.prompt)
        sections = []
        if search_results:
            sections.append(
                self._build_search_result_context(
                    search_results,
                    "### 공문서 작성 참고 자료 (법률/매뉴얼/공시정보):",
                )
            )
        sections.append(
            (
                f"요청된 문서 유형은 '{request.doc_type}' 입니다. "
                "격식 있는 공문서체로 제목, 배경, 주요 내용, 후속 조치를 구조화하여 작성하세요."
            )
        )
        sections.append(safe_message)
        augmented_prompt = self._build_persona_prompt(
            "generator_public_doc",
            "\n\n".join(section for section in sections if section),
        )

        gen_defaults = runtime_config.generation
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or gen_defaults.stop_sequences,
            repetition_penalty=gen_defaults.repetition_penalty,
        )

        return PreparedGeneration(
            prompt=augmented_prompt,
            sampling_params=sampling_params,
            retrieved_cases=[],
            search_results=search_results,
        )

    async def _prepare_civil_response_generation(
        self,
        request: GenerateCivilResponseRequest,
        flags: "FeatureFlags | None" = None,
        external_cases: Optional[List[dict]] = None,
    ) -> PreparedGeneration:
        """민원 답변 생성용 프롬프트와 RAG 컨텍스트를 구성한다."""
        effective_flags = flags or self.feature_flags
        query = self._escape_special_tokens(self._extract_query(request.prompt))
        search_results: List[SearchResult] = []

        if request.use_rag and effective_flags.use_rag_pipeline:
            search_results = await self._retrieve_search_results(
                query,
                [IndexType.CASE, IndexType.LAW, IndexType.MANUAL, IndexType.NOTICE],
            )

        retrieved_cases = self._search_results_to_cases(search_results)
        if external_cases:
            retrieved_cases.extend(external_cases)

        safe_message = self._escape_special_tokens(request.prompt)
        sections = []
        if search_results:
            sections.append(
                self._build_search_result_context(
                    search_results,
                    "### 민원 답변 참고 자료 (사례/법률/매뉴얼/공시정보):",
                )
            )
        if retrieved_cases:
            sections.append(self._build_rag_context(retrieved_cases[:5]))
        sections.append(
            "위 근거를 바탕으로 민원인의 질문에 공감 표현, 처리 절차, 담당 부서 안내를 포함해 답변하세요."
        )
        sections.append(safe_message)
        augmented_prompt = self._build_persona_prompt(
            "generator_civil_response",
            "\n\n".join(section for section in sections if section),
        )

        gen_defaults = runtime_config.generation
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or gen_defaults.stop_sequences,
            repetition_penalty=gen_defaults.repetition_penalty,
        )

        return PreparedGeneration(
            prompt=augmented_prompt,
            sampling_params=sampling_params,
            retrieved_cases=retrieved_cases[:5],
            search_results=search_results,
        )

    async def _run_engine(self, prompt: str, sampling_params: SamplingParams, request_id: str):
        """vLLM 엔진의 generate를 실행하고 최종 결과를 반환한다. (제너레이터 처리 포함)"""
        result = self.engine.generate(prompt, sampling_params, request_id)

        if hasattr(result, "__aiter__"):
            final_output = None
            async for output in result:
                final_output = output
            return final_output
        else:
            return await result

    async def generate(
        self, request: GenerateRequest, request_id: str, flags: "FeatureFlags | None" = None
    ) -> tuple:
        """레거시 /v1/generate 호환용 민원 답변 생성."""
        output, retrieved_cases, _ = await self.generate_civil_response(request, request_id, flags)
        return output, retrieved_cases

    async def generate_public_doc(
        self,
        request: GeneratePublicDocRequest,
        request_id: str,
        flags: "FeatureFlags | None" = None,
    ) -> tuple:
        """공문서 생성. Returns (RequestOutput, search_results)."""
        prepared = await self._prepare_public_doc_generation(request, flags)
        output = await self._run_engine(prepared.prompt, prepared.sampling_params, request_id)
        return output, prepared.search_results

    async def generate_civil_response(
        self,
        request: GenerateCivilResponseRequest,
        request_id: str,
        flags: "FeatureFlags | None" = None,
        external_cases: Optional[List[dict]] = None,
    ) -> tuple:
        """민원 답변 생성. Returns (RequestOutput, retrieved_cases, search_results)."""
        prepared = await self._prepare_civil_response_generation(request, flags, external_cases)
        output = await self._run_engine(prepared.prompt, prepared.sampling_params, request_id)
        return output, prepared.retrieved_cases, prepared.search_results

    def _init_agent_loop(self) -> None:
        """에이전트 루프에 사용할 tool 어댑터를 등록하고 AgentLoop를 생성한다."""
        engine_ref = self  # 클로저에서 참조

        async def _classify_tool(query: str, context: dict, session: Any) -> dict:
            """classify tool 어댑터."""
            if not engine_ref.agent_manager or not engine_ref.agent_manager.get_agent("classifier"):
                return {"classification": None, "error": "분류 에이전트 미로드"}

            classifier = engine_ref.agent_manager.get_agent("classifier")
            safe_prompt = engine_ref._escape_special_tokens(query)
            classify_prompt = engine_ref.agent_manager.build_prompt("classifier", safe_prompt)

            request_id = str(uuid.uuid4())
            sampling_params = SamplingParams(
                temperature=classifier.temperature,
                top_p=0.9,
                max_tokens=classifier.max_tokens,
            )

            final_output = await engine_ref._run_engine(
                classify_prompt, sampling_params, request_id
            )
            if final_output is None:
                return {"classification": None, "error": "분류 처리 실패"}

            response_text = final_output.outputs[0].text
            try:
                json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
                if json_match:
                    cls_result = ClassificationResult.model_validate_json(json_match.group())
                    return {
                        "classification": cls_result.model_dump(),
                        "raw_text": response_text,
                    }
                return {"classification": None, "error": "JSON 파싱 실패"}
            except Exception as e:
                return {"classification": None, "error": f"분류 결과 검증 실패: {e}"}

        from src.inference.actions.data_go_kr import MinwonAnalysisAction

        minwon_action = MinwonAnalysisAction()

        async def _search_similar_tool(query: str, context: dict, session: Any) -> dict:
            """search_similar tool 어댑터."""
            payload = await minwon_action.fetch_similar_cases(query, context)
            results = payload["results"]
            if results is None:
                return {"results": [], "error": "유사 민원 사례 조회 실패"}

            return {
                "results": results,
                "query": payload["query"],
                "count": payload["count"],
                "context_text": payload["context_text"],
                "citations": [citation.to_dict() for citation in payload["citations"]],
                "source": "data.go.kr",
            }

        async def _generate_public_doc_tool(query: str, context: dict, session: Any) -> dict:
            """generate_public_doc tool 어댑터."""
            session_summary = context.get("session_context", "")
            augmented_query = f"{session_summary}\n\n현재 요청: {query}" if session_summary else query

            gen_request = GeneratePublicDocRequest(
                prompt=augmented_query,
                doc_type="official_document",
                max_tokens=768,
                temperature=0.5,
                use_rag=True,
            )

            request_id = str(uuid.uuid4())
            final_output, search_results = await engine_ref.generate_public_doc(
                gen_request,
                request_id,
            )
            if final_output is None:
                return {"text": "", "error": "공문서 생성 실패"}

            text = engine_ref._strip_thought_blocks(final_output.outputs[0].text)
            return {
                "text": text,
                "doc_type": gen_request.doc_type,
                "formatted_html": engine_ref._render_public_doc_html(text),
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "search_results": [result.model_dump() for result in search_results],
            }

        async def _generate_civil_response_tool(query: str, context: dict, session: Any) -> dict:
            """generate_civil_response tool 어댑터."""
            session_summary = context.get("session_context", "")
            augmented_query = f"{session_summary}\n\n현재 요청: {query}" if session_summary else query

            search_similar_data = context.get("search_similar", {})
            external_cases = []
            for item in search_similar_data.get("results", [])[:3]:
                complaint = item.get("content") or item.get("qnaContent") or item.get("question", "")
                answer = item.get("answer") or item.get("qnaAnswer") or item.get("title", "")
                external_cases.append(
                    {
                        "complaint": complaint,
                        "answer": answer,
                        "score": float(item.get("score", 0.0)),
                    }
                )

            gen_request = GenerateCivilResponseRequest(
                prompt=augmented_query,
                max_tokens=512,
                temperature=0.7,
                use_rag=True,
            )

            request_id = str(uuid.uuid4())
            final_output, retrieved_cases, search_results = await engine_ref.generate_civil_response(
                gen_request,
                request_id,
                external_cases=external_cases,
            )
            if final_output is None:
                return {"text": "", "error": "민원 답변 생성 실패"}

            text = engine_ref._strip_thought_blocks(final_output.outputs[0].text)
            return {
                "text": text,
                "retrieved_cases": retrieved_cases,
                "search_results": [result.model_dump() for result in search_results],
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
            }

        tool_registry = {
            ToolType.CLASSIFY: _classify_tool,
            ToolType.SEARCH_SIMILAR: _search_similar_tool,
            ToolType.GENERATE_PUBLIC_DOC: _generate_public_doc_tool,
            ToolType.GENERATE_CIVIL_RESPONSE: _generate_civil_response_tool,
            ToolType.API_LOOKUP: minwon_action,
        }
        self.agent_loop = AgentLoop(tool_registry=tool_registry)
        self._init_langgraph_runtime(
            tool_registry=tool_registry,
            action_registry={"minwon_analysis": minwon_action},
        )

    def _init_langgraph_runtime(
        self,
        tool_registry: Dict[Any, Any],
        action_registry: Dict[str, Any],
    ) -> None:
        """LangGraph foundation runtime을 초기화한다.

        #415 범위에서는 graph runtime의 접합층만 준비한다.
        기존 /v1/agent/* 경로는 유지하고, 이후 #409/#410/#418에서 본격적으로 연결한다.
        """

        try:
            from src.inference.langgraph_runtime import (
                LangGraphDependencyError,
                build_langgraph_runtime,
            )

            self.langgraph_runtime = build_langgraph_runtime(
                runtime_config=runtime_config,
                tool_registry=tool_registry,
                action_registry=action_registry,
            )
            logger.info(
                "LangGraph runtime foundation 초기화 완료: tools={}",
                list(self.langgraph_runtime.tools.keys()),
            )
        except LangGraphDependencyError as exc:
            logger.warning(f"LangGraph runtime foundation 초기화 건너뜀: {exc}")
            self.langgraph_runtime = None
        except Exception as exc:
            logger.warning(f"LangGraph runtime foundation 초기화 실패: {exc}")
            self.langgraph_runtime = None

    async def generate_stream(
        self, request: GenerateRequest, request_id: str, flags: "FeatureFlags | None" = None
    ) -> tuple:
        """레거시 /v1/stream 호환용 민원 답변 스트리밍 생성."""
        prepared = await self._prepare_civil_response_generation(
            request,
            flags,
        )
        # vLLM V1에서는 generate() 자체가 스트림을 반환하므로 stream() 메서드 대신 generate()를 우선적으로 고려한다.
        if hasattr(self.engine, "stream"):
            stream = self.engine.stream(prepared.prompt, prepared.sampling_params, request_id)
        else:
            stream = self.engine.generate(prepared.prompt, prepared.sampling_params, request_id)

        return stream, prepared.retrieved_cases


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
        "profile": runtime_config.profile.value,
        "model": runtime_config.model.model_path,
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

    final_output = await manager._run_engine(classify_prompt, sampling_params, request_id)

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


@app.post("/v1/generate-public-doc", response_model=GeneratePublicDocResponse)
@_rate_limit("30/minute")
async def generate_public_doc(
    request: GeneratePublicDocRequest,
    _: None = Depends(verify_api_key),
    flags: FeatureFlags = Depends(get_feature_flags),
):
    """공문서 초안 생성 엔드포인트."""
    if request.stream:
        raise HTTPException(status_code=400, detail="공문서 스트리밍은 아직 지원하지 않습니다.")

    request_id = str(uuid.uuid4())
    final_output, search_results = await manager.generate_public_doc(request, request_id, flags)

    if final_output is None:
        raise HTTPException(status_code=500, detail="공문서 생성에 실패했습니다.")

    text = manager._strip_thought_blocks(final_output.outputs[0].text)
    return GeneratePublicDocResponse(
        request_id=request_id,
        text=text,
        doc_type=request.doc_type,
        formatted_html=manager._render_public_doc_html(text),
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.outputs[0].token_ids),
        search_results=search_results,
    )


@app.post("/v1/generate-civil-response", response_model=GenerateCivilResponseResponse)
@_rate_limit("30/minute")
async def generate_civil_response(
    request: GenerateCivilResponseRequest,
    _: None = Depends(verify_api_key),
    flags: FeatureFlags = Depends(get_feature_flags),
):
    """민원 답변 초안 생성 엔드포인트."""
    if request.stream:
        raise HTTPException(status_code=400, detail="민원 답변 스트리밍은 /v1/stream을 사용하세요.")

    request_id = str(uuid.uuid4())
    final_output, retrieved_cases, search_results = await manager.generate_civil_response(
        request,
        request_id,
        flags,
    )

    if final_output is None:
        raise HTTPException(status_code=500, detail="민원 답변 생성에 실패했습니다.")

    return GenerateCivilResponseResponse(
        request_id=request_id,
        complaint_id=request.complaint_id,
        text=manager._strip_thought_blocks(final_output.outputs[0].text),
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.outputs[0].token_ids),
        retrieved_cases=[RetrievedCase(**case) for case in retrieved_cases],
        search_results=search_results,
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
    final_output, retrieved_cases = await manager.generate(request, request_id, flags)

    if final_output is None:
        raise HTTPException(status_code=500, detail="Generation failed.")

    return GenerateResponse(
        request_id=request_id,
        complaint_id=request.complaint_id,
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
    results_stream, retrieved_cases = await manager.generate_stream(request, request_id, flags)

    async def stream_results() -> AsyncGenerator[str, None]:
        cases_data = [RetrievedCase(**c).model_dump() for c in retrieved_cases]

        async for request_output in results_stream:
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


# --- Agent Loop 엔드포인트 (#393) ---


def _trace_to_schema(trace: AgentTrace) -> AgentTraceSchema:
    """AgentTrace를 Pydantic 스키마로 변환한다."""
    return AgentTraceSchema(
        request_id=trace.request_id,
        session_id=trace.session_id,
        plan=trace.plan.tool_names if trace.plan else [],
        plan_reason=trace.plan.reason if trace.plan else "",
        tool_results=[
            ToolResultSchema(
                tool=tool_name(r.tool),
                success=r.success,
                latency_ms=round(r.latency_ms, 2),
                data=r.data,
                error=r.error,
            )
            for r in trace.tool_results
        ],
        total_latency_ms=round(trace.total_latency_ms, 2),
        error=trace.error,
    )


@app.post("/v1/agent/run", response_model=AgentRunResponse)
@_rate_limit("30/minute")
async def agent_run(
    request: AgentRunRequest,
    _: None = Depends(verify_api_key),
):
    """세션 기반 에이전트 루프 실행 엔드포인트.

    하나의 요청에서 classify -> search -> generate 흐름을 통합 실행한다.
    스트리밍이 필요하면 /v1/agent/stream을 사용한다.
    """
    if not manager.agent_loop:
        raise HTTPException(status_code=503, detail="에이전트 루프가 초기화되지 않았습니다.")

    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="스트리밍은 /v1/agent/stream 엔드포인트를 사용하세요.",
        )

    session = manager.session_store.get_or_create(session_id=request.session_id)
    request_id = str(uuid.uuid4())

    trace = await manager.agent_loop.run(
        query=request.query,
        session=session,
        request_id=request_id,
        force_tools=request.force_tools,
    )

    if trace.error:
        logger.error(f"에이전트 루프 오류 (request_id={request_id}): {trace.error}")

    # 분류 결과 추출
    classification = None
    search_results = None
    for result in trace.tool_results:
        if tool_name(result.tool) == ToolType.CLASSIFY.value and result.success:
            classification = result.data.get("classification")
        if tool_name(result.tool) == ToolType.SEARCH_SIMILAR.value and result.success:
            search_results = result.data.get("results")

    return AgentRunResponse(
        request_id=request_id,
        session_id=session.session_id,
        text=trace.final_text,
        trace=_trace_to_schema(trace),
        classification=classification,
        search_results=search_results,
    )


@app.post("/v1/agent/stream")
@_rate_limit("30/minute")
async def agent_stream(
    request: AgentRunRequest,
    _: None = Depends(verify_api_key),
):
    """세션 기반 에이전트 루프 스트리밍 엔드포인트.

    각 단계의 진행 상황과 최종 결과를 SSE 이벤트로 전달한다.
    """
    if not manager.agent_loop:
        raise HTTPException(status_code=503, detail="에이전트 루프가 초기화되지 않았습니다.")

    session = manager.session_store.get_or_create(session_id=request.session_id)
    request_id = str(uuid.uuid4())

    async def stream_events() -> AsyncGenerator[str, None]:
        async for event in manager.agent_loop.run_stream(
            query=request.query,
            session=session,
            request_id=request_id,
            force_tools=request.force_tools,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, **runtime_config.to_uvicorn_kwargs())
