import asyncio
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

SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "false").lower() in ("true", "1", "yes")

from .agent_loop import AgentLoop, AgentTrace
from .agent_manager import AgentManager
from .bm25_indexer import BM25Indexer
from .feature_flags import FeatureFlags
from .hybrid_search import HybridSearchEngine, SearchMode
from .index_manager import IndexType, MultiIndexManager
from .retriever import CivilComplaintRetriever
from .runtime_config import RuntimeConfig
from .schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentTraceSchema,
    GenerateCivilResponseRequest,
    GenerateCivilResponseResponse,
    GenerateRequest,
    GenerateResponse,
    RetrievedCase,
    SearchRequest,
    SearchResponse,
    SearchResult,
    ToolResultSchema,
)
from .session_context import SessionContext, SessionStore
from .tool_router import ToolType, tool_name


async def _noop_tool(query: str, context: dict, session: Any) -> dict:
    """build_mvp_registry fallback용 no-op tool."""
    return {"success": False, "error": "tool이 초기화되지 않았습니다"}


if not SKIP_MODEL_LOAD:
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs

        from .vllm_stabilizer import apply_transformers_patch
    except ImportError:
        logger.warning("vllm modules not found. Model loading will fail if attempted.")
        AsyncEngineArgs = object
        apply_transformers_patch = lambda: None

try:
    from slowapi import Limiter
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    _RATE_LIMIT_AVAILABLE = True
except ImportError:
    limiter = None
    _RATE_LIMIT_AVAILABLE = False

_API_KEY = os.getenv("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    if _API_KEY is None:
        return
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


runtime_config = RuntimeConfig.from_env()
runtime_config.log_summary()

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
    prompt: str
    sampling_params: SamplingParams
    retrieved_cases: List[dict]
    search_results: List[SearchResult]


if not SKIP_MODEL_LOAD:
    apply_transformers_patch()


def _extract_content_by_type(result: dict, index_type: IndexType) -> str:
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


class vLLMEngineManager:
    """GovOn Shell MVP용 로컬 런타임 매니저."""

    def __init__(self):
        self.engine: AsyncLLM = None
        self.retriever: CivilComplaintRetriever = None
        self.index_manager: Optional[MultiIndexManager] = None
        self.hybrid_engine: Optional[HybridSearchEngine] = None
        self.bm25_indexers: dict[IndexType, BM25Indexer] = {}
        self.embed_model = None
        self.feature_flags = FeatureFlags.from_env()
        self.session_store = SessionStore()
        self.agent_manager = AgentManager(AGENTS_DIR)
        self.agent_loop: Optional[AgentLoop] = None
        self.graph = None  # LangGraph CompiledGraph (v2 엔드포인트용)
        self._init_agent_loop()
        self._init_graph()

    async def initialize(self):
        if SKIP_MODEL_LOAD:
            logger.info("SKIP_MODEL_LOAD=true: 모델 및 인덱스 로딩을 건너뜁니다.")
            return

        logger.info(f"Initializing vLLM runtime with model: {MODEL_PATH}")
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
                self.engine = AsyncLLM(engine_args)
        except Exception as exc:
            logger.error(f"vLLM 엔진 초기화 실패: {exc}")
            raise

        logger.info(f"Initializing retriever with index: {INDEX_PATH}")
        self.retriever = CivilComplaintRetriever(
            index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
            data_path=DATA_PATH if not os.path.exists(INDEX_PATH) else None,
        )
        if self.retriever.index is not None and not os.path.exists(INDEX_PATH):
            self.retriever.save_index(INDEX_PATH)

        faiss_index_dir = os.getenv("FAISS_INDEX_DIR", "models/faiss_index")
        if os.path.isdir(faiss_index_dir):
            self.index_manager = MultiIndexManager(base_dir=faiss_index_dir)
            logger.info(f"MultiIndexManager 초기화 완료: {faiss_index_dir}")
        else:
            logger.warning(f"FAISS 인덱스 디렉터리 미존재: {faiss_index_dir}")

        bm25_index_dir = os.getenv("BM25_INDEX_DIR", "models/bm25_index")
        if os.path.isdir(bm25_index_dir):
            for idx_type in IndexType:
                bm25_path = os.path.join(bm25_index_dir, f"{idx_type.value}.pkl")
                if not os.path.exists(bm25_path):
                    continue
                try:
                    indexer = BM25Indexer()
                    indexer.load(bm25_path)
                    self.bm25_indexers[idx_type] = indexer
                    logger.info(f"BM25 인덱스 로드 완료: {idx_type.value} ({indexer.doc_count}건)")
                except Exception as exc:
                    logger.warning(f"BM25 인덱스 로드 실패 ({idx_type.value}): {exc}")

        if self.retriever and hasattr(self.retriever, "model"):
            self.embed_model = self.retriever.model

        if self.index_manager and self.embed_model:
            self.hybrid_engine = HybridSearchEngine(
                index_manager=self.index_manager,
                bm25_indexers=self.bm25_indexers,
                embed_model=self.embed_model,
            )
            logger.info("HybridSearchEngine 초기화 완료")
        else:
            logger.warning("HybridSearchEngine 미초기화: index_manager 또는 embed_model 없음")

    def _escape_special_tokens(self, text: str) -> str:
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
        return re.sub(r"<thought>.*?</thought>\s*", "", text, flags=re.DOTALL).strip()

    def _build_rag_context(self, retrieved_cases: List[dict]) -> str:
        if not retrieved_cases:
            return ""
        rag_context = "### 참고 사례 (유사 민원 및 답변):\n"
        for i, case in enumerate(retrieved_cases, start=1):
            complaint = self._escape_special_tokens(case.get("complaint", ""))
            answer = self._escape_special_tokens(case.get("answer", ""))
            rag_context += f"{i}. [민원]: {complaint}\n   [답변]: {answer}\n\n"
        return rag_context

    def _augment_prompt(self, prompt: str, retrieved_cases: List[dict]) -> str:
        rag_context = self._build_rag_context(retrieved_cases)
        if not rag_context:
            return prompt
        user_tag = "[|user|]"
        if user_tag in prompt:
            return prompt.replace(user_tag, f"{user_tag}{rag_context}\n", 1)
        return f"{rag_context}\n{prompt}"

    def _build_search_result_context(self, search_results: List[SearchResult], heading: str) -> str:
        if not search_results:
            return ""

        lines = [heading]
        for index, result in enumerate(search_results, start=1):
            safe_title = self._escape_special_tokens(result.title)
            safe_content = self._escape_special_tokens(result.content[:300])
            lines.append(f"{index}. [{result.source_type.value}] {safe_title}")
            lines.append(f"   근거: {safe_content}")
        return "\n".join(lines)

    def _build_persona_prompt(self, agent_name: str, user_message: str) -> str:
        if self.agent_manager and self.agent_manager.get_agent(agent_name):
            return self.agent_manager.build_prompt(agent_name, user_message)
        return user_message

    def _extract_query(self, prompt: str) -> str:
        user_match = re.search(r"\[\|user\|\](.*?)\[\|endofturn\|\]", prompt, re.DOTALL)
        if user_match:
            user_block = user_match.group(1)
            complaint_match = re.search(r"민원\s*내용\s*:\s*(.+)", user_block, re.DOTALL)
            if complaint_match:
                return complaint_match.group(1).strip()
            return user_block.strip()
        return prompt

    def _search_results_to_cases(self, search_results: List[SearchResult]) -> List[dict]:
        retrieved_cases: List[dict] = []
        for result in search_results:
            if result.source_type != IndexType.CASE:
                continue
            metadata = result.metadata or {}
            complaint = (
                metadata.get("complaint_text") or metadata.get("complaint") or result.content
            )
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

    @staticmethod
    def _is_evidence_request(query: str) -> bool:
        return any(token in query for token in ("근거", "출처", "왜", "이유", "링크"))

    @staticmethod
    def _is_revision_request(query: str) -> bool:
        return any(token in query for token in ("다시", "수정", "고쳐", "정중", "공손", "보강"))

    def _latest_prior_turns(
        self,
        session: SessionContext,
        current_query: str,
    ) -> tuple[Optional[str], Optional[str]]:
        turns = list(session.recent_history)
        if turns and turns[-1].role == "user" and turns[-1].content == current_query:
            turns = turns[:-1]

        previous_user = next(
            (turn.content for turn in reversed(turns) if turn.role == "user"), None
        )
        previous_assistant = next(
            (turn.content for turn in reversed(turns) if turn.role == "assistant"),
            None,
        )
        return previous_user, previous_assistant

    def _build_working_query(self, query: str, session: SessionContext) -> str:
        query = query.strip()
        if not query:
            return query

        if not (self._is_evidence_request(query) or self._is_revision_request(query)):
            return query

        previous_user, previous_assistant = self._latest_prior_turns(session, query)
        parts: List[str] = []
        if previous_user:
            parts.append(f"원래 요청: {previous_user}")
        if previous_assistant:
            parts.append(f"이전 답변: {previous_assistant[:600]}")

        if self._is_revision_request(query):
            parts.append(f"수정 요청: {query}")

        return "\n\n".join(parts) if parts else query

    async def _retrieve_search_results(
        self,
        query: str,
        index_types: List[IndexType],
        top_k_per_type: int = 2,
    ) -> List[SearchResult]:
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

            grouped = await asyncio.gather(
                *[_search_index(index_type) for index_type in index_types],
                return_exceptions=True,
            )
            for result in grouped:
                if isinstance(result, BaseException):
                    logger.warning(f"로컬 검색 실패: {result}")
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

        return collected

    def _summarize_evidence(
        self,
        search_results: List[SearchResult],
        api_lookup_data: Dict[str, Any],
    ) -> str:
        lines = ["근거 요약"]

        if search_results:
            titles = ", ".join(result.title for result in search_results[:3] if result.title)
            lines.append(
                f"- 로컬 문서 {len(search_results)}건을 참고했습니다."
                + (f" 주요 문서: {titles}" if titles else "")
            )

        api_results = api_lookup_data.get("results", [])
        if api_results:
            titles = []
            for item in api_results[:3]:
                title = item.get("title") or item.get("qnaTitle") or item.get("question")
                if title:
                    titles.append(title)
            lines.append(
                f"- 외부 민원분석 API에서 유사 사례 {len(api_results)}건을 확인했습니다."
                + (f" 대표 사례: {', '.join(titles)}" if titles else "")
            )

        if len(lines) == 1:
            lines.append(
                "- 내부 검색 결과를 충분히 확보하지 못해 일반 행정 응대 원칙 기준으로 작성했습니다."
            )

        return "\n".join(lines)

    @staticmethod
    def _rag_source_line(index: int, item: Dict[str, Any]) -> str:
        metadata = item.get("metadata", {}) or {}
        location = (
            metadata.get("file_path")
            or metadata.get("source_path")
            or metadata.get("path")
            or metadata.get("source")
            or item.get("title")
            or item.get("doc_id")
            or "로컬 문서"
        )
        page = metadata.get("page") or metadata.get("page_number") or metadata.get("page_no")
        if page:
            return f"[{index}] {location} (p.{page})"
        return f"[{index}] {location}"

    @staticmethod
    def _api_source_line(index: int, item: Dict[str, Any]) -> str:
        title = item.get("title") or item.get("qnaTitle") or item.get("question") or "외부 API 결과"
        url = item.get("url") or item.get("detailUrl") or ""
        if url:
            return f"[{index}] {title} - {url}"
        return f"[{index}] {title}"

    def _build_evidence_section(
        self,
        session: SessionContext,
        current_query: str,
        rag_data: Dict[str, Any],
        api_data: Dict[str, Any],
    ) -> str:
        _, previous_answer = self._latest_prior_turns(session, current_query)
        lines = ["근거/출처"]
        cursor = 1

        for item in rag_data.get("results", [])[:5]:
            lines.append(self._rag_source_line(cursor, item))
            cursor += 1

        api_items = api_data.get("citations") or api_data.get("results") or []
        for item in api_items[:5]:
            lines.append(self._api_source_line(cursor, item))
            cursor += 1

        if cursor == 1:
            lines.append("- 검색 가능한 근거를 찾지 못했습니다.")

        section = "\n".join(lines)
        if previous_answer:
            return f"{previous_answer}\n\n{section}"
        return section

    async def _prepare_civil_response_generation(
        self,
        request: GenerateCivilResponseRequest,
        flags: Optional[FeatureFlags] = None,
        external_cases: Optional[List[dict]] = None,
    ) -> PreparedGeneration:
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
            "위 근거를 바탕으로 민원인의 불편에 공감하고, 현재 조치 상황과 처리 절차를 포함한 회신 초안을 작성하세요."
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
        if self.engine is None:
            return None

        result = self.engine.generate(prompt, sampling_params, request_id)
        if hasattr(result, "__aiter__"):
            final_output = None
            async for output in result:
                final_output = output
            return final_output
        return await result

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str,
        flags: Optional[FeatureFlags] = None,
    ) -> tuple[Any, List[dict]]:
        output, retrieved_cases, _ = await self.generate_civil_response(request, request_id, flags)
        return output, retrieved_cases

    async def generate_civil_response(
        self,
        request: GenerateCivilResponseRequest,
        request_id: str,
        flags: Optional[FeatureFlags] = None,
        external_cases: Optional[List[dict]] = None,
    ) -> tuple[Any, List[dict], List[SearchResult]]:
        prepared = await self._prepare_civil_response_generation(request, flags, external_cases)
        output = await self._run_engine(prepared.prompt, prepared.sampling_params, request_id)
        return output, prepared.retrieved_cases, prepared.search_results

    async def generate_stream(
        self,
        request: GenerateRequest,
        request_id: str,
        flags: Optional[FeatureFlags] = None,
    ) -> tuple[Any, List[dict], List[SearchResult]]:
        prepared = await self._prepare_civil_response_generation(request, flags)
        if self.engine is None:
            raise RuntimeError("모델 엔진이 초기화되지 않았습니다.")
        if hasattr(self.engine, "stream"):
            stream = self.engine.stream(prepared.prompt, prepared.sampling_params, request_id)
        else:
            stream = self.engine.generate(prepared.prompt, prepared.sampling_params, request_id)
        return stream, prepared.retrieved_cases, prepared.search_results

    def _init_agent_loop(self) -> None:
        from src.inference.actions.data_go_kr import MinwonAnalysisAction

        engine_ref = self
        minwon_action = MinwonAnalysisAction()

        async def _rag_search_tool(query: str, context: dict, session: SessionContext) -> dict:
            working_query = engine_ref._build_working_query(query, session)
            search_results = await engine_ref._retrieve_search_results(
                working_query,
                [IndexType.CASE, IndexType.LAW, IndexType.MANUAL, IndexType.NOTICE],
            )
            return {
                "query": working_query,
                "count": len(search_results),
                "results": [result.model_dump() for result in search_results],
                "context_text": engine_ref._build_search_result_context(
                    search_results,
                    "### 로컬 문서 검색 결과:",
                ),
            }

        async def _api_lookup_tool(query: str, context: dict, session: SessionContext) -> dict:
            working_query = engine_ref._build_working_query(query, session)
            payload = await minwon_action.fetch_similar_cases(
                working_query,
                {
                    **context,
                    "session_context": session.build_context_summary(),
                },
            )
            results = payload["results"] or []
            return {
                "query": payload["query"],
                "count": len(results),
                "results": results,
                "context_text": payload["context_text"],
                "citations": [citation.to_dict() for citation in payload["citations"]],
                "source": "data.go.kr",
            }

        async def _draft_civil_response_tool(
            query: str,
            context: dict,
            session: SessionContext,
        ) -> dict:
            working_query = engine_ref._build_working_query(query, session)
            api_lookup_data = context.get(ToolType.API_LOOKUP.value, {})

            external_cases = []
            for item in api_lookup_data.get("results", [])[:3]:
                complaint = (
                    item.get("content") or item.get("qnaContent") or item.get("question", "")
                )
                answer = item.get("answer") or item.get("qnaAnswer") or item.get("title", "")
                if complaint or answer:
                    external_cases.append(
                        {
                            "complaint": complaint,
                            "answer": answer,
                            "score": float(item.get("score", 0.0)),
                        }
                    )

            gen_request = GenerateCivilResponseRequest(
                prompt=working_query,
                max_tokens=512,
                temperature=0.7,
                use_rag=True,
            )
            request_id = str(uuid.uuid4())
            final_output, retrieved_cases, search_results = (
                await engine_ref.generate_civil_response(
                    gen_request,
                    request_id,
                    external_cases=external_cases,
                )
            )
            if final_output is None:
                return {"text": "", "error": "민원 답변 생성 실패"}

            draft_text = engine_ref._strip_thought_blocks(final_output.outputs[0].text)
            text = (
                engine_ref._summarize_evidence(search_results, api_lookup_data)
                + "\n\n최종 초안\n"
                + draft_text
            )
            return {
                "text": text,
                "draft_text": draft_text,
                "retrieved_cases": retrieved_cases,
                "search_results": [result.model_dump() for result in search_results],
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
            }

        async def _append_evidence_tool(
            query: str,
            context: dict,
            session: SessionContext,
        ) -> dict:
            rag_data = context.get(ToolType.RAG_SEARCH.value, {})
            api_data = context.get(ToolType.API_LOOKUP.value, {})
            return {
                "text": engine_ref._build_evidence_section(session, query, rag_data, api_data),
                "rag_results": rag_data.get("results", []),
                "api_citations": api_data.get("citations", []),
            }

        tool_registry = {
            ToolType.RAG_SEARCH: _rag_search_tool,
            ToolType.API_LOOKUP: _api_lookup_tool,
            ToolType.DRAFT_CIVIL_RESPONSE: _draft_civil_response_tool,
            ToolType.APPEND_EVIDENCE: _append_evidence_tool,
        }
        self.agent_loop = AgentLoop(tool_registry=tool_registry)

    def _build_tool_registry(self) -> Dict[str, Any]:
        """CapabilityBase 기반 MVP tool registry를 반환한다.

        build_mvp_registry()를 사용하여 단일 소스에서 registry를 구성한다.
        planner metadata와 executor binding이 동일한 CapabilityBase 인스턴스에서 나온다.
        AgentLoop 하위호환: AgentLoop._tools에서 closure를 추출하여 wrapper로 래핑한다.
        """
        if self.agent_loop is None:
            return {}

        from src.inference.graph.capabilities.registry import build_mvp_registry

        # AgentLoop의 tool_registry에서 기존 closure를 추출
        raw_tools = {
            str(k.value if hasattr(k, "value") else k): v for k, v in self.agent_loop._tools.items()
        }

        return build_mvp_registry(
            rag_search_fn=raw_tools.get("rag_search", _noop_tool),
            api_lookup_action=self._get_api_lookup_action(),
            draft_civil_response_fn=raw_tools.get("draft_civil_response", _noop_tool),
            append_evidence_fn=raw_tools.get("append_evidence", _noop_tool),
        )

    def _get_api_lookup_action(self) -> Any:
        """AgentLoop에 등록된 api_lookup의 MinwonAnalysisAction을 추출한다."""
        if self.agent_loop is None:
            return None
        tool_fn = self.agent_loop._tools.get(ToolType.API_LOOKUP)
        # ApiLookupCapability인 경우 action을 직접 추출
        if hasattr(tool_fn, "_action"):
            return tool_fn._action
        # closure인 경우 action을 추출할 수 없으므로 None 반환
        # (MinwonAnalysisAction은 _init_agent_loop에서 새로 생성한다)
        try:
            from src.inference.actions.data_go_kr import MinwonAnalysisAction

            return MinwonAnalysisAction()
        except Exception:
            return None

    def _init_graph(self) -> None:
        """LangGraph StateGraph를 초기화한다.

        MVP에서는 RegexPlannerAdapter(정규식 기반 fallback)와
        RegistryExecutorAdapter(기존 tool_registry 재사용)를 사용한다.
        checkpoint는 MemorySaver를 기본으로 사용하며,
        GOVON_HOME 환경변수로 AsyncSqliteSaver 경로를 설정할 수 있다.
        """
        try:
            from src.inference.graph.builder import build_govon_graph
            from src.inference.graph.executor_adapter import RegistryExecutorAdapter
            from src.inference.graph.planner_adapter import RegexPlannerAdapter
        except ImportError as exc:
            logger.warning(f"LangGraph graph 초기화 실패 (import 오류): {exc}")
            return

        tool_registry = self._build_tool_registry()
        planner = RegexPlannerAdapter()
        executor = RegistryExecutorAdapter(
            tool_registry=tool_registry,
            session_store=self.session_store,
        )

        # MVP: MemorySaver 사용 (AsyncSqliteSaver는 async context manager가 필요하므로
        # daemon startup의 lifespan event에서 별도 처리)
        checkpointer = None

        self.graph = build_govon_graph(
            planner_adapter=planner,
            executor_adapter=executor,
            session_store=self.session_store,
            checkpointer=checkpointer,
        )
        logger.info("LangGraph graph 초기화 완료")


manager = vLLMEngineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await manager.initialize()
    yield


app = FastAPI(
    title="GovOn Local Runtime",
    description="Local FastAPI daemon for the GovOn Agentic Shell MVP.",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
if ALLOWED_ORIGINS and ALLOWED_ORIGINS[0]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if _RATE_LIMIT_AVAILABLE and limiter is not None:
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)


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
        "feature_flags": {
            "use_rag_pipeline": manager.feature_flags.use_rag_pipeline,
            "model_version": manager.feature_flags.model_version,
        },
        "session_store": {
            "driver": "sqlite",
            "path": manager.session_store.db_path,
        },
    }


def _rate_limit(limit_string: str):
    if _RATE_LIMIT_AVAILABLE and limiter is not None:
        return limiter.limit(limit_string)

    def _noop(func):
        return func

    return _noop


def get_feature_flags(request: Request) -> FeatureFlags:
    header = request.headers.get("X-Feature-Flag")
    return manager.feature_flags.override_from_header(header)


@app.post("/v1/generate-civil-response", response_model=GenerateCivilResponseResponse)
@_rate_limit("30/minute")
async def generate_civil_response(
    request: GenerateCivilResponseRequest,
    _: None = Depends(verify_api_key),
    flags: FeatureFlags = Depends(get_feature_flags),
):
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
        retrieved_cases=[RetrievedCase(**case) for case in retrieved_cases],
    )


@app.post("/v1/stream")
@_rate_limit("30/minute")
async def stream_generate(
    request: GenerateRequest,
    _: None = Depends(verify_api_key),
    flags: FeatureFlags = Depends(get_feature_flags),
):
    if not request.stream:
        request.stream = True

    request_id = str(uuid.uuid4())
    results_stream, retrieved_cases, search_results = await manager.generate_stream(
        request,
        request_id,
        flags,
    )

    async def stream_results() -> AsyncGenerator[str, None]:
        cases_data = [RetrievedCase(**case).model_dump() for case in retrieved_cases]
        search_data = [result.model_dump() for result in search_results]

        async for request_output in results_stream:
            text = request_output.outputs[0].text
            finished = request_output.finished
            if finished:
                text = manager._strip_thought_blocks(text)

            response_obj = {"request_id": request_id, "text": text, "finished": finished}
            if finished:
                response_obj["retrieved_cases"] = cases_data
                response_obj["search_results"] = search_data

            yield f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")


@app.post("/v1/search", response_model=SearchResponse)
@app.post("/search", response_model=SearchResponse)
@_rate_limit("60/minute")
async def search(request: SearchRequest, _: Request, __: None = Depends(verify_api_key)):
    start_time = time.monotonic()
    try:
        if manager.hybrid_engine:
            results_raw, actual_mode = await manager.hybrid_engine.search(
                query=request.query,
                index_type=request.doc_type,
                top_k=request.top_k,
                mode=request.search_mode,
            )
            results = [
                SearchResult(
                    doc_id=result.get("doc_id", ""),
                    source_type=IndexType(result.get("doc_type", request.doc_type.value)),
                    title=result.get("title", ""),
                    content=_extract_content_by_type(result, request.doc_type),
                    score=result.get("score", 0.0),
                    reliability_score=result.get("reliability_score", 1.0),
                    metadata=result.get("extras", {}),
                    chunk_index=result.get("chunk_index", 0),
                    total_chunks=result.get("chunk_total", 1),
                )
                for result in results_raw
            ]
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
            raise HTTPException(status_code=503, detail="검색 엔진이 아직 초기화되지 않았습니다.")

        elapsed_ms = (time.monotonic() - start_time) * 1000
        actual_search_mode = actual_mode if actual_mode != request.search_mode else None
        return SearchResponse(
            query=request.query,
            doc_type=request.doc_type,
            search_mode=request.search_mode,
            actual_search_mode=actual_search_mode,
            results=results,
            total=len(results),
            search_time_ms=round(elapsed_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"검색 중 오류 발생: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="검색 처리 중 내부 오류가 발생했습니다.")


def _trace_to_schema(trace: AgentTrace) -> AgentTraceSchema:
    return AgentTraceSchema(
        request_id=trace.request_id,
        session_id=trace.session_id,
        plan=trace.plan.tool_names if trace.plan else [],
        plan_reason=trace.plan.reason if trace.plan else "",
        tool_results=[
            ToolResultSchema(
                tool=tool_name(result.tool),
                success=result.success,
                latency_ms=round(result.latency_ms, 2),
                data=result.data,
                error=result.error,
            )
            for result in trace.tool_results
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
    if not manager.agent_loop:
        raise HTTPException(status_code=503, detail="에이전트 루프가 초기화되지 않았습니다.")
    if request.stream:
        raise HTTPException(status_code=400, detail="스트리밍은 /v1/agent/stream을 사용하세요.")

    session = manager.session_store.get_or_create(session_id=request.session_id)
    request_id = str(uuid.uuid4())
    trace = await manager.agent_loop.run(
        query=request.query,
        session=session,
        request_id=request_id,
        force_tools=request.force_tools,
    )

    search_results = None
    for result in trace.tool_results:
        if tool_name(result.tool) == ToolType.RAG_SEARCH.value and result.success:
            search_results = result.data.get("results")
        elif (
            tool_name(result.tool) == ToolType.API_LOOKUP.value
            and result.success
            and not search_results
        ):
            search_results = result.data.get("results")

    return AgentRunResponse(
        request_id=request_id,
        session_id=session.session_id,
        text=trace.final_text,
        trace=_trace_to_schema(trace),
        search_results=search_results,
    )


@app.post("/v1/agent/stream")
@_rate_limit("30/minute")
async def agent_stream(
    request: AgentRunRequest,
    _: None = Depends(verify_api_key),
):
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


# ---------------------------------------------------------------------------
# v2 엔드포인트: LangGraph 기반 agent 실행 (interrupt/approve 패턴)
# ---------------------------------------------------------------------------


@app.post("/v2/agent/run")
async def v2_agent_run(
    request: AgentRunRequest,
    _: None = Depends(verify_api_key),
):
    """LangGraph 기반 agent 실행 (1단계: interrupt까지).

    graph를 실행하여 `approval_wait` 노드에서 interrupt되면
    `status: awaiting_approval`과 함께 승인 요청 정보를 반환한다.

    클라이언트는 반환된 `thread_id`를 저장해두고
    `/v2/agent/approve`로 승인/거절을 전달해야 한다.
    """
    if not manager.graph:
        raise HTTPException(status_code=503, detail="LangGraph graph가 초기화되지 않았습니다.")

    from langchain_core.messages import HumanMessage

    thread_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "session_id": request.session_id,
        "request_id": str(uuid.uuid4()),
        "messages": [HumanMessage(content=request.query)],
    }

    # interrupt()는 sync invoke 경로에서 가장 안정적으로 동작한다.
    await asyncio.to_thread(manager.graph.invoke, initial_state, config)

    # interrupt 상태 확인
    graph_state = await asyncio.to_thread(manager.graph.get_state, config)
    if graph_state.next:
        # interrupt 대기 중: approval_request 정보를 클라이언트에 반환
        approval_value = None
        if graph_state.tasks and graph_state.tasks[0].interrupts:
            approval_value = graph_state.tasks[0].interrupts[0].value
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "approval_request": approval_value,
        }

    # interrupt 없이 완료된 경우 (rejected 또는 오류)
    final_state = graph_state.values
    return {
        "status": "completed",
        "thread_id": thread_id,
        "text": final_state.get("final_text", ""),
    }


@app.post("/v2/agent/approve")
async def v2_agent_approve(
    thread_id: str,
    approved: bool,
    _: None = Depends(verify_api_key),
):
    """interrupt된 graph를 resume한다 (2단계: 승인/거절).

    Parameters
    ----------
    thread_id : str
        `/v2/agent/run`에서 반환된 thread_id.
    approved : bool
        True면 tool_execute로 진행, False면 graph가 END로 종료.
    """
    if not manager.graph:
        raise HTTPException(status_code=503, detail="LangGraph graph가 초기화되지 않았습니다.")

    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}

    # resume도 동일한 sync invoke 경로로 처리한다.
    result = await asyncio.to_thread(
        manager.graph.invoke,
        Command(resume={"approved": approved}),
        config,
    )

    return {
        "status": "completed",
        "thread_id": thread_id,
        "text": result.get("final_text", ""),
        "tool_results": result.get("tool_results", {}),
        "approval_status": result.get("approval_status", ""),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, **runtime_config.to_uvicorn_kwargs())
