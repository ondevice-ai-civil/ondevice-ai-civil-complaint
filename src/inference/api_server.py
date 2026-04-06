import asyncio
import json
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
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

SessionLocal = None
LocalDocumentIndexer = None
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "false").lower() in ("true", "1", "yes")


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
        case_text = "\n".join(
            part
            for part in (extras.get("complaint_text", ""), extras.get("answer_text", ""))
            if part
        ).strip()
        text = case_text or extras.get("content", "") or extras.get("chunk_text", "")
    elif index_type == IndexType.LAW:
        text = (
            extras.get("law_text", "") or extras.get("content", "") or extras.get("chunk_text", "")
        )
    elif index_type == IndexType.MANUAL:
        text = (
            extras.get("manual_text", "")
            or extras.get("content", "")
            or extras.get("chunk_text", "")
        )
    elif index_type == IndexType.NOTICE:
        text = (
            extras.get("notice_text", "")
            or extras.get("content", "")
            or extras.get("chunk_text", "")
        )
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
        self.local_document_indexer: Optional[Any] = None
        self.local_document_sync_status: Optional[Dict[str, Any]] = None
        self._local_document_sync_task: Optional[asyncio.Task] = None
        self._checkpointer_ctx = None  # AsyncSqliteSaver 컨텍스트 매니저 (lifespan에서 관리)
        self._sync_checkpointer_conn = None  # SqliteSaver용 sqlite3 connection (leak 방지)
        self._init_agent_loop()
        self._init_graph()

    async def initialize(self):
        if SKIP_MODEL_LOAD:
            logger.info("SKIP_MODEL_LOAD=true: 모델 및 인덱스 로딩을 건너뜁니다.")
            return

        logger.info(f"Initializing vLLM runtime with model: {MODEL_PATH}")
        # EXAONE 4.0-32B-AWQ 네이티브 tool calling 활성화:
        #   --enable-auto-tool-choice --tool-call-parser hermes
        # Multi-LoRA 서빙 시 --enable-lora --lora-modules 옵션 추가
        # HuggingFace Spaces L4 (24GB VRAM) 기준 ~20GB 점유
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

        faiss_index_dir = runtime_config.paths.faiss_index_dir
        local_docs_root = runtime_config.paths.local_docs_root
        if os.path.isdir(faiss_index_dir) or local_docs_root:
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
            self._schedule_local_document_sync()
        else:
            logger.warning("HybridSearchEngine 미초기화: index_manager 또는 embed_model 없음")

    def _schedule_local_document_sync(self) -> None:
        indexer = self._build_local_document_indexer()
        if indexer is None:
            return
        if self._local_document_sync_task and not self._local_document_sync_task.done():
            return

        self.local_document_sync_status = {
            "status": "syncing",
            "root_dir": str(indexer.root_dir),
            "source_name": indexer.source_name,
        }
        self._local_document_sync_task = asyncio.create_task(self._sync_local_documents_async())

    async def _sync_local_documents_async(self) -> Optional[Dict[str, Any]]:
        try:
            return await asyncio.to_thread(self.sync_local_documents)
        except Exception as exc:
            logger.error(f"백그라운드 로컬 문서 인덱싱 실패: {exc}", exc_info=True)
            if self.local_document_indexer is None:
                return None
            self.local_document_sync_status = {
                "status": "error",
                "root_dir": str(self.local_document_indexer.root_dir),
                "source_name": self.local_document_indexer.source_name,
                "error": str(exc),
            }
            return self.local_document_sync_status

    def _build_local_document_indexer(self) -> Optional[Any]:
        global SessionLocal, LocalDocumentIndexer

        root_dir = runtime_config.paths.local_docs_root
        if not root_dir:
            return None
        if self.index_manager is None or self.embed_model is None:
            logger.warning(
                "LOCAL_DOCS_ROOT가 설정됐지만 index_manager 또는 embed_model이 없습니다."
            )
            return None
        if self.local_document_indexer is None:
            if SessionLocal is None:
                from .db.database import SessionLocal as _SessionLocal

                SessionLocal = _SessionLocal
            if LocalDocumentIndexer is None:
                from .local_document_indexer import LocalDocumentIndexer as _LocalDocumentIndexer

                LocalDocumentIndexer = _LocalDocumentIndexer

            self.local_document_indexer = LocalDocumentIndexer(
                root_dir=root_dir,
                index_manager=self.index_manager,
                embed_model=self.embed_model,
                session_factory=SessionLocal,
            )
        return self.local_document_indexer

    def sync_local_documents(self) -> Optional[Dict[str, Any]]:
        indexer = self._build_local_document_indexer()
        if indexer is None:
            return None

        try:
            summary = indexer.sync()
        except Exception as exc:
            logger.error(f"로컬 문서 인덱싱 실패: {exc}", exc_info=True)
            self.local_document_sync_status = {
                "status": "error",
                "root_dir": str(indexer.root_dir),
                "source_name": indexer.source_name,
                "error": str(exc),
            }
            return self.local_document_sync_status

        self.local_document_sync_status = {
            "status": "ok",
            "root_dir": str(indexer.root_dir),
            "source_name": indexer.source_name,
            **asdict(summary),
        }
        logger.info(
            "로컬 문서 인덱싱 완료: "
            f"root={indexer.root_dir}, scanned={summary.scanned_files}, "
            f"indexed={summary.indexed_files}, unchanged={summary.unchanged_files}, "
            f"removed={summary.removed_files}"
        )
        return self.local_document_sync_status

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

    @staticmethod
    def _format_evidence_items(evidence_dict: Dict[str, Any]) -> str:
        """EvidenceEnvelope dict를 소비하여 출처 목록 텍스트를 생성한다.

        EvidenceItem이 있으면 source-specific branching 없이 단일 포매터로 처리한다.
        """
        items = evidence_dict.get("items", [])
        if not items:
            return ""

        lines: list[str] = []
        for idx, item in enumerate(items[:10], start=1):
            source_type = item.get("source_type", "")
            title = item.get("title", "")
            link = item.get("link_or_path", "")
            page = item.get("page")

            if source_type == "rag":
                loc = link or title or "로컬 문서"
                if page:
                    lines.append(f"[{idx}] {loc} (p.{page})")
                else:
                    lines.append(f"[{idx}] {loc}")
            elif source_type == "api":
                label = title or "외부 API 결과"
                if link:
                    lines.append(f"[{idx}] {label} - {link}")
                else:
                    lines.append(f"[{idx}] {label}")
            else:
                label = title or "생성 참조"
                if link:
                    lines.append(f"[{idx}] {label} - {link}")
                else:
                    lines.append(f"[{idx}] {label}")

        return "\n".join(lines)

    def _summarize_evidence(
        self,
        search_results: List[SearchResult],
        api_lookup_data: Dict[str, Any],
    ) -> str:
        # EvidenceEnvelope가 있으면 우선 사용
        evidence = api_lookup_data.get("evidence")
        if isinstance(evidence, dict) and evidence.get("items"):
            lines = ["근거 요약"]
            rag_items = [i for i in evidence["items"] if i.get("source_type") == "rag"]
            api_items = [i for i in evidence["items"] if i.get("source_type") == "api"]
            if rag_items:
                titles = ", ".join(i["title"] for i in rag_items[:3] if i.get("title"))
                lines.append(
                    f"- 로컬 문서 {len(rag_items)}건을 참고했습니다."
                    + (f" 주요 문서: {titles}" if titles else "")
                )
            if api_items:
                titles = ", ".join(i["title"] for i in api_items[:3] if i.get("title"))
                lines.append(
                    f"- 외부 민원분석 API에서 유사 사례 {len(api_items)}건을 확인했습니다."
                    + (f" 대표 사례: {titles}" if titles else "")
                )
            if len(lines) == 1:
                lines.append(
                    "- 내부 검색 결과를 충분히 확보하지 못해 일반 행정 응대 원칙 기준으로 작성했습니다."
                )
            return "\n".join(lines)

        # Legacy 포매터 (EvidenceItem 없을 때)
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

        # EvidenceEnvelope가 있으면 단일 포매터로 우선 처리
        rag_evidence = rag_data.get("evidence")
        api_evidence = api_data.get("evidence")

        if rag_evidence and isinstance(rag_evidence, dict) and rag_evidence.get("items"):
            for item in rag_evidence["items"][:5]:
                source_type = item.get("source_type", "rag")
                if source_type == "rag":
                    link = item.get("link_or_path", "")
                    page = item.get("page")
                    loc = link or item.get("title", "") or "로컬 문서"
                    if page:
                        lines.append(f"[{cursor}] {loc} (p.{page})")
                    else:
                        lines.append(f"[{cursor}] {loc}")
                    cursor += 1
        else:
            # Legacy RAG 포매터
            for item in rag_data.get("results", [])[:5]:
                lines.append(self._rag_source_line(cursor, item))
                cursor += 1

        if api_evidence and isinstance(api_evidence, dict) and api_evidence.get("items"):
            for item in api_evidence["items"][:5]:
                title = item.get("title", "") or "외부 API 결과"
                link = item.get("link_or_path", "")
                if link:
                    lines.append(f"[{cursor}] {title} - {link}")
                else:
                    lines.append(f"[{cursor}] {title}")
                cursor += 1
        else:
            # Legacy API 포매터
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
            working_query = query.strip()
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
            working_query = query.strip()
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

    def _init_graph_with_async_checkpointer(self, checkpointer: object) -> None:
        """lifespan에서 AsyncSqliteSaver가 준비된 후 graph를 재구성한다."""
        self._init_graph(checkpointer=checkpointer)

    def _init_graph(self, checkpointer: Optional[object] = None) -> None:
        """LangGraph StateGraph를 초기화한다.

        운영 환경에서는 LLMPlannerAdapter(vLLM OpenAI-compatible endpoint)를 사용한다.
        SKIP_MODEL_LOAD=true 환경(CI/테스트)에서는 RegexPlannerAdapter가 CI fallback으로 동작한다.
        RegistryExecutorAdapter는 기존 tool_registry를 재사용한다.

        Parameters
        ----------
        checkpointer : optional
            외부에서 주입할 LangGraph checkpointer.
            None이면 SqliteSaver(동기 sqlite3)를 시도하고,
            import 실패 시 MemorySaver로 fallback한다.
            SqliteSaver DB 경로는 SessionStore DB와 같은 디렉터리에
            ``langgraph_checkpoints.db``로 생성된다 (관심사 분리).
        """
        try:
            from src.inference.graph.builder import build_govon_graph
            from src.inference.graph.executor_adapter import RegistryExecutorAdapter
            from src.inference.graph.planner_adapter import LLMPlannerAdapter
        except ImportError as exc:
            logger.warning(f"LangGraph graph 초기화 실패 (import 오류): {exc}")
            return

        tool_registry = self._build_tool_registry()

        if SKIP_MODEL_LOAD:
            # CI/테스트 환경: LLM이 없으므로 RegexPlannerAdapter를 CI fallback으로 사용
            from src.inference.graph.planner_adapter import RegexPlannerAdapter

            planner = RegexPlannerAdapter(registry=tool_registry)
        else:
            # 운영 환경: vLLM OpenAI-compatible endpoint를 LLMPlannerAdapter로 연결
            # NOTE: ChatOpenAI는 lazy connection이므로 생성 시점에 vLLM이 미시작이어도 안전하다.
            # 실제 LLM 호출은 graph invoke 시점에 발생하며,
            # 그때는 lifespan에서 vLLM이 이미 시작된 상태다.
            from langchain_openai import ChatOpenAI

            planner_base_url = os.getenv(
                "LANGGRAPH_MODEL_BASE_URL",
                f"http://127.0.0.1:{runtime_config.port}/v1",
            )
            planner_api_key = os.getenv("LANGGRAPH_MODEL_API_KEY", "EMPTY")
            planner_model = os.getenv("LANGGRAPH_PLANNER_MODEL", runtime_config.model.model_path)
            llm = ChatOpenAI(
                base_url=planner_base_url,
                api_key=planner_api_key,
                model=planner_model,
                temperature=0.0,
            )
            planner = LLMPlannerAdapter(llm=llm, registry=tool_registry)
        executor = RegistryExecutorAdapter(
            tool_registry=tool_registry,
            session_store=self.session_store,
        )

        # checkpointer가 외부에서 주입되지 않으면 SqliteSaver를 시도한다.
        # SqliteSaver는 프로세스 재시작 후에도 interrupt 상태를 복원하므로
        # MemorySaver와 달리 재시작-안전(restart-safe)하다.
        if checkpointer is None:
            checkpointer, conn = _build_sync_sqlite_checkpointer(self.session_store.db_path)
            # 이전 동기 connection이 있으면 닫아 leak을 방지한다.
            if self._sync_checkpointer_conn is not None:
                try:
                    self._sync_checkpointer_conn.close()
                except Exception:
                    pass
            self._sync_checkpointer_conn = conn

        self.graph = build_govon_graph(
            planner_adapter=planner,
            executor_adapter=executor,
            session_store=self.session_store,
            checkpointer=checkpointer,
        )
        logger.info("LangGraph graph 초기화 완료")


def _build_sync_sqlite_checkpointer(
    session_db_path: str,
) -> tuple:
    """SqliteSaver(동기) 또는 MemorySaver(fallback)를 반환한다.

    LangGraph checkpointer용 SQLite DB는 SessionStore의 sessions.sqlite3와
    같은 디렉터리에 별도 파일 ``langgraph_checkpoints.db``로 생성한다.
    두 DB를 분리함으로써 관심사(세션 메타 vs. graph 체크포인트)를 명확히 구분한다.

    SqliteSaver는 프로세스 재시작 후에도 interrupt 상태를 SQLite에서 복원하므로
    MemorySaver와 달리 재시작-안전(restart-safe)하다.

    Parameters
    ----------
    session_db_path : str
        SessionStore가 사용 중인 sessions.sqlite3 파일 경로.
        이 경로의 부모 디렉터리에 langgraph_checkpoints.db를 생성한다.

    Returns
    -------
    tuple[SqliteSaver | MemorySaver, sqlite3.Connection | None]
        (checkpointer, conn) 튜플.
        SqliteSaver 사용 시 conn은 열린 sqlite3.Connection이며,
        호출자가 적절한 시점에 close해야 한다.
        MemorySaver fallback 시 conn은 None이다.
    """
    cp_db_path = str(Path(session_db_path).parent / "langgraph_checkpoints.db")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = __import__("sqlite3").connect(cp_db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        logger.info(f"LangGraph checkpointer: SqliteSaver ({cp_db_path})")
        return saver, conn
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite 미설치 — MemorySaver로 fallback합니다. "
            "프로세스 재시작 시 interrupt 상태가 소멸됩니다."
        )
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver(), None


manager = vLLMEngineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: 모델/인덱스 초기화 및 AsyncSqliteSaver 업그레이드.

    startup 단계에서 AsyncSqliteSaver가 사용 가능하면 graph를 재구성한다.
    AsyncSqliteSaver는 async 컨텍스트 매니저로 관리하며, shutdown 시 정리한다.
    AsyncSqliteSaver import 실패 시 _init_graph에서 이미 설정된
    SqliteSaver(또는 MemorySaver fallback)를 그대로 유지한다.
    """
    await manager.initialize()

    # AsyncSqliteSaver로 graph 재구성 시도 (더 높은 async 성능)
    async_cp_db = str(Path(manager.session_store.db_path).parent / "langgraph_checkpoints.db")
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(async_cp_db) as async_saver:
            # 동기 SqliteSaver가 보유하던 connection을 닫아 leak을 방지한다.
            if manager._sync_checkpointer_conn is not None:
                try:
                    manager._sync_checkpointer_conn.close()
                except Exception:
                    pass
                manager._sync_checkpointer_conn = None
            manager._checkpointer_ctx = async_saver
            manager._init_graph_with_async_checkpointer(async_saver)
            logger.info(f"LangGraph checkpointer: AsyncSqliteSaver ({async_cp_db})")
            yield
        manager._checkpointer_ctx = None
    except ImportError:
        logger.info("AsyncSqliteSaver 미설치 — SqliteSaver(동기) 또는 MemorySaver로 실행합니다.")
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
        "local_documents": {
            "enabled": bool(runtime_config.paths.local_docs_root),
            "root_dir": runtime_config.paths.local_docs_root or None,
            "last_sync": manager.local_document_sync_status,
        },
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


@app.post("/v2/agent/stream")
async def v2_agent_stream(
    request: AgentRunRequest,
    _: None = Depends(verify_api_key),
):
    """LangGraph 기반 agent SSE 스트리밍 실행.

    graph.stream()을 asyncio.to_thread()로 감싸 노드별 완료 이벤트를 SSE로 전송한다.

    이벤트 형식 (각 줄: ``data: <JSON>\\n\\n``):
      - 노드 진행: ``{"node": "<name>", "status": "completed", ...}``
      - approval_wait 도달:
        ``{"node": "approval_wait", "status": "awaiting_approval",
           "approval_request": {...}, "thread_id": "..."}``
      - 오류: ``{"node": "error", "status": "error", "error": "..."}``

    승인 흐름:
    - 클라이언트는 ``awaiting_approval`` 이벤트 수신 후 스트림이 종료됨을 인지하고
      ``/v2/agent/approve``로 승인/거절을 전달한다.
    """
    if not manager.graph:

        async def _no_graph():
            yield 'data: {"node": "error", "status": "error", "error": "LangGraph graph가 초기화되지 않았습니다."}\n\n'

        return StreamingResponse(_no_graph(), media_type="text/event-stream")

    from langchain_core.messages import HumanMessage

    thread_id = request.session_id or str(uuid.uuid4())
    session_id = thread_id
    request_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "session_id": session_id,
        "request_id": request_id,
        "messages": [HumanMessage(content=request.query)],
    }

    async def _generate() -> AsyncGenerator[str, None]:
        event_queue: "queue.Queue[dict | None]" = queue.Queue()

        def _stream_in_thread():
            """graph.stream()을 별도 스레드에서 실행하고 결과를 queue에 넣는다."""
            try:
                for chunk in manager.graph.stream(initial_state, config, stream_mode="updates"):
                    # chunk: {node_name: state_delta}
                    for node_name, state_delta in chunk.items():
                        event: dict = {
                            "node": node_name,
                            "status": "completed",
                        }
                        # synthesis 완료 시 evidence_items와 task_type을 이벤트에 포함.
                        # 전제: stream_mode="updates"에서 state_delta는 노드의 raw return dict다.
                        # LangGraph 버전 업그레이드 시 이 구조가 변경될 수 있으므로 주의.
                        # evidence_items 스키마: EvidenceItem.to_dict() 필드를 따른다.
                        #   source_type: "rag" | "api" | "llm_generated"
                        #   title, excerpt, link_or_path, page, score, provider_meta
                        #   (웹 프론트엔드에서 직접 렌더링 시 XSS 방지를 위해 이스케이프 필요)
                        if node_name == "synthesis" and isinstance(state_delta, dict):
                            if state_delta.get("final_text"):
                                event["final_text"] = state_delta["final_text"]
                            if state_delta.get("evidence_items"):
                                event["evidence_items"] = state_delta["evidence_items"]
                            if state_delta.get("task_type"):
                                event["task_type"] = state_delta["task_type"]
                        # approval_wait가 interrupt()를 호출하면 stream이 끝나기 전에
                        # 이 이벤트가 마지막으로 전송된다.
                        if node_name == "approval_wait":
                            # interrupt 상태 확인
                            try:
                                graph_state = manager.graph.get_state(config)
                                if graph_state.next:
                                    approval_value = None
                                    if graph_state.tasks and graph_state.tasks[0].interrupts:
                                        approval_value = graph_state.tasks[0].interrupts[0].value
                                    event = {
                                        "node": "approval_wait",
                                        "status": "awaiting_approval",
                                        "approval_request": approval_value,
                                        "thread_id": thread_id,
                                        "session_id": session_id,
                                    }
                            except Exception as exc:
                                logger.warning(f"[v2/agent/stream] get_state 실패: {exc}")
                                # get_state 실패해도 approval_wait 이벤트는 전송
                                event["status"] = "awaiting_approval"
                                event["approval_request"] = {
                                    "prompt": "승인 정보를 불러올 수 없습니다. /v2/agent/approve로 진행하세요."
                                }
                        event_queue.put(event)
            except Exception as exc:
                logger.error(f"[v2/agent/stream] 스레드 예외: {exc}")
                event_queue.put({"node": "error", "status": "error", "error": str(exc)})
            finally:
                event_queue.put(None)  # sentinel

        thread = threading.Thread(target=_stream_in_thread, daemon=True)
        thread.start()

        while True:
            try:
                event = await asyncio.to_thread(event_queue.get, True, 0.5)
            except queue.Empty:
                # timeout — check if thread is still alive
                if not thread.is_alive() and event_queue.empty():
                    break
                continue

            if event is None:
                # sentinel: stream complete
                break

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # Stop streaming after awaiting_approval (client must call /v2/agent/approve)
            if event.get("status") == "awaiting_approval" or event.get("node") == "error":
                break

        thread.join(timeout=1.0)
        if thread.is_alive():
            logger.warning("[v2/agent/stream] 스트리밍 스레드가 1초 내에 종료되지 않았습니다.")

    return StreamingResponse(_generate(), media_type="text/event-stream")


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

    Session Resume Contract
    -----------------------
    동일 session_id로 재요청하는 경우 다음 규칙을 따른다:

    1. **interrupt 대기 중**: graph가 approval_wait에서 interrupt 상태이면
       현재 checkpoint에서 resume하지 않고 새 메시지를 *추가하여* 이어서 실행한다.
       (재요청은 새 graph_run으로 처리한다.)
       승인/거절은 반드시 `/v2/agent/approve`를 통해 처리해야 한다.

    2. **완료된 graph**: graph가 END에 도달한 상태(state.next == [])이면
       동일 thread_id에 새 graph_run을 시작한다. LangGraph checkpointer가
       동일 thread_id에서 이전 상태를 누적하므로 대화 히스토리가 보존된다.

    3. **프로세스 재시작 후**: SqliteSaver 사용 시 DB에서 checkpoint가 복원되므로
       interrupt 상태가 유지된다. 클라이언트는 기존 thread_id로 `/v2/agent/approve`
       를 다시 호출하면 중단된 지점에서 resume할 수 있다.

    Note: session_id == thread_id. 두 값은 항상 동일하게 유지된다.
    """
    if not manager.graph:
        raise HTTPException(status_code=503, detail="LangGraph graph가 초기화되지 않았습니다.")

    from langchain_core.messages import HumanMessage

    thread_id = request.session_id or str(uuid.uuid4())
    session_id = thread_id  # thread_id를 session_id로 확정 (session_id == thread_id 불변)
    request_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "session_id": session_id,
        "request_id": request_id,
        "messages": [HumanMessage(content=request.query)],
    }

    try:
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
                "session_id": session_id,
                "graph_run_id": request_id,
                "approval_request": approval_value,
            }

        # interrupt 없이 완료된 경우 (rejected 또는 오류)
        final_state = graph_state.values
        return {
            "status": "completed",
            "thread_id": thread_id,
            "session_id": session_id,
            "graph_run_id": request_id,
            "text": final_state.get("final_text", ""),
            "evidence_items": final_state.get("evidence_items", []),
            "task_type": final_state.get("task_type", ""),
        }
    except Exception as exc:
        logger.error(f"[v2/agent/run] 예외 발생: {exc}")
        # graph_run을 "error" status로 기록 시도
        try:
            if manager.session_store:
                session = manager.session_store.get_or_create(session_id)
                session.add_graph_run(
                    request_id=request_id,
                    plan_summary=f"[error] {exc}",
                    approval_status="",
                    executed_capabilities=[],
                    status="error",
                    total_latency_ms=0.0,
                )
        except Exception as persist_exc:
            logger.warning(f"[v2/agent/run] error persist 실패: {persist_exc}")
        return {
            "status": "error",
            "thread_id": thread_id,
            "session_id": session_id,
            "graph_run_id": request_id,
            "error": str(exc),
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

    try:
        # resume도 동일한 sync invoke 경로로 처리한다.
        result = await asyncio.to_thread(
            manager.graph.invoke,
            Command(resume={"approved": approved}),
            config,
        )

        # 거절이면 "rejected", 승인 완료면 "completed"
        approval_status = result.get("approval_status", "")
        if not approved:
            response_status = "rejected"
        else:
            response_status = "completed"

        return {
            "status": response_status,
            "thread_id": thread_id,
            "session_id": result.get("session_id", ""),
            "graph_run_id": result.get("request_id", ""),
            "text": result.get("final_text", ""),
            "evidence_items": result.get("evidence_items", []),
            "task_type": result.get("task_type", ""),
            "tool_results": result.get("tool_results", {}),
            "approval_status": approval_status,
        }
    except Exception as exc:
        logger.error(f"[v2/agent/approve] 예외 발생: {exc}")
        # graph_run을 "error" status로 기록 시도
        session_id = ""
        request_id = ""
        try:
            if manager.session_store:
                graph_state = await asyncio.to_thread(manager.graph.get_state, config)
                state_values = graph_state.values if graph_state else {}
                session_id = state_values.get("session_id", "")
                request_id = state_values.get("request_id", "")
                if session_id:
                    session = manager.session_store.get_or_create(session_id)
                    session.add_graph_run(
                        request_id=request_id,
                        plan_summary=f"[error] {exc}",
                        approval_status="",
                        executed_capabilities=[],
                        status="error",
                        total_latency_ms=0.0,
                    )
        except Exception as persist_exc:
            logger.warning(f"[v2/agent/approve] error persist 실패: {persist_exc}")
        return {
            "status": "error",
            "thread_id": thread_id,
            "session_id": session_id,
            "graph_run_id": request_id,
            "error": str(exc),
        }


@app.post("/v2/agent/cancel")
async def v2_agent_cancel(
    thread_id: str,
    _: None = Depends(verify_api_key),
):
    """interrupt 대기 중인 graph를 강제 취소한다.

    interrupt 상태에서 거절 처리(Command(resume={"approved": False}))를 수행하되,
    state에 interrupt_reason="user_cancel"을 전달하여
    persist 노드가 graph_run status를 "interrupted"로 기록하게 한다.

    Parameters
    ----------
    thread_id : str
        `/v2/agent/run`에서 반환된 thread_id.
    """
    if not manager.graph:
        raise HTTPException(status_code=503, detail="LangGraph graph가 초기화되지 않았습니다.")

    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # interrupt 상태 확인
        graph_state = await asyncio.to_thread(manager.graph.get_state, config)
        if not graph_state or not graph_state.next:
            raise HTTPException(
                status_code=409,
                detail="해당 thread는 현재 interrupt 대기 상태가 아닙니다.",
            )

        session_id = graph_state.values.get("session_id", "")

        # 강제 거절 + interrupt_reason 전달로 resume
        result = await asyncio.to_thread(
            manager.graph.invoke,
            Command(resume={"approved": False, "cancel": True}),
            config,
        )

        # persist 노드에서 "interrupted" 기록을 위해 state update
        # (approval_wait_node가 cancel 신호를 interrupt_reason으로 변환)
        return {
            "status": "cancelled",
            "thread_id": thread_id,
            "session_id": session_id,
            "graph_run_id": result.get("request_id", ""),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[v2/agent/cancel] 예외 발생: {exc}")
        return {
            "status": "error",
            "thread_id": thread_id,
            "error": str(exc),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, **runtime_config.to_uvicorn_kwargs())
