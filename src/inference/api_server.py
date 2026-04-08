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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

try:
    import httpx as _httpx
except ImportError:
    _httpx = None


from .adapter_registry import AdapterRegistry
from .agent_loop import AgentLoop, AgentTrace
from .agent_manager import AgentManager
from .feature_flags import FeatureFlags
from .runtime_config import RuntimeConfig
from .schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentTraceSchema,
    GenerateCivilResponseRequest,
    GenerateCivilResponseResponse,
    GenerateRequest,
    GenerateResponse,
    ToolResultSchema,
)
from .session_context import SessionContext, SessionStore
from .tool_router import ToolType, tool_name

SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "false").lower() in ("true", "1", "yes")


async def _noop_tool(query: str, context: dict, session: Any) -> dict:
    """build_all_tools fallback용 no-op tool."""
    return {"success": False, "error": "tool이 초기화되지 않았습니다"}


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


_ALLOW_NO_AUTH = os.getenv("ALLOW_NO_AUTH", "false").lower() in ("true", "1")


async def verify_api_key(api_key: str = Security(_api_key_header)):
    if _API_KEY is None:
        if _ALLOW_NO_AUTH:
            return
        raise HTTPException(
            status_code=401, detail="API_KEY가 설정되지 않았습니다. 서버 관리자에게 문의하세요."
        )
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
class SamplingParams:
    """vLLM HTTP API용 샘플링 파라미터. vLLM 직접 import 없이 동작."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[list] = None
    repetition_penalty: float = 1.0


@dataclass
class PreparedGeneration:
    prompt: str
    sampling_params: SamplingParams


class _VLLMOutputItem:
    """vLLM HTTP 응답의 단일 choice를 기존 인터페이스로 래핑."""

    def __init__(self, text: str, finish_reason: str, token_ids: list):
        self.text = text
        self.finish_reason = finish_reason
        self.token_ids = token_ids


class _StreamChunk:
    """SSE 스트리밍 청크를 /v1/stream 엔드포인트 인터페이스로 래핑.

    /v1/stream 에서 ``request_output.outputs[0].text``,
    ``request_output.finished``에 접근하므로 동일한 속성을 제공한다.
    """

    def __init__(self, text: str, finished: bool) -> None:
        self.outputs = [type("Output", (), {"text": text})()]
        self.finished = finished


class _VLLMHttpResult:
    """vLLM HTTP 응답을 기존 AsyncLLM 결과 인터페이스로 래핑.

    기존 코드가 ``output.outputs[0].text``, ``output.prompt_token_ids`` 등에
    접근하므로 동일한 속성을 제공한다.
    """

    def __init__(self, data: dict):
        self._data = data
        choices = data.get("choices", [])
        usage = data.get("usage", {})
        self.outputs = []
        for choice in choices:
            msg = choice.get("message", {})
            text = msg.get("content", "")
            finish = choice.get("finish_reason", "stop")
            self.outputs.append(
                _VLLMOutputItem(
                    text=text,
                    finish_reason=finish,
                    token_ids=list(range(usage.get("completion_tokens", 0))),
                )
            )
        self.prompt_token_ids = list(range(usage.get("prompt_tokens", 0)))


def _extract_approval_request(graph_state: Any) -> Any:
    """LangGraph interrupt state에서 approval payload를 추출한다."""
    if not graph_state or not getattr(graph_state, "tasks", None):
        return None
    task = graph_state.tasks[0]
    if not getattr(task, "interrupts", None):
        return None
    return task.interrupts[0].value


class vLLMEngineManager:
    """GovOn Shell MVP용 로컬 런타임 매니저.

    vLLM은 별도 프로세스(entrypoint.sh)에서 OpenAI-compatible 서버로 실행된다.
    이 클래스는 httpx로 vLLM HTTP API를 호출한다.
    """

    def __init__(self):
        self._vllm_base_url = f"http://localhost:{os.getenv('VLLM_PORT', '8000')}"
        self._http_client: Optional[Any] = None
        self.feature_flags = FeatureFlags.from_env()
        self.session_store = SessionStore()
        self.agent_manager = AgentManager(AGENTS_DIR)
        self.agent_loop: Optional[AgentLoop] = None
        self.graph = None  # LangGraph CompiledGraph (v2 엔드포인트용)
        self._checkpointer_ctx = None  # AsyncSqliteSaver 컨텍스트 매니저 (lifespan에서 관리)
        self._sync_checkpointer_conn = None  # SqliteSaver용 sqlite3 connection (leak 방지)
        self._init_agent_loop()
        # _init_graph()는 lifespan()에서 호출 — 모듈 로드 시점 실행 방지

    async def initialize(self):
        if SKIP_MODEL_LOAD:
            logger.info("SKIP_MODEL_LOAD=true: 모델 및 인덱스 로딩을 건너뜁니다.")
            return

        # vLLM 서버는 entrypoint.sh에서 이미 기동됨 — health check만 수행
        logger.info(f"vLLM 서버 연결 확인: {self._vllm_base_url}")
        if _httpx is None:
            raise RuntimeError("httpx가 설치되어 있지 않습니다. pip install httpx")

        self._http_client = _httpx.AsyncClient(
            base_url=self._vllm_base_url,
            timeout=_httpx.Timeout(300.0, connect=30.0),
        )

        # vLLM 서버 health check (entrypoint.sh에서 이미 확인했지만 이중 검증)
        for attempt in range(10):
            try:
                resp = await self._http_client.get("/health")
                if resp.status_code == 200:
                    logger.info("vLLM 서버 연결 성공")
                    return
            except Exception:
                pass
            logger.debug(f"vLLM 서버 대기 중... ({attempt + 1}/10)")
            await asyncio.sleep(3)

        raise RuntimeError(f"vLLM 서버에 연결할 수 없습니다: {self._vllm_base_url}")

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
        # <thought>...</thought> (구형) 및 <think>...</think> (EXAONE-4.0 추론 모드) 모두 제거
        text = re.sub(r"<thought>.*?</thought>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return text.strip()

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
            # source_type에 따라 기본 label만 다르고 포맷은 동일
            label = (title or "외부 API 결과") if source_type == "api" else (title or "생성 참조")
            lines.append(f"[{idx}] {label} - {link}" if link else f"[{idx}] {label}")

        return "\n".join(lines)

    def _summarize_evidence(
        self,
        api_lookup_data: Dict[str, Any],
    ) -> str:
        # EvidenceEnvelope가 있으면 우선 사용
        evidence = api_lookup_data.get("evidence")
        if isinstance(evidence, dict) and evidence.get("items"):
            lines = ["근거 요약"]
            api_items = [i for i in evidence["items"] if i.get("source_type") == "api"]
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
        api_data: Dict[str, Any],
    ) -> str:
        _, previous_answer = self._latest_prior_turns(session, current_query)
        lines = ["근거/출처"]
        cursor = 1

        # EvidenceEnvelope가 있으면 단일 포매터로 우선 처리
        api_evidence = api_data.get("evidence")

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
        gen_defaults = runtime_config.generation

        safe_message = self._escape_special_tokens(self._extract_query(request.prompt))
        user_content = f"다음 민원에 대한 답변을 작성해 주세요.\n\n{safe_message}"
        prompt = self._build_persona_prompt("draft_response", user_content)

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or gen_defaults.stop_sequences,
            repetition_penalty=gen_defaults.repetition_penalty,
        )

        return PreparedGeneration(
            prompt=prompt,
            sampling_params=sampling_params,
        )

    async def _prepare_draft_only(
        self,
        request: GenerateCivilResponseRequest,
        flags: Optional[FeatureFlags] = None,
    ) -> PreparedGeneration:
        """LoRA 초안 생성용: 쿼리만으로 프롬프트 생성.

        사용자 쿼리를 persona 프롬프트로 감싸서 반환한다.
        """
        gen_defaults = runtime_config.generation

        safe_message = self._escape_special_tokens(self._extract_query(request.prompt))
        # 학습 데이터 형식: user = instruction + "\n\n" + input
        user_content = f"다음 민원에 대한 답변을 작성해 주세요.\n\n{safe_message}"
        prompt = self._build_persona_prompt("draft_response", user_content)

        sampling_params = SamplingParams(
            temperature=(
                request.temperature if request.temperature is not None else gen_defaults.temperature
            ),
            top_p=request.top_p if request.top_p is not None else gen_defaults.top_p,
            max_tokens=request.max_tokens or gen_defaults.max_tokens,
            stop=request.stop or gen_defaults.stop_sequences,
            repetition_penalty=gen_defaults.repetition_penalty,
        )

        return PreparedGeneration(
            prompt=prompt,
            sampling_params=sampling_params,
        )

    async def synthesize_final(
        self,
        draft_text: str,
        evidence_items: list,
        query: str,
        adapter_name: str = "public_admin",
    ) -> str:
        """초안 + 도구 결과를 베이스 모델로 통합하여 최종 답변 생성.

        LoRA 어댑터는 학습 형식(질문→답변)에 특화되어 있어
        초안+근거 통합 같은 범용 태스크에는 베이스 모델이 적합하다.
        """
        safe_query = self._escape_special_tokens(query[:400])
        safe_draft = self._escape_special_tokens(draft_text[:800])

        # 근거 텍스트 조립
        evidence_text = ""
        for item in evidence_items[:5]:
            source_type = item.get("source_type", "")
            title = item.get("title", "")
            excerpt = item.get("excerpt", "")[:200]
            label = "[외부]" if source_type == "api" else "[생성]"
            if title or excerpt:
                evidence_text += f"- {label} {title}: {excerpt}\n"

        if not evidence_text.strip():
            evidence_text = "(검색 근거 없음)"

        # 베이스 모델 범용 합성 프롬프트
        synthesis_prompt = (
            "[|system|]당신은 민원 답변을 보강하는 전문가입니다. "
            "초안과 참고 근거를 결합하여 정확하고 공감적인 최종 답변을 작성하세요. "
            "법적 근거가 있으면 인용하고, 절차와 조치사항을 명확히 포함하세요."
            "[|endofturn|]\n"
            "[|user|]다음 초안과 근거를 결합하여 최종 민원 답변을 작성하세요.\n\n"
            f"[민원 질의]\n{safe_query}\n\n"
            f"[초안]\n{safe_draft}\n\n"
            f"[참고 근거]\n{evidence_text}"
            "[|endofturn|]\n[|assistant|]"
        )

        # 베이스 모델 사용 (LoRA 없음) — 합성은 범용 태스크
        sampling_params = SamplingParams(
            max_tokens=768,
            temperature=0.6,
            top_p=0.9,
            stop=["[|endofturn|]"],
        )

        import uuid as _uuid

        request_id = str(_uuid.uuid4())

        try:
            output = await self._run_engine(
                synthesis_prompt, sampling_params, request_id, lora_request=None
            )
        except Exception as exc:
            logger.warning(f"[synthesize_final] 합성 실패: {exc}")
            return draft_text

        if output is None or not output.outputs:
            return draft_text

        return self._strip_thought_blocks(output.outputs[0].text)

    async def _run_engine(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request=None,
    ):
        """vLLM OpenAI-compatible HTTP API를 통해 텍스트를 생성한다."""
        if self._http_client is None:
            return None

        # EXAONE chat template 형식의 prompt를 messages로 변환
        messages = self._prompt_to_messages(prompt)

        body: Dict[str, Any] = {
            "model": MODEL_PATH,
            "messages": messages,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "stream": False,
        }
        if sampling_params.top_p is not None and sampling_params.top_p < 1.0:
            body["top_p"] = sampling_params.top_p
        if sampling_params.stop:
            body["stop"] = list(sampling_params.stop)
        if sampling_params.repetition_penalty and sampling_params.repetition_penalty != 1.0:
            body["repetition_penalty"] = sampling_params.repetition_penalty

        # LoRA 어댑터 지정
        if lora_request is not None:
            body["model"] = lora_request.lora_name

        try:
            resp = await self._http_client.post("/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
        except _httpx.TimeoutException as exc:
            logger.error(f"vLLM HTTP 타임아웃: {exc}")
            return None
        except _httpx.HTTPStatusError as exc:
            logger.error(f"vLLM HTTP {exc.response.status_code}: {exc}")
            return None
        except Exception as exc:
            logger.error(f"vLLM HTTP 호출 실패: {exc}")
            return None

        # OpenAI 응답을 기존 인터페이스와 호환되는 객체로 래핑
        return _VLLMHttpResult(data)

    @staticmethod
    def _prompt_to_messages(prompt: str) -> list:
        """EXAONE chat template 형식 프롬프트를 OpenAI messages로 변환."""
        messages = []
        # [|system|]...[|endofturn|], [|user|]...[|endofturn|], [|assistant|]... 파싱
        import re as _re

        # 이스케이프된 토큰은 _escape_special_tokens()에서 이미 처리되어
        # 이 시점에는 원본 형태 [|role|]로 전달된다.
        parts = _re.split(r"\[\|(\w+)\|\]", prompt)
        role = None
        for part in parts:
            if part in ("system", "user", "assistant"):
                role = part
            elif role and part.strip():
                content = part.replace("[|endofturn|]", "").strip()
                if content:
                    messages.append({"role": role, "content": content})
                role = None

        if not messages:
            messages = [{"role": "user", "content": prompt}]
        return messages

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str,
        flags: Optional[FeatureFlags] = None,
    ) -> Any:
        return await self.generate_civil_response(request, request_id, flags)

    async def generate_civil_response(
        self,
        request: GenerateCivilResponseRequest,
        request_id: str,
        flags: Optional[FeatureFlags] = None,
        external_cases: Optional[List[dict]] = None,
        lora_request=None,
    ) -> Any:
        prepared = await self._prepare_civil_response_generation(request, flags, external_cases)
        return await self._run_engine(
            prepared.prompt, prepared.sampling_params, request_id, lora_request=lora_request
        )

    async def generate_stream(
        self,
        request: GenerateRequest,
        request_id: str,
        flags: Optional[FeatureFlags] = None,
    ) -> AsyncGenerator:
        """SSE 스트리밍 응답을 파싱하여 _StreamChunk를 yield하는 async generator를 반환.

        기존 코드가 ``async for request_output in results_stream:``으로 소비하므로
        context manager가 아닌 async generator로 반환해야 한다.
        """
        prepared = await self._prepare_civil_response_generation(request, flags)
        if self._http_client is None:
            raise RuntimeError("vLLM 서버에 연결되지 않았습니다.")

        messages = self._prompt_to_messages(prepared.prompt)
        body = {
            "model": MODEL_PATH,
            "messages": messages,
            "max_tokens": prepared.sampling_params.max_tokens,
            "temperature": prepared.sampling_params.temperature,
            "stream": True,
        }
        if prepared.sampling_params.stop:
            body["stop"] = list(prepared.sampling_params.stop)

        return self._stream_vllm_response(body)

    async def _stream_vllm_response(self, body: dict) -> AsyncGenerator:
        """vLLM SSE 스트리밍을 파싱하여 _StreamChunk를 yield하는 async generator.

        httpx .stream()은 context manager이므로 직접 ``async for``로 소비할 수 없다.
        이 메서드가 내부에서 context manager를 관리하고 파싱된 청크를 yield한다.
        """
        accumulated_text = ""
        async with self._http_client.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                logger.error(f"vLLM 스트리밍 HTTP {resp.status_code}")
                return
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        accumulated_text += content
                    finish = chunk.get("choices", [{}])[0].get("finish_reason")
                    yield _StreamChunk(text=accumulated_text, finished=finish is not None)
                except json.JSONDecodeError:
                    continue

    def _init_agent_loop(self) -> None:
        from src.inference.actions.data_go_kr import MinwonAnalysisAction

        engine_ref = self
        minwon_action = MinwonAnalysisAction()

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

        async def _draft_response_tool(
            query: str,
            context: dict,
            session: SessionContext,
        ) -> dict:
            working_query = engine_ref._build_working_query(query, session)

            # LoRA-First: 쿼리만으로 초안 생성
            adapter_name = context.get("adapter") if context else None
            if not adapter_name:
                adapter_name = "public_admin"
            _adapter_reg = AdapterRegistry.get_instance()
            lora_req = _adapter_reg.get_lora_request(adapter_name)

            gen_request = GenerateCivilResponseRequest(
                prompt=working_query,
                max_tokens=2048,
                temperature=0.7,
            )
            request_id = str(uuid.uuid4())
            prepared = await engine_ref._prepare_draft_only(gen_request)
            final_output = await engine_ref._run_engine(
                prepared.prompt, prepared.sampling_params, request_id, lora_request=lora_req
            )

            if final_output is None or not final_output.outputs:
                return {
                    "text": "",
                    "draft_text": "",
                    "success": False,
                    "error": "민원 답변 초안 생성 실패",
                    "results": [],
                    "context_text": "",
                }

            draft_text = engine_ref._strip_thought_blocks(final_output.outputs[0].text)

            return {
                "text": draft_text,
                "draft_text": draft_text,
                "success": True,
                "results": [],
                "context_text": draft_text,
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
            }

        tool_registry = {
            ToolType.API_LOOKUP: _api_lookup_tool,
            "draft_response": _draft_response_tool,
        }
        self.agent_loop = AgentLoop(tool_registry=tool_registry)

    def _build_langgraph_tools(self) -> list:
        """LangGraph ToolNode용 도구 목록을 생성한다.

        build_all_tools()를 사용하여 StructuredTool 목록을 반환한다.
        AgentLoop의 tool_registry에서 기존 closure를 추출하여 전달한다.
        """
        from src.inference.graph.tools import build_all_tools

        if self.agent_loop is None:
            return build_all_tools(
                api_lookup_action=self._get_api_lookup_action(),
            )

        # AgentLoop의 tool_registry에서 기존 closure를 추출
        raw_tools = {
            str(k.value if hasattr(k, "value") else k): v for k, v in self.agent_loop._tools.items()
        }

        return build_all_tools(
            api_lookup_action=self._get_api_lookup_action(),
            draft_response_fn=raw_tools.get("draft_response"),
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

        v4 아키텍처: ReAct + ToolNode 기반.
        LLM이 자율적으로 도구 호출을 결정하며, 정적 planner/executor를 사용하지 않는다.

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
        except ImportError as exc:
            logger.warning(f"LangGraph graph 초기화 실패 (import 오류): {exc}")
            return

        tools = self._build_langgraph_tools()

        # LLM 인스턴스 구성
        if SKIP_MODEL_LOAD:
            # CI/테스트 환경: LLM이 없으므로 graph 초기화 스킵
            logger.info("SKIP_MODEL_LOAD=true: LangGraph graph 초기화 스킵")
            return
        elif os.getenv("LANGGRAPH_MODEL_BASE_URL"):
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                base_url=os.environ["LANGGRAPH_MODEL_BASE_URL"],
                api_key=os.getenv("LANGGRAPH_MODEL_API_KEY", "EMPTY"),
                model=os.getenv("LANGGRAPH_PLANNER_MODEL", runtime_config.model.model_path),
                temperature=0.0,
                max_tokens=1024,
            )
        else:
            # 운영 환경: vLLM OpenAI-compatible endpoint 사용
            from langchain_openai import ChatOpenAI

            vllm_port = os.getenv("VLLM_PORT", "8000")
            llm = ChatOpenAI(
                base_url=f"http://localhost:{vllm_port}/v1",
                api_key="EMPTY",
                model=runtime_config.model.model_path,
                temperature=0.0,
                max_tokens=1024,
            )

        # checkpointer가 외부에서 주입되지 않으면 SqliteSaver를 시도한다.
        if checkpointer is None:
            checkpointer, conn = _build_sync_sqlite_checkpointer(self.session_store.db_path)
            if self._sync_checkpointer_conn is not None:
                try:
                    self._sync_checkpointer_conn.close()
                except Exception:
                    pass
            self._sync_checkpointer_conn = conn

        self.graph = build_govon_graph(
            llm=llm,
            tools=tools,
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
    """FastAPI lifespan: 모델/인덱스 초기화 및 graph 단일 초기화.

    AsyncSqliteSaver가 사용 가능하면 그것으로 한 번만 초기화한다.
    import 실패 시 SqliteSaver(동기) 또는 MemorySaver로 fallback한다.
    graph 이중 초기화를 방지하기 위해 _init_graph()를 한 번만 호출한다.
    shutdown 시 httpx AsyncClient를 반드시 종료하여 leak을 방지한다.
    """
    await manager.initialize()

    # API_KEY 미설정 경고 (운영 프로필에서 명시적으로 알림)
    if _API_KEY is None and runtime_config.profile.value not in ("local",):
        logger.warning("API_KEY 미설정: 운영 환경에서는 반드시 API_KEY 환경변수를 설정하세요.")

    # checkpointer 결정 후 graph를 한 번만 초기화 (이중 초기화 방지)
    async_cp_db = str(Path(manager.session_store.db_path).parent / "langgraph_checkpoints.db")
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(async_cp_db) as async_saver:
            manager._init_graph(checkpointer=async_saver)
            manager._checkpointer_ctx = async_saver
            logger.info(f"LangGraph: AsyncSqliteSaver ({async_cp_db})")
            try:
                yield
            finally:
                manager._checkpointer_ctx = None
                if manager._http_client:
                    await manager._http_client.aclose()
                    logger.info("httpx AsyncClient 종료 완료")
        return
    except ImportError:
        pass

    # fallback: SqliteSaver(동기) 또는 MemorySaver
    manager._init_graph()
    logger.info("LangGraph: SqliteSaver(동기) 또는 MemorySaver로 실행합니다.")
    try:
        yield
    finally:
        if manager._http_client:
            await manager._http_client.aclose()
            logger.info("httpx AsyncClient 종료 완료")


app = FastAPI(
    title="GovOn Local Runtime",
    description="Local FastAPI daemon for the GovOn Agentic Shell MVP.",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
if ALLOWED_ORIGINS and ALLOWED_ORIGINS[0]:
    # wildcard(*)와 allow_credentials=True는 CORS 스펙상 공존 불가.
    # wildcard가 포함된 경우 credentials를 비활성화한다.
    allow_creds = "*" not in ALLOWED_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=allow_creds,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "Accept"],
    )

if _RATE_LIMIT_AVAILABLE and limiter is not None:
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)


@app.get("/health")
async def health():
    """vLLM 연결 상태를 포함한 헬스 체크.

    내부 경로(session_store.path 등)는 보안상 노출하지 않는다.
    """
    vllm_ok = False
    if manager._http_client:
        try:
            resp = await manager._http_client.get("/health", timeout=3.0)
            vllm_ok = resp.status_code == 200
        except Exception as exc:
            logger.debug(f"vLLM health check 실패: {exc}")
    # SKIP_MODEL_LOAD 환경에서는 vLLM 없이도 healthy
    status = "healthy" if (vllm_ok or SKIP_MODEL_LOAD) else "degraded"
    return {
        "status": status,
        "profile": runtime_config.profile.value,
        "model": runtime_config.model.model_path,
        "vllm_connected": vllm_ok,
        "agents_loaded": manager.agent_manager.list_agents() if manager.agent_manager else [],
        "feature_flags": {
            "model_version": manager.feature_flags.model_version,
        },
        "session_store": {
            "driver": "sqlite",
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
    final_output = await manager.generate_civil_response(
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
    final_output = await manager.generate(request, request_id, flags)
    if final_output is None:
        raise HTTPException(status_code=500, detail="Generation failed.")

    return GenerateResponse(
        request_id=request_id,
        complaint_id=request.complaint_id,
        text=manager._strip_thought_blocks(final_output.outputs[0].text),
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.outputs[0].token_ids),
    )


@app.post("/v1/chat/completions")
@_rate_limit("30/minute")
async def chat_completions(
    request: Request,
    _: None = Depends(verify_api_key),
):
    """OpenAI-compatible /v1/chat/completions.

    vLLM HTTP API를 경유하여 텍스트를 생성한다.
    v2 ReAct graph는 ChatOpenAI가 vLLM OpenAI 서버에 직접 연결하므로
    이 엔드포인트는 v1 호환 유지용이다.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    messages: list[dict] = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=422, detail="messages must not be empty.")

    # 첫 번째 이후의 system role 메시지를 제거 (모델 제약 준수)
    filtered_messages = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "system" and i > 0:
            logger.warning(f"chat_completions: 첫 번째 이후 system role 무시 (index={i})")
            continue
        filtered_messages.append(msg)
    messages = filtered_messages

    try:
        max_tokens = int(body.get("max_tokens", 512))
        temperature = float(body.get("temperature", 0.7))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid max_tokens or temperature value.")

    if not (1 <= max_tokens <= runtime_config.max_model_len):
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be between 1 and {runtime_config.max_model_len}.",
        )
    if not (0.0 <= temperature <= 2.0):
        raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 2.0.")

    model: str = body.get("model", runtime_config.model.model_path)

    # 메시지 → 프롬프트 변환 (EXAONE chat template 형식)
    prompt_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"[|system|]{content}[|endofturn|]")
        elif role == "user":
            prompt_parts.append(f"[|user|]{content}[|endofturn|]")
        elif role == "assistant":
            prompt_parts.append(f"[|assistant|]{content}[|endofturn|]")
        else:
            logger.warning(f"chat_completions: 지원하지 않는 role 무시: {role!r}")
    prompt_parts.append("[|assistant|]")
    prompt = "\n".join(prompt_parts)

    if manager._http_client is None:
        raise HTTPException(status_code=503, detail="vLLM server not connected.")

    request_id = str(uuid.uuid4())
    logger.info(
        f"chat_completions request_id={request_id} messages={len(messages)} max_tokens={max_tokens}"
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["[|endofturn|]"],
    )

    try:
        final_output = await manager._run_engine(prompt, sampling_params, request_id)
    except Exception as exc:
        logger.error(f"chat_completions generation failed: {exc}")
        raise HTTPException(status_code=500, detail="Generation failed due to internal error.")

    if final_output is None or not final_output.outputs:
        raise HTTPException(status_code=500, detail="Generation failed.")

    output = final_output.outputs[0]
    text = manager._strip_thought_blocks(output.text)
    prompt_tokens = len(final_output.prompt_token_ids)
    completion_tokens = len(output.token_ids)
    vllm_reason = getattr(output, "finish_reason", None)
    finish_reason = "length" if vllm_reason == "length" else "stop"

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


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
    results_stream = await manager.generate_stream(
        request,
        request_id,
        flags,
    )

    async def stream_results() -> AsyncGenerator[str, None]:
        async for request_output in results_stream:
            text = request_output.outputs[0].text
            finished = request_output.finished
            if finished:
                text = manager._strip_thought_blocks(text)

            response_obj = {"request_id": request_id, "text": text, "finished": finished}
            yield f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")


def _trace_to_schema(trace: AgentTrace) -> AgentTraceSchema:
    return AgentTraceSchema(
        request_id=trace.request_id,
        session_id=trace.session_id,
        plan=trace.plan_tools,
        plan_reason=trace.plan_reason,
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

    return AgentRunResponse(
        request_id=request_id,
        session_id=session.session_id,
        text=trace.final_text,
        trace=_trace_to_schema(trace),
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
@_rate_limit("30/minute")
async def v2_agent_stream(
    request: AgentRunRequest,
    _http_request: Request,
    _: None = Depends(verify_api_key),
):
    """LangGraph 기반 agent SSE 스트리밍 실행.

    graph.astream()을 사용해 노드별 완료 이벤트를 SSE로 전송한다.

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

    # 기존 interrupt 상태가 남아있으면 거절(cancel)로 해소
    try:
        from langgraph.types import Command

        existing_state = await manager.graph.aget_state(config)
        if existing_state and existing_state.next:
            await manager.graph.ainvoke(
                Command(resume={"approved": False, "cancel": True}),
                config,
            )
    except Exception as clear_exc:
        logger.warning(f"[v2] interrupt 상태 확인/해소 실패 (무시): {type(clear_exc).__name__}")

    async def _generate() -> AsyncGenerator[str, None]:
        try:
            async for chunk in manager.graph.astream(initial_state, config, stream_mode="updates"):
                # chunk: {node_name: state_delta}
                for node_name, state_delta in chunk.items():
                    event: dict = {
                        "node": node_name,
                        "status": "completed",
                    }
                    # persist 완료 시 evidence_items를 이벤트에 포함.
                    # 전제: stream_mode="updates"에서 state_delta는 노드의 raw return dict다.
                    # evidence_items 스키마: EvidenceItem.to_dict() 필드를 따른다.
                    #   source_type: "api" | "llm_generated"
                    #   title, excerpt, link_or_path, page, score, provider_meta
                    if node_name == "persist" and isinstance(state_delta, dict):
                        if state_delta.get("final_text"):
                            event["final_text"] = state_delta["final_text"]
                        if state_delta.get("evidence_items"):
                            event["evidence_items"] = state_delta["evidence_items"]
                    # approval_wait: 명시적 노드명 또는 LangGraph interrupt() 호출 시
                    # stream_mode="updates"에서 emit되는 "__interrupt__" 청크 모두 처리
                    if node_name in ("approval_wait", "__interrupt__"):
                        try:
                            graph_state = await manager.graph.aget_state(config)
                            if graph_state.next:
                                event = {
                                    "node": "approval_wait",
                                    "status": "awaiting_approval",
                                    "approval_request": _extract_approval_request(graph_state),
                                    "thread_id": thread_id,
                                    "session_id": session_id,
                                }
                        except Exception as exc:
                            logger.warning(f"[v2/agent/stream] aget_state 실패: {exc}")
                            event["node"] = "approval_wait"
                            event["status"] = "awaiting_approval"
                            event["thread_id"] = thread_id
                            event["session_id"] = session_id
                            event["approval_request"] = {
                                "prompt": "승인 정보를 불러올 수 없습니다. /v2/agent/approve로 진행하세요."
                            }

                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                    # Stop streaming after awaiting_approval (client must call /v2/agent/approve)
                    if event.get("status") == "awaiting_approval":
                        return
        except Exception as exc:
            logger.error(f"[v2/agent/stream] 스트림 예외: {exc}")
            error_event = {"node": "error", "status": "error", "error": str(exc)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/v2/agent/run")
@_rate_limit("30/minute")
async def v2_agent_run(
    request: AgentRunRequest,
    _http_request: Request,
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

    # 기존 interrupt 상태가 남아있으면 거절(cancel)로 해소
    try:
        existing_state = await manager.graph.aget_state(config)
        if existing_state and existing_state.next:
            from langgraph.types import Command

            await manager.graph.ainvoke(
                Command(resume={"approved": False, "cancel": True}),
                config,
            )
    except Exception as clear_exc:
        logger.warning(f"[v2] interrupt 상태 확인/해소 실패 (무시): {type(clear_exc).__name__}")

    try:
        await manager.graph.ainvoke(initial_state, config)

        # interrupt 상태 확인
        graph_state = await manager.graph.aget_state(config)
        if graph_state.next:
            # interrupt 대기 중: approval_request 정보를 클라이언트에 반환
            return {
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "session_id": session_id,
                "graph_run_id": request_id,
                "approval_request": _extract_approval_request(graph_state),
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
        logger.exception(f"[v2/agent/run] 요청 처리 실패: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "thread_id": thread_id,
                "session_id": session_id,
                "graph_run_id": request_id,
                "error": "요청 처리 중 내부 오류가 발생했습니다.",
            },
        )


@app.post("/v2/agent/approve")
@_rate_limit("30/minute")
async def v2_agent_approve(
    thread_id: str,
    approved: bool,
    _http_request: Request,
    _: None = Depends(verify_api_key),
):
    """interrupt된 graph를 resume한다 (2단계: 승인/거절).

    Parameters
    ----------
    thread_id : str
        `/v2/agent/run`에서 반환된 thread_id.
    approved : bool
        True면 tool_execute로 진행, False면 graph가 END로 종료.

    TODO(M7): thread_id, approved 파라미터를 query string에서 request body로 이동하는 것이
    REST 관례에 부합하나, 기존 클라이언트 호환성을 유지하기 위해 현재 방식을 유지한다.
    클라이언트 마이그레이션 이후 body 방식으로 전환한다.
    """
    if not manager.graph:
        raise HTTPException(status_code=503, detail="LangGraph graph가 초기화되지 않았습니다.")

    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await manager.graph.ainvoke(
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
            "approval_status": approval_status,
        }
    except Exception as exc:
        logger.error(f"[v2/agent/approve] 예외 발생: {exc}")
        # graph_run을 "error" status로 기록 시도
        session_id = ""
        request_id = ""
        try:
            if manager.session_store:
                graph_state = await manager.graph.aget_state(config)
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
        logger.exception(f"[v2/agent/approve] 승인 처리 실패: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "thread_id": thread_id,
                "session_id": session_id,
                "graph_run_id": request_id,
                "error": "승인 처리 중 내부 오류가 발생했습니다.",
            },
        )


@app.post("/v2/agent/cancel")
@_rate_limit("30/minute")
async def v2_agent_cancel(
    thread_id: str,
    _http_request: Request,
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
        graph_state = await manager.graph.aget_state(config)
        if not graph_state or not graph_state.next:
            raise HTTPException(
                status_code=409,
                detail="해당 thread는 현재 interrupt 대기 상태가 아닙니다.",
            )

        session_id = graph_state.values.get("session_id", "")

        # 강제 거절 + interrupt_reason 전달로 resume
        result = await manager.graph.ainvoke(
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
        logger.exception(f"[v2/agent/cancel] 취소 처리 실패: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "thread_id": thread_id,
                "error": "취소 처리 중 내부 오류가 발생했습니다.",
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, **runtime_config.to_uvicorn_kwargs())
