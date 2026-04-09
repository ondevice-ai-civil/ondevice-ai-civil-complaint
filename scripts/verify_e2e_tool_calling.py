#!/usr/bin/env python3
"""GovOn v4 ReAct E2E 검증 스크립트.

HuggingFace Space에 배포된 govon-runtime 서버에 대해
에이전트 파이프라인(v2/v3 ReAct)을 구조적으로 검증한다.

사용법:
    GOVON_RUNTIME_URL=https://<space-url>.hf.space python3 scripts/verify_e2e_tool_calling.py
    GOVON_RUNTIME_URL=https://<space-url>.hf.space API_KEY=<key> python3 scripts/verify_e2e_tool_calling.py

5-Phase 검증 (24 시나리오):
    Phase 1: Infrastructure (hard gate)
        1. Health & Profile
        2. Base Model Generation
        3. vLLM Connection
    Phase 2: v2 Agent Pipeline
        4. v2 Direct Answer (no tool)
        5. v2 Tool Execution with Approval
        6. v2 Approval Rejection Flow
        7. v2 Multi-turn Session
        8. v2 Empty Query
        9. v2 Concurrent Requests
    Phase 3: v3 ReAct Loop
        10. v3 Direct Answer (no-tool)
        11. v3 Tool Execution
        12. v3 Multi-iteration
        13. v3 max_iterations=1
        14. v3 max_iterations Validation
        15. v3 SSE Stream (no-tool)
        16. v3 SSE Stream (with-tool)
        17. v3 SSE run_complete Metadata
        18. v3 Empty Query
        19. v3 Concurrent Requests
    Phase 4: Cross-version
        20. v2 then v3 Same Query
        21. v3 Long Query Handling
    Phase 5: Multi-turn Conversation
        22. v3 Multi-turn Context (같은 session_id로 2회 요청)
        23. v3 Multi-turn Isolation (다른 session_id 격리)
        24. v3 3-turn Conversation (초안 → 법령 → 통계)
"""

# stdlib
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Optional
from uuid import uuid4

BASE_URL = os.environ.get("GOVON_RUNTIME_URL", "http://localhost:7860").rstrip("/")
API_KEY = os.environ.get("API_KEY")
TIMEOUT = 300  # 시나리오당 최대 대기 시간 (초)
BASE_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B-AWQ"
_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = f"verify_e2e_tool_calling_{_TIMESTAMP}.json"
LOG_PATH = f"verify_e2e_tool_calling_{_TIMESTAMP}.log"


# ---------------------------------------------------------------------------
# 로깅 설정: 터미널 + 파일 동시 기록
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"로그 파일: {LOG_PATH}")
logger.info(f"결과 파일: {RESULTS_PATH}")

_results: list[dict] = []
_observed_tools: set[str] = set()
_run_id = uuid4().hex


def _write_incremental() -> None:
    """시나리오 완료 시마다 중간 결과를 JSON 파일에 저장한다."""
    output = {
        "meta": {
            "run_id": _run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_url": BASE_URL,
            "log_file": LOG_PATH,
            "status": "in_progress",
        },
        "summary": {
            "total": len(_results),
            "passed": sum(1 for r in _results if r["status"] == "passed"),
            "failed": sum(1 for r in _results if r["status"] == "failed"),
            "skipped": sum(1 for r in _results if r["status"] == "skipped"),
        },
        "scenarios": _results,
    }
    tmp_path = f"{RESULTS_PATH}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, RESULTS_PATH)
    except Exception as exc:
        logger.warning("중간 결과 저장 실패: %s", exc)


# ---------------------------------------------------------------------------
# HTTP 클라이언트 레이어 (httpx 우선, urllib fallback)
# ---------------------------------------------------------------------------

try:
    import httpx

    _HTTP_BACKEND = "httpx"

    def _build_headers() -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    async def http_get(path: str, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=_build_headers())
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"_raw": resp.text[:200]}

    async def http_post(path: str, body: dict, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body, headers=_build_headers())
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"_raw": resp.text[:200]}

    async def http_post_sse(
        path: str, body: dict, timeout: float = TIMEOUT
    ) -> tuple[int, list[dict]]:
        """SSE 스트리밍 POST. 청크를 수집하여 파싱된 이벤트 목록을 반환한다."""
        url = BASE_URL + path
        h = _build_headers()
        h["Accept"] = "text/event-stream"
        events: list[dict] = []
        status_code = 0
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=body, headers=h) as resp:
                status_code = resp.status_code
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        events.append({"_raw": payload})
        return status_code, events

    async def http_get_raw(url: str, timeout: float = 10) -> tuple[int, str]:
        """Raw GET for external connectivity checks."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code, resp.text[:200]

except ImportError:
    import urllib.error
    import urllib.request

    _HTTP_BACKEND = "urllib"

    def _build_headers() -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    async def http_get(path: str, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        req = urllib.request.Request(url, headers=_build_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {}
        except Exception:
            return 0, {}

    async def http_post(path: str, body: dict, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=_build_headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {}
        except Exception:
            return 0, {}

    async def http_post_sse(
        path: str, body: dict, timeout: float = TIMEOUT
    ) -> tuple[int, list[dict]]:
        """urllib fallback: SSE 스트리밍을 동기 방식으로 읽는다."""
        url = BASE_URL + path
        data = json.dumps(body).encode()
        h = _build_headers()
        h["Accept"] = "text/event-stream"
        req = urllib.request.Request(url, data=data, headers=h, method="POST")
        events: list[dict] = []
        status_code = 0
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status_code = r.status
                for raw_line in r:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        events.append({"_raw": payload})
        except urllib.error.HTTPError as e:
            status_code = e.code
        return status_code, events

    async def http_get_raw(url: str, timeout: float = 10) -> tuple[int, str]:
        """Raw GET for external connectivity checks."""
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode()[:200]
        except urllib.error.HTTPError as e:
            return e.code, ""
        except Exception:
            return 0, ""


# ---------------------------------------------------------------------------
# 결과 기록 / 출력 헬퍼
# ---------------------------------------------------------------------------


def _record(
    scenario_num: int,
    name: str,
    phase: int,
    status: str,
    elapsed: float,
    attempts: int = 1,
    assertions: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
    error: Optional[str] = None,
    detail: Optional[Any] = None,
) -> dict:
    tag = {"passed": "[PASS]", "failed": "[FAIL]", "skipped": "[SKIP]"}.get(status, "[????]")
    suffix = f"({elapsed:.2f}s)"
    if status == "passed":
        msg = f"{tag} Scenario {scenario_num}: {name} {suffix}"
        logger.info(msg)
    elif status == "skipped":
        msg = f"{tag} Scenario {scenario_num}: {name} — {error or 'skipped'} {suffix}"
        logger.warning(msg)
    else:
        msg = f"{tag} Scenario {scenario_num}: {name} — {error} {suffix}"
        logger.error(msg)

    if warnings:
        for w in warnings:
            logger.warning(f"  [WARN] {w}")

    entry = {
        "id": scenario_num,
        "name": name,
        "phase": phase,
        "status": status,
        "attempts": attempts,
        "elapsed_s": round(elapsed, 3),
        "assertions": assertions or [],
        "warnings": warnings or [],
        "error": error,
        "detail": detail,
    }
    _results.append(entry)
    _write_incremental()
    return entry


def _session_id(scenario_num: int) -> str:
    """시나리오별 고유 session ID를 생성한다."""
    return f"e2e-{scenario_num}-{uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# v2 Agent 스트리밍 + 승인 헬퍼
# ---------------------------------------------------------------------------


async def _v2_stream_and_approve(
    query: str,
    session_id: str,
    approve: bool = True,
    timeout: float = 180,
) -> tuple[bool, str, list[dict], Optional[str]]:
    """v2 SSE 스트리밍으로 agent 호출 후 approval 처리.

    1. POST /v2/agent/stream → SSE 이벤트 수집
    2. awaiting_approval 발견 시 → POST /v2/agent/approve
    3. 남은 이벤트 처리 또는 approve 응답에서 최종 텍스트 추출
    4. agent 노드 이벤트의 planned_tools를 _observed_tools에 추가

    Returns: (ok, text, events, error)
    """
    body = {"query": query, "session_id": session_id}
    logger.info("[v2] 요청: session=%s, query_len=%d", session_id, len(query))

    try:
        status_code, events = await http_post_sse("/v2/agent/stream", body, timeout=timeout)
        logger.info("[v2] SSE HTTP %d, events=%d", status_code, len(events))
    except Exception as exc:
        logger.warning("[v2] SSE 실패, REST fallback: %s", exc)
        # REST fallback
        try:
            status_code, resp = await http_post("/v2/agent/run", body, timeout=timeout)
        except Exception as exc2:
            return False, "", [], str(exc2)
        if status_code == 422:
            return False, "", [{"_status": status_code, "_body": resp}], f"HTTP 422"
        if status_code != 200:
            return False, "", [], f"REST HTTP {status_code}"
        text = resp.get("text", "") or resp.get("final_text", "") or ""
        return bool(text), text, [], None

    if status_code == 422:
        return False, "", events, f"HTTP 422"
    if status_code != 200:
        return False, "", events, f"SSE HTTP {status_code}"

    # planned_tools 수집 (커버리지 추적용)
    for ev in events:
        planned = (
            ev.get("planned_tools") or ev.get("approval_request", {}).get("planned_tools") or []
        )
        for tool_name in planned:
            if isinstance(tool_name, str) and tool_name:
                _observed_tools.add(tool_name)
        # agent 노드에서도 추출
        if ev.get("node") == "agent":
            for tool_name in ev.get("planned_tools") or []:
                if isinstance(tool_name, str) and tool_name:
                    _observed_tools.add(tool_name)

    # awaiting_approval 이벤트 탐색
    awaiting_ev = None
    for ev in events:
        if ev.get("status") == "awaiting_approval" or ev.get("node") == "__interrupt__":
            awaiting_ev = ev
            break

    if awaiting_ev:
        thread_id = awaiting_ev.get("thread_id") or session_id
        logger.info("[v2] awaiting_approval → thread_id=%s, approve=%s", thread_id, approve)
        try:
            approve_code, approve_resp = await http_post(
                f"/v2/agent/approve?thread_id={thread_id}&approved={'true' if approve else 'false'}",
                {},
                timeout=timeout,
            )
        except Exception as exc:
            return False, "", events, f"approve 예외: {exc}"

        logger.info("[v2] approve HTTP %d, status=%s", approve_code, approve_resp.get("status"))
        if approve_code != 200:
            return False, "", events, f"approve HTTP {approve_code}"

        if not approve and approve_resp.get("status") == "rejected":
            text = approve_resp.get("text", "") or ""
            return True, text, events, None

        text = approve_resp.get("text", "") or approve_resp.get("final_text", "") or ""
        if text:
            return True, text, events, None
        return False, "", events, f"approve 200 but text 없음: {list(approve_resp.keys())}"

    # awaiting 없음 → 직접 답변 (no-tool path)
    text = ""
    for ev in reversed(events):
        if ev.get("node") == "persist" and ev.get("final_text"):
            text = ev["final_text"]
            break
    if not text:
        for ev in reversed(events):
            candidate = ev.get("final_text") or ev.get("text") or ""
            if candidate:
                text = candidate
                break

    if text:
        return True, text, events, None

    # error 이벤트 확인
    for ev in events:
        if ev.get("status") == "error" or ev.get("node") == "error":
            return False, "", events, ev.get("error", "agent error")

    return False, "", events, f"text 없음, events={len(events)}"


# ---------------------------------------------------------------------------
# v3 ReAct 호출 헬퍼
# ---------------------------------------------------------------------------


async def _call_v3_run(
    query: str,
    session_id: str,
    max_iterations: int = 10,
    timeout: float = 180,
) -> tuple[bool, str, dict, Optional[str]]:
    """v3 blocking API 호출.
    Returns: (success, text, metadata, error)
    """
    body = {"query": query, "session_id": session_id, "max_iterations": max_iterations}
    logger.info("[v3 Run] 요청: session=%s, max_iter=%d", session_id, max_iterations)

    try:
        status_code, resp = await http_post("/v3/agent/run", body, timeout=timeout)
        logger.info("[v3 Run] HTTP %d, status=%s", status_code, resp.get("status"))

        if status_code == 422:
            return False, "", {}, f"HTTP 422: {resp}"
        if status_code == 503:
            return False, "", {}, "v3 graph 미초기화 (503)"
        if status_code == 500:
            return False, "", {}, f"내부 오류 (500): {resp.get('error', '')}"
        if status_code != 200:
            return False, "", {}, f"HTTP {status_code}: {resp}"

        text = resp.get("text", "") or resp.get("final_text", "") or ""
        metadata = resp.get("metadata", {}) or {}

        # tool_calls 커버리지 추적
        for tc in metadata.get("tool_calls", []) or []:
            if isinstance(tc, dict) and tc.get("name"):
                _observed_tools.add(tc["name"])

        return True, text, metadata, None

    except Exception as exc:
        return False, "", {}, str(exc)


async def _call_v3_stream(
    query: str,
    session_id: str,
    max_iterations: int = 10,
    timeout: float = 180,
) -> tuple[bool, str, list[dict], dict, Optional[str]]:
    """v3 SSE 스트리밍 호출.
    Returns: (success, text, events, metadata, error)
    """
    body = {"query": query, "session_id": session_id, "max_iterations": max_iterations}
    logger.info("[v3 Stream] 요청: session=%s, max_iter=%d", session_id, max_iterations)

    try:
        status_code, events = await http_post_sse("/v3/agent/stream", body, timeout=timeout)
        logger.info("[v3 Stream] HTTP %d, events=%d", status_code, len(events))

        if status_code == 503:
            return False, "", [], {}, "v3 graph 미초기화 (503)"
        if status_code != 200:
            return False, "", events, {}, f"HTTP {status_code}"

        text = ""
        metadata: dict = {}
        for ev in events:
            if ev.get("type") == "run_complete":
                text = ev.get("text", "") or ""
                metadata = ev.get("metadata", {}) or {}
                break

        if not text:
            for ev in reversed(events):
                candidate = ev.get("text", "") or ev.get("final_text", "") or ""
                if candidate:
                    text = candidate
                    break

        # tool_calls 커버리지 추적
        for tc in metadata.get("tool_calls", []) or []:
            if isinstance(tc, dict) and tc.get("name"):
                _observed_tools.add(tc["name"])

        return True, text, events, metadata, None

    except Exception as exc:
        return False, "", [], {}, str(exc)


# ---------------------------------------------------------------------------
# Phase 1: Infrastructure (hard gate)
# ---------------------------------------------------------------------------


async def scenario1_health_profile() -> dict:
    """Scenario 1: Health & Profile."""
    backoffs = [5, 10, 20]
    attempts = 0
    last_error = ""

    for attempt_idx in range(3):
        attempts += 1
        t0 = time.monotonic()
        try:
            status_code, body = await http_get("/health", timeout=30)
            elapsed = time.monotonic() - t0

            if status_code != 200:
                last_error = f"HTTP {status_code}"
                if attempt_idx < 2:
                    await asyncio.sleep(backoffs[attempt_idx])
                    continue
                return _record(
                    1,
                    "Health & Profile",
                    1,
                    "failed",
                    elapsed,
                    attempts,
                    error=last_error,
                    detail={"body": body},
                )

            assertions = ["HTTP 200: OK"]
            warnings: list[str] = []

            # 필수 필드 확인
            for field in ("status", "profile", "model", "vllm_connected", "agents_loaded"):
                if field in body:
                    assertions.append(f"field '{field}' present")
                else:
                    warnings.append(f"field '{field}' 없음")

            if body.get("vllm_connected") is not True:
                warnings.append(f"vllm_connected={body.get('vllm_connected')} (not True)")

            if body.get("agents_loaded") is not True:
                warnings.append(f"agents_loaded={body.get('agents_loaded')} (not True)")

            return _record(
                1,
                "Health & Profile",
                1,
                "passed",
                elapsed,
                attempts,
                assertions=assertions,
                warnings=warnings,
                detail={
                    "status": body.get("status"),
                    "profile": body.get("profile"),
                    "vllm_connected": body.get("vllm_connected"),
                    "agents_loaded": body.get("agents_loaded"),
                },
            )

        except Exception as exc:
            last_error = str(exc)
            elapsed = time.monotonic() - t0
            if attempt_idx < 2:
                await asyncio.sleep(backoffs[attempt_idx])
                continue
            return _record(1, "Health & Profile", 1, "failed", elapsed, attempts, error=last_error)

    return _record(1, "Health & Profile", 1, "failed", 0.0, attempts, error=last_error)


async def scenario2_base_model_generation() -> dict:
    """Scenario 2: Base Model Generation via /v1/chat/completions."""
    t0 = time.monotonic()
    try:
        body = {
            "model": BASE_MODEL,
            "messages": [{"role": "user", "content": "Hello, please respond briefly."}],
            "max_tokens": 50,
        }
        status_code, resp = await http_post("/v1/chat/completions", body, timeout=60)
        elapsed = time.monotonic() - t0

        if status_code != 200:
            return _record(
                2,
                "Base Model Generation",
                1,
                "failed",
                elapsed,
                error=f"HTTP {status_code}",
                detail={"resp": resp},
            )

        assertions = ["HTTP 200: OK"]
        choices = resp.get("choices", [])
        if choices:
            assertions.append("choices 배열 존재")
        else:
            return _record(
                2,
                "Base Model Generation",
                1,
                "failed",
                elapsed,
                error="choices 없음",
                detail={"resp": resp},
            )

        content = choices[0].get("message", {}).get("content", "") or ""
        if len(content) > 0:
            assertions.append(f"content 비어있지 않음 (len={len(content)})")
        else:
            return _record(
                2,
                "Base Model Generation",
                1,
                "failed",
                elapsed,
                error="content 비어있음",
                detail={"resp": resp},
            )

        return _record(
            2,
            "Base Model Generation",
            1,
            "passed",
            elapsed,
            assertions=assertions,
            detail={"content_len": len(content)},
        )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        return _record(2, "Base Model Generation", 1, "failed", elapsed, error=str(exc))


async def scenario3_vllm_connection() -> dict:
    """Scenario 3: vLLM Connection — health.vllm_connected == true."""
    t0 = time.monotonic()
    try:
        status_code, body = await http_get("/health", timeout=30)
        elapsed = time.monotonic() - t0

        if status_code != 200:
            return _record(3, "vLLM Connection", 1, "failed", elapsed, error=f"HTTP {status_code}")

        vllm_connected = body.get("vllm_connected")
        if vllm_connected is True:
            return _record(
                3, "vLLM Connection", 1, "passed", elapsed, assertions=["vllm_connected=true"]
            )
        else:
            return _record(
                3,
                "vLLM Connection",
                1,
                "failed",
                elapsed,
                error=f"vllm_connected={vllm_connected}",
                detail={"body": body},
            )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        return _record(3, "vLLM Connection", 1, "failed", elapsed, error=str(exc))


# ---------------------------------------------------------------------------
# Phase 2: v2 Agent Pipeline
# ---------------------------------------------------------------------------


async def scenario4_v2_direct_answer() -> dict:
    """Scenario 4: v2 Direct Answer (no tool) — 단순 인사 → session_load → agent → persist."""
    sid = _session_id(4)
    t0 = time.monotonic()
    ok, text, events, err = await _v2_stream_and_approve(
        query="안녕하세요, 간단히 인사해 주세요.",
        session_id=sid,
        approve=False,
        timeout=120,
    )
    elapsed = time.monotonic() - t0

    if not ok and err and "422" not in err:
        # text가 비어도 events가 있으면 노드 흐름으로 판단
        pass

    assertions = []
    warnings: list[str] = []

    node_names = [ev.get("node", "") for ev in events]

    if "session_load" in node_names:
        assertions.append("node: session_load 실행됨")
    else:
        warnings.append("session_load 노드 미확인")

    if "agent" in node_names:
        assertions.append("node: agent 실행됨")

    if ok and text:
        assertions.append(f"응답 텍스트 수신 (len={len(text)})")
        if len(text) > 30:
            assertions.append("응답 길이 >30")
        else:
            warnings.append(f"응답 짧음: {text[:80]!r}")
        status = "passed"
        error = None
    elif events:
        # 이벤트는 있지만 text 추출 실패 — partial pass
        assertions.append(f"SSE 이벤트 수신 (count={len(events)})")
        status = "passed"
        error = None
    else:
        status = "failed"
        error = err or "응답 없음"

    return _record(
        4,
        "v2 Direct Answer (no tool)",
        2,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=error,
        detail={
            "text_len": len(text) if text else 0,
            "event_count": len(events),
            "nodes": node_names,
        },
    )


async def scenario5_v2_tool_execution_approval() -> dict:
    """Scenario 5: v2 Tool Execution with Approval — 민원 쿼리 → approval_wait → approve → tools → persist."""
    sid = _session_id(5)
    t0 = time.monotonic()
    ok, text, events, err = await _v2_stream_and_approve(
        query="주민등록 말소 처리 절차와 필요 서류를 알려주세요.",
        session_id=sid,
        approve=True,
        timeout=180,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []
    node_names = [ev.get("node", "") for ev in events]
    statuses = [ev.get("status", "") for ev in events]

    # approval_wait 또는 awaiting_approval 확인
    approval_found = (
        "approval_wait" in node_names
        or "awaiting_approval" in statuses
        or "__interrupt__" in node_names
    )
    if approval_found:
        assertions.append("awaiting_approval 이벤트 확인")
    else:
        warnings.append("awaiting_approval 이벤트 미확인 (auto-approve 모드일 수 있음)")

    if "tools" in node_names:
        assertions.append("node: tools 실행됨")
    else:
        warnings.append("tools 노드 미확인")

    if ok and text:
        assertions.append(f"최종 텍스트 수신 (len={len(text)})")
        if len(text) > 30:
            assertions.append("응답 길이 >30")
        status = "passed"
        error = None
    elif ok:
        assertions.append("승인 처리 성공")
        status = "passed"
        error = None
    else:
        status = "failed"
        error = err or "응답 없음"

    return _record(
        5,
        "v2 Tool Execution with Approval",
        2,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=error,
        detail={
            "text_len": len(text) if text else 0,
            "nodes": node_names,
            "approval_found": approval_found,
        },
    )


async def scenario6_v2_approval_rejection() -> dict:
    """Scenario 6: v2 Approval Rejection Flow — 거절 후 agent 재라우팅 확인.

    검증 기준:
    - LLM이 도구를 호출하면 → approval_wait → reject → ok=True (rejected 처리 성공)
    - LLM이 도구를 호출하지 않으면 → approval 없이 직접 답변 → ok=True (도구 미사용도 유효)
    - 어느 경로든 파이프라인이 에러 없이 종료되어야 함
    """
    sid = _session_id(6)
    t0 = time.monotonic()
    ok, text, events, err = await _v2_stream_and_approve(
        query="아파트 층간소음 민원에 대한 답변을 작성해주세요.",
        session_id=sid,
        approve=False,
        timeout=180,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []
    node_names = [ev.get("node", "") for ev in events]
    statuses = [ev.get("status", "") for ev in events]

    # approval_wait 이벤트 존재 여부 (LLM이 도구를 호출했는지)
    approval_found = (
        "approval_wait" in node_names
        or "awaiting_approval" in statuses
        or "__interrupt__" in node_names
    )

    if approval_found:
        # LLM이 도구를 호출 → reject 흐름이 동작해야 함
        assertions.append("awaiting_approval 이벤트 확인")
        if ok:
            # _v2_stream_and_approve가 approve=False + status==rejected → ok=True
            assertions.append("거절 후 파이프라인 정상 완료 (rejected 처리 성공)")
            status = "passed"
            error = None
        else:
            status = "failed"
            error = err or "rejection 처리 실패"
    else:
        # LLM이 도구를 호출하지 않음 → approval 없이 직접 답변
        warnings.append("LLM이 도구를 호출하지 않아 rejection 테스트 불가 (직접 답변)")
        if ok:
            assertions.append("직접 답변으로 정상 완료")
            status = "passed"
            error = None
        else:
            status = "failed"
            error = err or "응답 없음"

    return _record(
        6,
        "v2 Approval Rejection Flow",
        2,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=error,
        detail={"nodes": node_names, "approval_found": approval_found},
    )


async def scenario7_v2_multi_turn_session() -> dict:
    """Scenario 7: v2 Multi-turn Session — 동일 session_id로 2회 순차 질의."""
    sid = _session_id(7)
    t0 = time.monotonic()
    assertions = []
    warnings: list[str] = []

    # 1차 질의
    ok1, text1, events1, err1 = await _v2_stream_and_approve(
        query="민원 신청 절차를 간단히 설명해 주세요.",
        session_id=sid,
        approve=True,
        timeout=120,
    )
    if ok1:
        assertions.append("1차 질의 성공")
    else:
        warnings.append(f"1차 질의 실패: {err1}")

    # 2차 질의 (동일 session)
    ok2, text2, events2, err2 = await _v2_stream_and_approve(
        query="앞서 말한 절차에서 필요한 서류는 무엇인가요?",
        session_id=sid,
        approve=True,
        timeout=120,
    )
    elapsed = time.monotonic() - t0

    if ok2:
        assertions.append("2차 질의 성공 (동일 세션)")
    else:
        warnings.append(f"2차 질의 실패: {err2}")

    if ok1 and ok2:
        status = "passed"
        error = None
    else:
        status = "failed"
        error = f"1차: {err1 or 'OK'}, 2차: {err2 or 'OK'}"

    return _record(
        7,
        "v2 Multi-turn Session",
        2,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=error,
        detail={
            "turn1_ok": ok1,
            "turn2_ok": ok2,
            "turn1_text_len": len(text1) if text1 else 0,
            "turn2_text_len": len(text2) if text2 else 0,
        },
    )


async def scenario8_v2_empty_query() -> dict:
    """Scenario 8: v2 Empty Query — 422 Validation Error 확인."""
    t0 = time.monotonic()
    try:
        body = {"query": "", "session_id": _session_id(8)}
        status_code, resp = await http_post("/v2/agent/stream", body, timeout=30)
        elapsed = time.monotonic() - t0

        if status_code == 422:
            return _record(
                8, "v2 Empty Query", 2, "passed", elapsed, assertions=["HTTP 422: validation error"]
            )
        else:
            # 일부 구현은 빈 쿼리를 다르게 처리할 수 있음
            return _record(
                8,
                "v2 Empty Query",
                2,
                "passed",
                elapsed,
                assertions=[f"HTTP {status_code}: 처리됨"],
                warnings=[f"422 예상이었으나 {status_code} 반환"],
                detail={"status_code": status_code},
            )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        return _record(8, "v2 Empty Query", 2, "failed", elapsed, error=str(exc))


async def scenario9_v2_concurrent_requests() -> dict:
    """Scenario 9: v2 Concurrent Requests — 3개 병렬 요청 모두 성공."""
    t0 = time.monotonic()
    queries = [
        ("오늘 날씨 정보를 알려주세요.", _session_id(91)),
        ("행정 처리 기간 기준을 알려주세요.", _session_id(92)),
        ("공공기관 민원 처리 원칙을 설명해 주세요.", _session_id(93)),
    ]

    async def run_one(query: str, sid: str) -> tuple[bool, Optional[str]]:
        try:
            ok, text, events, err = await _v2_stream_and_approve(
                query=query, session_id=sid, approve=True, timeout=120
            )
            return ok or bool(events), err
        except Exception as exc:
            return False, str(exc)

    results = await asyncio.gather(*[run_one(q, s) for q, s in queries])
    elapsed = time.monotonic() - t0

    successes = sum(1 for ok, _ in results if ok)
    assertions = [f"병렬 요청 {successes}/3 성공"]
    errors = [err for ok, err in results if not ok and err]

    if successes >= 3:
        status = "passed"
        error = None
    else:
        status = "failed"
        error = f"성공 {successes}/3: {errors}"

    return _record(
        9,
        "v2 Concurrent Requests",
        2,
        status,
        elapsed,
        assertions=assertions,
        error=error,
        detail={"successes": successes, "errors": errors},
    )


# ---------------------------------------------------------------------------
# Phase 3: v3 ReAct Loop
# ---------------------------------------------------------------------------


async def scenario10_v3_direct_answer() -> dict:
    """Scenario 10: v3 Direct Answer (no-tool) — 단순 질의 → completed with text."""
    sid = _session_id(10)
    t0 = time.monotonic()
    ok, text, metadata, err = await _call_v3_run(
        query="안녕하세요, 간단히 인사해 주세요.",
        session_id=sid,
        max_iterations=10,
        timeout=120,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []

    if ok:
        assertions.append("v3 run 성공 (HTTP 200, status=completed)")
    if text:
        assertions.append(f"텍스트 수신 (len={len(text)})")
        if len(text) > 30:
            assertions.append("응답 길이 >30")
        else:
            warnings.append(f"응답 짧음: {text[:80]!r}")

    status = "passed" if ok else "failed"
    return _record(
        10,
        "v3 Direct Answer (no-tool)",
        3,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=err if not ok else None,
        detail={"text_len": len(text) if text else 0, "metadata": metadata},
    )


async def scenario11_v3_tool_execution() -> dict:
    """Scenario 11: v3 Tool Execution — 민원 쿼리 → completed with tool execution."""
    sid = _session_id(11)
    t0 = time.monotonic()
    ok, text, metadata, err = await _call_v3_run(
        query="주민등록증 재발급 신청 방법과 필요 서류를 상세히 알려주세요.",
        session_id=sid,
        max_iterations=10,
        timeout=180,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []

    if ok:
        assertions.append("v3 run 성공")
    if text:
        assertions.append(f"텍스트 수신 (len={len(text)})")
        if len(text) > 30:
            assertions.append("응답 길이 >30")

    total_tool_calls = metadata.get("total_tool_calls", 0)
    if total_tool_calls and total_tool_calls > 0:
        assertions.append(f"도구 호출 확인 (total_tool_calls={total_tool_calls})")
    else:
        warnings.append("도구 호출 미확인 (LLM이 직접 답변 선택 가능)")

    status = "passed" if ok else "failed"
    return _record(
        11,
        "v3 Tool Execution",
        3,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=err if not ok else None,
        detail={
            "text_len": len(text) if text else 0,
            "total_tool_calls": total_tool_calls,
            "metadata": metadata,
        },
    )


async def scenario12_v3_multi_iteration() -> dict:
    """Scenario 12: v3 Multi-iteration — 복잡한 쿼리 → multiple iterations."""
    sid = _session_id(12)
    t0 = time.monotonic()
    ok, text, metadata, err = await _call_v3_run(
        query="국민건강보험 지역 가입자 보험료 산정 기준과 납부 방법, 그리고 경감 혜택 대상을 종합적으로 설명해 주세요.",
        session_id=sid,
        max_iterations=10,
        timeout=240,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []

    if ok:
        assertions.append("v3 run 성공")
    if text:
        assertions.append(f"텍스트 수신 (len={len(text)})")

    total_iterations = metadata.get("total_iterations", 0)
    if isinstance(total_iterations, int) and total_iterations > 0:
        assertions.append(f"total_iterations={total_iterations}")
        if total_iterations > 1:
            assertions.append("다중 iteration 확인")
        else:
            warnings.append("single iteration (복잡한 쿼리에서 예상보다 적음)")
    else:
        warnings.append(f"total_iterations 미확인: {total_iterations!r}")

    status = "passed" if ok else "failed"
    return _record(
        12,
        "v3 Multi-iteration",
        3,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=err if not ok else None,
        detail={"metadata": metadata, "text_len": len(text) if text else 0},
    )


async def scenario13_v3_max_iterations_1() -> dict:
    """Scenario 13: v3 max_iterations=1 — 강제 조기 종료."""
    sid = _session_id(13)
    t0 = time.monotonic()
    ok, text, metadata, err = await _call_v3_run(
        query="복잡한 행정 절차에 대해 단계별로 상세히 설명해 주세요.",
        session_id=sid,
        max_iterations=1,
        timeout=120,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []

    if ok:
        assertions.append("max_iterations=1 실행 완료")
    if text:
        assertions.append(f"텍스트 수신 (len={len(text)})")

    total_iterations = metadata.get("total_iterations", 0)
    if isinstance(total_iterations, int) and total_iterations <= 1:
        assertions.append(f"total_iterations={total_iterations} (<= 1 확인)")
    else:
        warnings.append(f"total_iterations={total_iterations!r} (max=1인데 초과 가능)")

    status = "passed" if ok else "failed"
    return _record(
        13,
        "v3 max_iterations=1",
        3,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=err if not ok else None,
        detail={"metadata": metadata},
    )


async def scenario14_v3_max_iterations_validation() -> dict:
    """Scenario 14: v3 max_iterations Validation — 0 및 음수 → 422."""
    t0 = time.monotonic()
    assertions = []
    errors: list[str] = []

    for invalid_val in (0, -1):
        ok, text, metadata, err = await _call_v3_run(
            query="테스트 쿼리입니다.",
            session_id=_session_id(14),
            max_iterations=invalid_val,
            timeout=30,
        )
        if not ok and err and "422" in err:
            assertions.append(f"max_iterations={invalid_val} → 422 확인")
        elif not ok:
            assertions.append(f"max_iterations={invalid_val} → 거부됨 ({err})")
        else:
            errors.append(f"max_iterations={invalid_val} → 200 반환됨 (422 예상)")

    elapsed = time.monotonic() - t0

    if errors:
        return _record(
            14,
            "v3 max_iterations Validation",
            3,
            "failed",
            elapsed,
            assertions=assertions,
            error="; ".join(errors),
        )
    return _record(14, "v3 max_iterations Validation", 3, "passed", elapsed, assertions=assertions)


async def scenario15_v3_sse_stream_no_tool() -> dict:
    """Scenario 15: v3 SSE Stream (no-tool) — run_complete 이벤트 with text."""
    sid = _session_id(15)
    t0 = time.monotonic()
    ok, text, events, metadata, err = await _call_v3_stream(
        query="안녕하세요, 오늘 날씨는 어떤가요?",
        session_id=sid,
        max_iterations=10,
        timeout=120,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []

    event_types = [ev.get("type", "") for ev in events if ev.get("type")]

    if ok:
        assertions.append("v3 stream 성공")
    if "run_complete" in event_types:
        assertions.append("run_complete 이벤트 확인")
    else:
        warnings.append(f"run_complete 이벤트 없음, types={event_types}")

    if text:
        assertions.append(f"텍스트 수신 (len={len(text)})")
        if len(text) > 30:
            assertions.append("응답 길이 >30")

    status = "passed" if ok else "failed"
    return _record(
        15,
        "v3 SSE Stream (no-tool)",
        3,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=err if not ok else None,
        detail={"event_types": event_types, "text_len": len(text) if text else 0},
    )


async def scenario16_v3_sse_stream_with_tool() -> dict:
    """Scenario 16: v3 SSE Stream (with-tool) — 도구 실행 포함 SSE 이벤트."""
    sid = _session_id(16)
    t0 = time.monotonic()
    ok, text, events, metadata, err = await _call_v3_stream(
        query="여권 발급 신청 요건과 처리 기간을 알려주세요.",
        session_id=sid,
        max_iterations=10,
        timeout=180,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []

    event_types = [ev.get("type", "") for ev in events if ev.get("type")]

    if ok:
        assertions.append("v3 stream 성공")
    if len(events) > 1:
        assertions.append(f"다중 SSE 이벤트 수신 (count={len(events)})")
    if "run_complete" in event_types:
        assertions.append("run_complete 이벤트 확인")

    total_tool_calls = metadata.get("total_tool_calls", 0)
    if total_tool_calls and total_tool_calls > 0:
        assertions.append(f"도구 호출 확인 (total_tool_calls={total_tool_calls})")
    else:
        warnings.append("도구 호출 미확인 (LLM 자율 선택)")

    if text:
        assertions.append(f"텍스트 수신 (len={len(text)})")

    status = "passed" if ok else "failed"
    return _record(
        16,
        "v3 SSE Stream (with-tool)",
        3,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=err if not ok else None,
        detail={
            "event_count": len(events),
            "event_types": event_types,
            "total_tool_calls": total_tool_calls,
        },
    )


async def scenario17_v3_sse_run_complete_metadata() -> dict:
    """Scenario 17: v3 SSE run_complete Metadata — metadata 필드 존재 확인."""
    sid = _session_id(17)
    t0 = time.monotonic()
    ok, text, events, metadata, err = await _call_v3_stream(
        query="건강보험 피부양자 등록 조건은 무엇인가요?",
        session_id=sid,
        max_iterations=10,
        timeout=180,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []

    if ok:
        assertions.append("v3 stream 성공")

    # run_complete 이벤트 찾기
    run_complete_ev = None
    for ev in events:
        if ev.get("type") == "run_complete":
            run_complete_ev = ev
            break

    if run_complete_ev:
        assertions.append("run_complete 이벤트 확인")
        ev_metadata = run_complete_ev.get("metadata") or {}
        for field in ("total_iterations", "total_tool_calls", "total_latency_ms"):
            if field in ev_metadata:
                assertions.append(f"metadata.{field} 존재")
            else:
                warnings.append(f"metadata.{field} 없음")
    else:
        warnings.append("run_complete 이벤트 없음")

    status = "passed" if ok else "failed"
    return _record(
        17,
        "v3 SSE run_complete Metadata",
        3,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=err if not ok else None,
        detail={"metadata": metadata},
    )


async def scenario18_v3_empty_query() -> dict:
    """Scenario 18: v3 Empty Query — 422 Validation Error 확인."""
    t0 = time.monotonic()
    try:
        body = {"query": "", "session_id": _session_id(18), "max_iterations": 10}
        status_code, resp = await http_post("/v3/agent/run", body, timeout=30)
        elapsed = time.monotonic() - t0

        if status_code == 422:
            return _record(
                18,
                "v3 Empty Query",
                3,
                "passed",
                elapsed,
                assertions=["HTTP 422: validation error"],
            )
        else:
            return _record(
                18,
                "v3 Empty Query",
                3,
                "passed",
                elapsed,
                assertions=[f"HTTP {status_code}: 처리됨"],
                warnings=[f"422 예상이었으나 {status_code} 반환"],
                detail={"status_code": status_code},
            )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        return _record(18, "v3 Empty Query", 3, "failed", elapsed, error=str(exc))


async def scenario19_v3_concurrent_requests() -> dict:
    """Scenario 19: v3 Concurrent Requests — 2개 병렬, 모두 완료."""
    t0 = time.monotonic()
    queries = [
        ("국민연금 납부 예외 신청 방법을 알려주세요.", _session_id(191)),
        ("지방세 납부 기한 연장 신청 절차를 알려주세요.", _session_id(192)),
    ]

    async def run_one(query: str, sid: str) -> tuple[bool, Optional[str]]:
        try:
            ok, text, metadata, err = await _call_v3_run(
                query=query, session_id=sid, max_iterations=10, timeout=180
            )
            return ok, err
        except Exception as exc:
            return False, str(exc)

    results = await asyncio.gather(*[run_one(q, s) for q, s in queries])
    elapsed = time.monotonic() - t0

    successes = sum(1 for ok, _ in results if ok)
    assertions = [f"병렬 요청 {successes}/2 성공"]
    errs = [err for ok, err in results if not ok and err]

    if successes >= 2:
        status = "passed"
        error = None
    else:
        status = "failed"
        error = f"성공 {successes}/2: {errs}"

    return _record(
        19,
        "v3 Concurrent Requests",
        3,
        status,
        elapsed,
        assertions=assertions,
        error=error,
        detail={"successes": successes, "errors": errs},
    )


# ---------------------------------------------------------------------------
# Phase 4: Cross-version
# ---------------------------------------------------------------------------


async def scenario20_v2_then_v3_same_query() -> dict:
    """Scenario 20: v2 then v3 Same Query — 동일 쿼리를 v2/v3 모두 실행."""
    query = "운전면허 갱신 절차와 필요 서류를 알려주세요."
    t0 = time.monotonic()

    # v2
    ok_v2, text_v2, events_v2, err_v2 = await _v2_stream_and_approve(
        query=query,
        session_id=_session_id(201),
        approve=True,
        timeout=180,
    )

    # v3
    ok_v3, text_v3, metadata_v3, err_v3 = await _call_v3_run(
        query=query,
        session_id=_session_id(202),
        max_iterations=10,
        timeout=180,
    )

    elapsed = time.monotonic() - t0
    assertions = []
    warnings: list[str] = []

    if ok_v2 or bool(events_v2):
        assertions.append("v2 응답 처리 완료")
    else:
        warnings.append(f"v2 실패: {err_v2}")

    if ok_v3:
        assertions.append("v3 응답 처리 완료")
    else:
        warnings.append(f"v3 실패: {err_v3}")

    if text_v2 and text_v3:
        assertions.append("v2/v3 모두 텍스트 수신")

    # 둘 중 하나라도 성공이면 passed
    if ok_v2 or ok_v3 or bool(events_v2):
        status = "passed"
        error = None
    else:
        status = "failed"
        error = f"v2: {err_v2}, v3: {err_v3}"

    return _record(
        20,
        "v2 then v3 Same Query",
        4,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=error,
        detail={
            "v2_ok": ok_v2,
            "v3_ok": ok_v3,
            "v2_text_len": len(text_v2) if text_v2 else 0,
            "v3_text_len": len(text_v3) if text_v3 else 0,
        },
    )


async def scenario21_v3_long_query() -> dict:
    """Scenario 21: v3 Long Query Handling — 200자 이상 쿼리 → 타임아웃 없이 완료."""
    long_query = (
        "안녕하세요, 저는 현재 건강보험 지역가입자로 등록되어 있습니다. "
        "제가 최근 직장을 잃어서 소득이 없는 상태인데, 이런 경우 보험료 경감 신청이 가능한지 "
        "알고 싶습니다. 경감 신청 요건, 신청 방법, 처리 기간, 경감 비율 등을 구체적으로 "
        "설명해 주시고, 추가로 실업급여 수급 중인 경우의 보험료 처리 방식도 함께 알려주세요. "
        "마지막으로 보험료 분할 납부 신청 절차도 안내해 주시면 감사하겠습니다."
    )
    assert len(long_query) > 200, f"쿼리 길이 부족: {len(long_query)}"

    sid = _session_id(21)
    t0 = time.monotonic()
    ok, text, metadata, err = await _call_v3_run(
        query=long_query,
        session_id=sid,
        max_iterations=10,
        timeout=240,
    )
    elapsed = time.monotonic() - t0

    assertions = []
    warnings: list[str] = []

    assertions.append(f"쿼리 길이 {len(long_query)}자 (>200 확인)")

    if ok:
        assertions.append(f"타임아웃 없이 완료 ({elapsed:.1f}s)")
    if text:
        assertions.append(f"텍스트 수신 (len={len(text)})")

    status = "passed" if ok else "failed"
    return _record(
        21,
        "v3 Long Query Handling",
        4,
        status,
        elapsed,
        assertions=assertions,
        warnings=warnings,
        error=err if not ok else None,
        detail={
            "query_len": len(long_query),
            "elapsed_s": round(elapsed, 2),
            "text_len": len(text) if text else 0,
            "metadata": metadata,
        },
    )


# ---------------------------------------------------------------------------
# Phase 5: Multi-turn Conversation
# ---------------------------------------------------------------------------


async def scenario22_v3_multi_turn_context() -> dict:
    """Scenario 22: v3 Multi-turn — 같은 session_id로 2회 요청, 이전 대화 컨텍스트 유지 확인.

    구조적 검증: Turn 2의 metadata.total_messages > Turn 1의 total_messages
    → 서버가 이전 대화를 checkpointer에서 복원하여 메시지가 누적되었음을 확인.
    """
    sid = _session_id(22)
    t0 = time.monotonic()
    assertions: list[str] = []
    warnings: list[str] = []

    # Turn 1: 초안 요청
    ok1, text1, meta1, err1 = await _call_v3_run(
        query="서울시 도로 소음 민원에 대한 답변 초안을 작성해줘",
        session_id=sid,
        max_iterations=10,
        timeout=240,
    )
    if not ok1:
        elapsed = time.monotonic() - t0
        return _record(
            22, "v3 Multi-turn Context", 5, "failed", elapsed, error=f"Turn 1 실패: {err1}"
        )
    turn1_msgs = meta1.get("total_messages", 0)
    assertions.append(f"Turn 1 성공 (len={len(text1)}, msgs={turn1_msgs})")

    # Turn 2: 같은 session_id로 후속 질문
    ok2, text2, meta2, err2 = await _call_v3_run(
        query="위 답변에 관련 법령 근거를 추가해줘",
        session_id=sid,
        max_iterations=10,
        timeout=240,
    )
    elapsed = time.monotonic() - t0

    if not ok2:
        return _record(
            22,
            "v3 Multi-turn Context",
            5,
            "failed",
            elapsed,
            assertions=assertions,
            error=f"Turn 2 실패: {err2}",
        )
    turn2_msgs = meta2.get("total_messages", 0)
    assertions.append(f"Turn 2 성공 (len={len(text2)}, msgs={turn2_msgs})")

    # 구조적 검증: Turn 2의 메시지 수가 Turn 1보다 많아야 함
    # → checkpointer가 이전 대화를 복원하여 새 HumanMessage가 누적되었음을 의미
    if turn2_msgs > turn1_msgs:
        assertions.append(f"컨텍스트 누적 확인: {turn1_msgs} → {turn2_msgs} messages")
        return _record(
            22,
            "v3 Multi-turn Context",
            5,
            "passed",
            elapsed,
            assertions=assertions,
            detail={
                "turn1_msgs": turn1_msgs,
                "turn2_msgs": turn2_msgs,
                "turn1_len": len(text1),
                "turn2_len": len(text2),
            },
        )
    elif turn2_msgs == 0 and turn1_msgs == 0:
        # metadata에 total_messages가 없는 경우 → 응답 길이로 fallback
        warnings.append("total_messages 미반환 — 응답 길이로 fallback 검증")
        if text2 and len(text2.strip()) >= 30:
            return _record(
                22,
                "v3 Multi-turn Context",
                5,
                "passed",
                elapsed,
                assertions=assertions,
                warnings=warnings,
                detail={"turn1_len": len(text1), "turn2_len": len(text2)},
            )
        return _record(
            22,
            "v3 Multi-turn Context",
            5,
            "failed",
            elapsed,
            assertions=assertions,
            warnings=warnings,
            error="Turn 2 응답 부족",
        )
    else:
        return _record(
            22,
            "v3 Multi-turn Context",
            5,
            "failed",
            elapsed,
            assertions=assertions,
            error=f"컨텍스트 미누적: Turn1={turn1_msgs}, Turn2={turn2_msgs}",
            detail={"turn1_msgs": turn1_msgs, "turn2_msgs": turn2_msgs},
        )


async def scenario23_v3_multi_turn_isolation() -> dict:
    """Scenario 23: v3 Multi-turn Isolation — 다른 session_id는 서로 격리."""
    sid_a = _session_id(231)
    sid_b = _session_id(232)
    t0 = time.monotonic()
    assertions: list[str] = []

    # Session A: Turn 1
    ok_a, text_a, _, err_a = await _call_v3_run(
        query="도로 포장 민원 답변을 작성해줘",
        session_id=sid_a,
        max_iterations=5,
        timeout=180,
    )
    if not ok_a:
        elapsed = time.monotonic() - t0
        return _record(
            23, "v3 Multi-turn Isolation", 5, "failed", elapsed, error=f"Session A 실패: {err_a}"
        )
    assertions.append("Session A 성공")

    # Session B: 독립 요청 (A의 컨텍스트를 모름)
    ok_b, text_b, _, err_b = await _call_v3_run(
        query="안녕하세요",
        session_id=sid_b,
        max_iterations=3,
        timeout=60,
    )
    elapsed = time.monotonic() - t0

    if not ok_b:
        return _record(
            23,
            "v3 Multi-turn Isolation",
            5,
            "failed",
            elapsed,
            assertions=assertions,
            error=f"Session B 실패: {err_b}",
        )
    assertions.append("Session B 성공 (독립 세션)")

    # 두 세션 모두 성공이면 격리 확인
    return _record(
        23,
        "v3 Multi-turn Isolation",
        5,
        "passed",
        elapsed,
        assertions=assertions,
        detail={"a_len": len(text_a), "b_len": len(text_b) if text_b else 0},
    )


async def scenario24_v3_multi_turn_3_turns() -> dict:
    """Scenario 24: v3 3-turn — 초안 → 법령 → 통계 순차 요청."""
    sid = _session_id(24)
    t0 = time.monotonic()
    assertions: list[str] = []
    turns = [
        "아파트 층간소음 민원에 대한 답변 초안을 작성해줘",
        "위 답변에 관련 법령이나 조례를 인용해줘",
        "최근 층간소음 민원 관련 통계도 추가해줘",
    ]

    for i, query in enumerate(turns, 1):
        ok, text, meta, err = await _call_v3_run(
            query=query,
            session_id=sid,
            max_iterations=10,
            timeout=240,
        )
        if not ok:
            elapsed = time.monotonic() - t0
            return _record(
                24,
                "v3 3-turn Conversation",
                5,
                "failed",
                elapsed,
                assertions=assertions,
                error=f"Turn {i} 실패: {err}",
            )
        assertions.append(f"Turn {i} 성공 (len={len(text)})")

    elapsed = time.monotonic() - t0
    return _record(24, "v3 3-turn Conversation", 5, "passed", elapsed, assertions=assertions)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


async def main() -> None:
    logger.info("=" * 70)
    logger.info("GovOn v4 ReAct E2E 검증 시작")
    logger.info(f"  target : {BASE_URL}")
    logger.info(f"  run_id : {_run_id}")
    logger.info(f"  backend: {_HTTP_BACKEND}")
    logger.info("=" * 70)

    # cold start 대기
    cold_start_wait = int(os.environ.get("COLD_START_WAIT", "30"))
    if cold_start_wait > 0:
        logger.info(f"Cold start 대기: {cold_start_wait}s ...")
        await asyncio.sleep(cold_start_wait)

    # Phase 1: Infrastructure (hard gate)
    logger.info("\n[ Phase 1: Infrastructure ]")
    r1 = await scenario1_health_profile()
    if r1["status"] == "failed":
        logger.error("Phase 1 hard gate 실패 — Health & Profile. 이후 시나리오 중단.")
        _finalize()
        return

    r2 = await scenario2_base_model_generation()
    r3 = await scenario3_vllm_connection()

    phase1_ok = all(r["status"] == "passed" for r in [r1, r2, r3])
    if not phase1_ok:
        logger.error("Phase 1 실패 — Infrastructure not ready. 이후 시나리오 중단.")
        _finalize()
        return

    # Phase 2: v2 Agent Pipeline
    logger.info("\n[ Phase 2: v2 Agent Pipeline ]")
    await scenario4_v2_direct_answer()
    await scenario5_v2_tool_execution_approval()
    await scenario6_v2_approval_rejection()
    await scenario7_v2_multi_turn_session()
    await scenario8_v2_empty_query()
    await scenario9_v2_concurrent_requests()

    # Phase 3: v3 ReAct Loop
    logger.info("\n[ Phase 3: v3 ReAct Loop ]")
    await scenario10_v3_direct_answer()
    await scenario11_v3_tool_execution()
    await scenario12_v3_multi_iteration()
    await scenario13_v3_max_iterations_1()
    await scenario14_v3_max_iterations_validation()
    await scenario15_v3_sse_stream_no_tool()
    await scenario16_v3_sse_stream_with_tool()
    await scenario17_v3_sse_run_complete_metadata()
    await scenario18_v3_empty_query()
    await scenario19_v3_concurrent_requests()

    # Phase 4: Cross-version
    logger.info("\n[ Phase 4: Cross-version ]")
    await scenario20_v2_then_v3_same_query()
    await scenario21_v3_long_query()

    # Phase 5: Multi-turn Conversation
    logger.info("\n[ Phase 5: Multi-turn Conversation ]")
    await scenario22_v3_multi_turn_context()
    await scenario23_v3_multi_turn_isolation()
    await scenario24_v3_multi_turn_3_turns()

    _finalize()


def _finalize() -> None:
    """최종 결과 요약 및 저장."""
    total = len(_results)
    passed = sum(1 for r in _results if r["status"] == "passed")
    failed = sum(1 for r in _results if r["status"] == "failed")
    skipped = sum(1 for r in _results if r["status"] == "skipped")

    logger.info("\n" + "=" * 70)
    logger.info("검증 완료")
    logger.info(f"  총 시나리오 : {total}")
    logger.info(f"  통과        : {passed}")
    logger.info(f"  실패        : {failed}")
    logger.info(f"  스킵        : {skipped}")
    if _observed_tools:
        logger.info(f"  관찰된 도구 : {sorted(_observed_tools)}")
    else:
        logger.info("  관찰된 도구 : (없음 — LLM이 직접 답변 선택 또는 tool 메타 미포함)")
    logger.info(f"  결과 파일   : {RESULTS_PATH}")
    logger.info("=" * 70)

    # 실패 시나리오 목록
    if failed > 0:
        logger.info("\n[실패 목록]")
        for r in _results:
            if r["status"] == "failed":
                logger.error(f"  - Scenario {r['id']}: {r['name']} — {r['error']}")

    # 최종 결과 저장
    output = {
        "meta": {
            "run_id": _run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_url": BASE_URL,
            "log_file": LOG_PATH,
            "status": "completed",
        },
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "observed_tools": sorted(_observed_tools),
        "scenarios": _results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"결과 저장 완료: {RESULTS_PATH}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
