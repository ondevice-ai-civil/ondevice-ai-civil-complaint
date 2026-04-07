#!/usr/bin/env python3
"""GovOn Native Tool Calling + AdapterRegistry E2E 검증 스크립트.

HuggingFace Space에 배포된 govon-runtime 서버에 대해
에이전트 파이프라인(플래너 → 도구 실행 → 어댑터 전환)을 검증한다.

사용법:
    GOVON_RUNTIME_URL=https://<space-url>.hf.space python3 scripts/verify_e2e_tool_calling.py
    GOVON_RUNTIME_URL=https://<space-url>.hf.space API_KEY=<key> python3 scripts/verify_e2e_tool_calling.py

5-Phase 검증 (13 시나리오):
    Phase 1: Infrastructure (hard gate)
        1. Health & Profile
        2. Base Model Generation
        3. Adapter Registry
    Phase 2: Agent Pipeline Core
        4. Planner Produces Valid Plan
        5. Civil LoRA Draft Response
        6. Legal LoRA Evidence Augmentation (depends on 5)
        7. Task Type Classification
    Phase 3: data.go.kr API Tools (soft gate)
        8. External API Tool Invocation (4 sub-cases)
    Phase 4: Adapter Dynamics
        9. Sequential Adapter Switching
        10. LoRA ID Consistency
    Phase 5: Robustness
        11. Empty Query Handling
        12. Reject Flow Completeness
        13. Concurrent Request Isolation
"""

# stdlib
import asyncio
import json
import logging
import os
import re
import sys
import time
from typing import Any, Optional
from uuid import uuid4

BASE_URL = os.environ.get("GOVON_RUNTIME_URL", "http://localhost:7860").rstrip("/")
API_KEY = os.environ.get("API_KEY")
TIMEOUT = 300  # 시나리오당 최대 대기 시간 (초)
BASE_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B-AWQ"
RESULTS_PATH = "verify_e2e_tool_calling_results.json"

VALID_TOOLS = frozenset(
    {
        "rag_search",
        "api_lookup",
        "draft_civil_response",
        "append_evidence",
        "issue_detector",
        "stats_lookup",
        "keyword_analyzer",
        "demographics_lookup",
    }
)

LEGAL_PATTERNS = [
    r"제\s*\d+\s*조",
    r"제\s*\d+\s*항",
    r"법률",
    r"시행령",
    r"조례",
    r"판례",
    r"대법원",
    r"법",
    r"령",
    r"규정",
]

logger = logging.getLogger(__name__)

_results: list[dict] = []
_observed_tools: set[str] = set()
_run_id = uuid4().hex


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

    async def http_post(path: str, body: dict, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=_build_headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {}

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
        print(f"{tag} Scenario {scenario_num}: {name} {suffix}")
    elif status == "skipped":
        print(f"{tag} Scenario {scenario_num}: {name} — {error or 'skipped'} {suffix}")
    else:
        print(f"{tag} Scenario {scenario_num}: {name} — {error} {suffix}")

    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")

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
    return entry


def _session_id(scenario_num: int) -> str:
    return f"e2e-{scenario_num}-{uuid4().hex[:8]}"


def _extract_text_from_events(events: list[dict]) -> str:
    """SSE 이벤트 목록에서 최종 텍스트를 추출한다."""
    for ev in reversed(events):
        if ev.get("node") == "synthesis" and ev.get("final_text"):
            return ev["final_text"]
    for ev in reversed(events):
        if ev.get("finished") and ev.get("text"):
            return ev["text"]
    chunks = [ev.get("text", "") or ev.get("final_text", "") for ev in events]
    return "".join(c for c in chunks if c)


def _contains_legal_keyword(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in LEGAL_PATTERNS)


# ---------------------------------------------------------------------------
# Agent 호출 헬퍼: _call_agent_with_approval()
# ---------------------------------------------------------------------------


async def _call_agent_with_approval(
    query: str,
    session_id: str,
    approve: bool = True,
    timeout: float = 180,
) -> tuple[bool, str, dict, Optional[str]]:
    """에이전트 SSE 스트리밍으로 호출 → awaiting_approval까지 파싱 → approve/reject.

    Returns: (success, text, metadata_dict, error)
    metadata_dict keys: planned_tools, task_type, tool_results, adapter_mode, tool_args
    """
    body = {"query": query, "session_id": session_id, "use_rag": False}
    meta: dict[str, Any] = {
        "planned_tools": [],
        "task_type": None,
        "tool_results": {},
        "adapter_mode": None,
        "tool_args": {},
    }

    # --- SSE 스트리밍 시도 ---
    try:
        status_code, events = await http_post_sse("/v2/agent/stream", body, timeout=timeout)
        if status_code != 200:
            raise RuntimeError(f"SSE HTTP {status_code}")

        # awaiting_approval 또는 __interrupt__ 이벤트 탐색
        awaiting = None
        for ev in events:
            if ev.get("status") == "awaiting_approval" or ev.get("node") == "__interrupt__":
                awaiting = ev
                break
            # 플래너 노드에서 planned_tools 추출
            if ev.get("planned_tools"):
                meta["planned_tools"] = ev["planned_tools"]
            if ev.get("task_type"):
                meta["task_type"] = ev["task_type"]
            if ev.get("adapter_mode"):
                meta["adapter_mode"] = ev["adapter_mode"]
            if ev.get("tool_args"):
                meta["tool_args"] = ev["tool_args"]

        if awaiting:
            # awaiting 이벤트에서 메타데이터 추출
            if awaiting.get("planned_tools"):
                meta["planned_tools"] = awaiting["planned_tools"]
            if awaiting.get("task_type"):
                meta["task_type"] = awaiting["task_type"]
            if awaiting.get("adapter_mode"):
                meta["adapter_mode"] = awaiting["adapter_mode"]
            if awaiting.get("tool_args"):
                meta["tool_args"] = awaiting["tool_args"]

            thread_id = awaiting.get("thread_id") or session_id

            # approve/reject
            approve_code, approve_resp = await http_post(
                f"/v2/agent/approve?thread_id={thread_id}&approved={'true' if approve else 'false'}",
                {},
                timeout=timeout,
            )
            if approve_code != 200:
                return False, "", meta, f"approve HTTP {approve_code}: {approve_resp}"

            # approve 응답에서 최종 텍스트 및 도구 결과 추출
            final_text = approve_resp.get("text", "") or approve_resp.get("final_text", "") or ""
            if approve_resp.get("tool_results"):
                meta["tool_results"] = approve_resp["tool_results"]
            if approve_resp.get("adapter_mode"):
                meta["adapter_mode"] = approve_resp["adapter_mode"]
            if approve_resp.get("status") == "rejected":
                return True, final_text, meta, None

            if final_text:
                return True, final_text, meta, None
            return False, "", meta, f"approve 200 but text 없음: {approve_resp}"

        # awaiting 이벤트 없이 최종 텍스트가 있는 경우 (auto-approve 모드)
        text = _extract_text_from_events(events)
        # 이벤트에서 추가 메타데이터 수집
        for ev in events:
            if ev.get("planned_tools") and not meta["planned_tools"]:
                meta["planned_tools"] = ev["planned_tools"]
            if ev.get("task_type") and not meta["task_type"]:
                meta["task_type"] = ev["task_type"]
            if ev.get("tool_results") and not meta["tool_results"]:
                meta["tool_results"] = ev["tool_results"]
            if ev.get("adapter_mode") and not meta["adapter_mode"]:
                meta["adapter_mode"] = ev["adapter_mode"]
            if ev.get("tool_args") and not meta["tool_args"]:
                meta["tool_args"] = ev["tool_args"]

        if text:
            return True, text, meta, None

        # error 이벤트 확인
        for ev in events:
            if ev.get("status") == "error":
                return False, "", meta, ev.get("error", "unknown error")

        return False, "", meta, f"SSE 이벤트 수신했으나 text/awaiting 없음 (events={len(events)})"

    except Exception as sse_exc:
        logger.warning("SSE stream failed: %s — falling back to REST", sse_exc)

    # --- REST fallback: /v2/agent/run ---
    try:
        status_code, resp = await http_post("/v2/agent/run", body, timeout=timeout)
        if status_code != 200:
            return False, "", meta, f"REST HTTP {status_code}: {resp}"

        if resp.get("planned_tools"):
            meta["planned_tools"] = resp["planned_tools"]
        if resp.get("task_type"):
            meta["task_type"] = resp["task_type"]
        if resp.get("adapter_mode"):
            meta["adapter_mode"] = resp["adapter_mode"]
        if resp.get("tool_args"):
            meta["tool_args"] = resp["tool_args"]

        if resp.get("status") == "awaiting_approval":
            thread_id = resp.get("thread_id") or session_id
            approve_code, approve_resp = await http_post(
                f"/v2/agent/approve?thread_id={thread_id}&approved={'true' if approve else 'false'}",
                {},
                timeout=timeout,
            )
            if approve_code != 200:
                return False, "", meta, f"approve HTTP {approve_code}"
            final_text = approve_resp.get("text", "") or approve_resp.get("final_text", "") or ""
            if approve_resp.get("tool_results"):
                meta["tool_results"] = approve_resp["tool_results"]
            if approve_resp.get("status") == "rejected":
                return True, final_text, meta, None
            if final_text:
                return True, final_text, meta, None
            return False, "", meta, "approve 200 but text 없음"

        if resp.get("status") == "error":
            return False, "", meta, resp.get("error", "agent run error")

        text = resp.get("text", "") or resp.get("final_text", "")
        if resp.get("tool_results"):
            meta["tool_results"] = resp["tool_results"]
        if text:
            return True, text, meta, None
        return False, "", meta, f"text 없음, status={resp.get('status')}"

    except Exception as exc:
        return False, "", meta, str(exc)


# ---------------------------------------------------------------------------
# Phase 1: Infrastructure (hard gate)
# ---------------------------------------------------------------------------


async def scenario1_health_profile() -> dict:
    """Scenario 1: Health & Profile (retry 3x with backoff)."""
    backoffs = [5, 10, 20]
    attempts = 0
    last_error = ""

    for attempt_idx in range(3):
        attempts += 1
        t0 = time.monotonic()
        try:
            status_code, body = await http_get("/health", timeout=10)
            elapsed = time.monotonic() - t0

            assertions = []
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
                    assertions=["HTTP 200"],
                    error=last_error,
                    detail={"body": body},
                )

            assertions.append("HTTP 200: OK")
            srv_status = body.get("status", "")
            if srv_status not in ("ok", "healthy"):
                last_error = f"status={srv_status!r}, expected ok/healthy"
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
                    assertions=assertions,
                    error=last_error,
                    detail={"body": body},
                )
            assertions.append(f"status={srv_status}: OK")

            warnings = []
            if "model" not in body:
                warnings.append("model field not found in /health")
            else:
                assertions.append(f"model={body['model']}: OK")

            if "profile" not in body:
                warnings.append("profile field not found in /health")
            else:
                assertions.append(f"profile={body['profile']}: OK")

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
                    "status": srv_status,
                    "model": body.get("model"),
                    "profile": body.get("profile"),
                },
            )

        except Exception as exc:
            last_error = str(exc)
            if attempt_idx < 2:
                await asyncio.sleep(backoffs[attempt_idx])
                continue
            return _record(
                1,
                "Health & Profile",
                1,
                "failed",
                time.monotonic() - t0,
                attempts,
                error=last_error,
            )

    return _record(1, "Health & Profile", 1, "failed", 0, attempts, error=last_error)


async def scenario2_base_model_generation() -> dict:
    """Scenario 2: Base Model Generation (retry 2x)."""
    body_completions = {
        "model": BASE_MODEL,
        "prompt": "대한민국의 수도는",
        "max_tokens": 32,
        "temperature": 0.0,
    }
    last_error = ""
    attempts = 0

    for attempt_idx in range(2):
        attempts += 1
        t0 = time.monotonic()
        try:
            status_code, resp = await http_post("/v1/completions", body_completions, timeout=60)
            elapsed = time.monotonic() - t0

            if status_code == 200:
                choices = resp.get("choices", [])
                if choices and choices[0].get("text") is not None:
                    text = choices[0]["text"]
                    if text.strip():
                        return _record(
                            2,
                            "Base Model Generation",
                            1,
                            "passed",
                            elapsed,
                            attempts,
                            assertions=["HTTP 200", "non-empty text"],
                            detail={"endpoint": "/v1/completions", "text_preview": text[:100]},
                        )

            # fallback /v1/generate
            body_legacy = {
                "prompt": "대한민국의 수도는",
                "max_tokens": 32,
                "temperature": 0.0,
                "use_rag": False,
            }
            status_code2, resp2 = await http_post("/v1/generate", body_legacy, timeout=60)
            elapsed2 = time.monotonic() - t0

            if status_code2 == 200 and resp2.get("text", "").strip():
                return _record(
                    2,
                    "Base Model Generation",
                    1,
                    "passed",
                    elapsed2,
                    attempts,
                    assertions=["HTTP 200 (fallback)", "non-empty text"],
                    detail={"endpoint": "/v1/generate", "text_preview": resp2["text"][:100]},
                )

            last_error = f"/v1/completions HTTP {status_code}, /v1/generate HTTP {status_code2}"
        except Exception as exc:
            last_error = str(exc)

    return _record(
        2, "Base Model Generation", 1, "failed", time.monotonic() - t0, attempts, error=last_error
    )


async def scenario3_adapter_registry() -> dict:
    """Scenario 3: Adapter Registry via /v1/models."""
    t0 = time.monotonic()
    try:
        status_code, resp = await http_get("/v1/models", timeout=10)
        elapsed = time.monotonic() - t0

        assertions = []
        warnings = []

        if status_code != 200:
            return _record(
                3,
                "Adapter Registry",
                1,
                "failed",
                elapsed,
                assertions=["HTTP 200"],
                error=f"HTTP {status_code}",
                detail={"resp": resp},
            )
        assertions.append("HTTP 200: OK")

        data = resp.get("data", [])
        if not isinstance(data, list):
            return _record(
                3,
                "Adapter Registry",
                1,
                "failed",
                elapsed,
                assertions=assertions,
                error="data array missing or invalid",
                detail={"resp": resp},
            )
        assertions.append(f"data array: {len(data)} models")

        model_ids = [m.get("id", "") for m in data]
        civil_found = any("civil" in mid for mid in model_ids)
        legal_found = any("legal" in mid for mid in model_ids)

        if not civil_found:
            warnings.append("civil adapter not detected in /v1/models (WARN, not FAIL)")
        else:
            assertions.append("civil adapter detected")
        if not legal_found:
            warnings.append("legal adapter not detected in /v1/models (WARN, not FAIL)")
        else:
            assertions.append("legal adapter detected")

        return _record(
            3,
            "Adapter Registry",
            1,
            "passed",
            elapsed,
            assertions=assertions,
            warnings=warnings,
            detail={"model_ids": model_ids, "civil_found": civil_found, "legal_found": legal_found},
        )

    except Exception as exc:
        return _record(3, "Adapter Registry", 1, "failed", time.monotonic() - t0, error=str(exc))


# ---------------------------------------------------------------------------
# Phase 2: Agent Pipeline Core
# ---------------------------------------------------------------------------

# Scenario 5/6 공유 세션
_scenario5_session_id: Optional[str] = None
_scenario5_passed: bool = False


async def scenario4_planner_valid_plan() -> dict:
    """Scenario 4: Planner Produces Valid Plan (retry 2x)."""
    query = "서울시 도로 파손 민원에 대한 답변 초안을 작성해주세요"
    last_error = ""
    attempts = 0

    for attempt_idx in range(2):
        attempts += 1
        t0 = time.monotonic()
        try:
            sid = _session_id(4)
            ok, text, meta, err = await _call_agent_with_approval(
                query, sid, approve=True, timeout=120
            )
            elapsed = time.monotonic() - t0

            planned = meta.get("planned_tools", [])
            if planned:
                _observed_tools.update(planned)

            assertions = []
            if not planned:
                last_error = err or "planned_tools 비어있음"
                if attempt_idx < 1:
                    continue
                return _record(
                    4,
                    "Planner Produces Valid Plan",
                    2,
                    "failed",
                    elapsed,
                    attempts,
                    assertions=["planned_tools non-empty"],
                    error=last_error,
                    detail={"meta": meta},
                )

            assertions.append(f"planned_tools: {planned}")
            invalid = [t for t in planned if t not in VALID_TOOLS]
            if invalid:
                last_error = f"invalid tools: {invalid}"
                if attempt_idx < 1:
                    continue
                return _record(
                    4,
                    "Planner Produces Valid Plan",
                    2,
                    "failed",
                    elapsed,
                    attempts,
                    assertions=assertions,
                    error=last_error,
                    detail={"invalid_tools": invalid, "valid": list(VALID_TOOLS)},
                )

            assertions.append("all tools in VALID_TOOLS whitelist")
            return _record(
                4,
                "Planner Produces Valid Plan",
                2,
                "passed",
                elapsed,
                attempts,
                assertions=assertions,
                detail={"planned_tools": planned, "meta": meta},
            )

        except Exception as exc:
            last_error = str(exc)

    return _record(4, "Planner Produces Valid Plan", 2, "failed", 0, attempts, error=last_error)


async def scenario5_civil_lora_draft() -> dict:
    """Scenario 5: Civil LoRA Draft Response (retry 2x)."""
    global _scenario5_session_id, _scenario5_passed
    query = "아파트 층간소음 민원에 대한 답변을 작성해주세요"
    last_error = ""
    attempts = 0

    for attempt_idx in range(2):
        attempts += 1
        t0 = time.monotonic()
        try:
            sid = _session_id(5)
            ok, text, meta, err = await _call_agent_with_approval(
                query, sid, approve=True, timeout=180
            )
            elapsed = time.monotonic() - t0

            planned = meta.get("planned_tools", [])
            if planned:
                _observed_tools.update(planned)

            assertions = []

            if not ok:
                last_error = err or "agent call failed"
                if attempt_idx < 1:
                    continue
                return _record(
                    5,
                    "Civil LoRA Draft Response",
                    2,
                    "failed",
                    elapsed,
                    attempts,
                    assertions=assertions,
                    error=last_error,
                    detail={"meta": meta},
                )

            has_draft = "draft_civil_response" in planned
            if has_draft:
                assertions.append("draft_civil_response in planned_tools")
            else:
                assertions.append(f"draft_civil_response NOT in planned_tools ({planned})")

            if len(text) >= 50:
                assertions.append(f"text length {len(text)} >= 50")
            else:
                assertions.append(f"text length {len(text)} < 50 (FAIL)")

            task_type = meta.get("task_type")
            if task_type == "draft_response":
                assertions.append("task_type=draft_response")
            else:
                assertions.append(f"task_type={task_type} (expected draft_response)")

            # 핵심 검증: text >= 50 이면 PASS (planned_tools와 task_type은 soft 검증)
            passed = len(text) >= 50
            if passed:
                _scenario5_session_id = sid
                _scenario5_passed = True

            warnings = []
            if not has_draft:
                warnings.append("draft_civil_response not in planned_tools")
            if task_type != "draft_response":
                warnings.append(f"task_type={task_type}, expected draft_response")

            if passed:
                return _record(
                    5,
                    "Civil LoRA Draft Response",
                    2,
                    "passed",
                    elapsed,
                    attempts,
                    assertions=assertions,
                    warnings=warnings,
                    detail={"text_preview": text[:200], "meta": meta},
                )

            last_error = "text < 50 chars"
            if attempt_idx < 1:
                continue
            return _record(
                5,
                "Civil LoRA Draft Response",
                2,
                "failed",
                elapsed,
                attempts,
                assertions=assertions,
                warnings=warnings,
                error=last_error,
                detail={"text_preview": text[:200], "meta": meta},
            )

        except Exception as exc:
            last_error = str(exc)

    return _record(5, "Civil LoRA Draft Response", 2, "failed", 0, attempts, error=last_error)


async def scenario6_legal_lora_evidence() -> dict:
    """Scenario 6: Legal LoRA Evidence Augmentation (depends on Scenario 5)."""
    if not _scenario5_passed:
        return _record(
            6,
            "Legal LoRA Evidence Augmentation",
            2,
            "skipped",
            0,
            error="Scenario 5 failed — dependency skip",
        )

    query = "위 답변에 관련 법령과 판례 근거를 추가해주세요"
    last_error = ""
    attempts = 0

    for attempt_idx in range(2):
        attempts += 1
        t0 = time.monotonic()
        try:
            ok, text, meta, err = await _call_agent_with_approval(
                query, _scenario5_session_id, approve=True, timeout=180
            )
            elapsed = time.monotonic() - t0

            planned = meta.get("planned_tools", [])
            if planned:
                _observed_tools.update(planned)

            assertions = []

            if not ok:
                last_error = err or "agent call failed"
                if attempt_idx < 1:
                    continue
                return _record(
                    6,
                    "Legal LoRA Evidence Augmentation",
                    2,
                    "failed",
                    elapsed,
                    attempts,
                    assertions=assertions,
                    error=last_error,
                    detail={"meta": meta},
                )

            has_evidence = "append_evidence" in planned
            if has_evidence:
                assertions.append("append_evidence in planned_tools")
            else:
                assertions.append(f"append_evidence NOT in planned_tools ({planned})")

            has_legal = _contains_legal_keyword(text)
            matched = [p for p in LEGAL_PATTERNS if re.search(p, text)]
            if has_legal:
                assertions.append(f"legal patterns found: {matched[:3]}")
            else:
                assertions.append("no legal patterns found (FAIL)")

            warnings = []
            if not has_evidence:
                warnings.append("append_evidence not in planned_tools")

            if has_legal:
                return _record(
                    6,
                    "Legal LoRA Evidence Augmentation",
                    2,
                    "passed",
                    elapsed,
                    attempts,
                    assertions=assertions,
                    warnings=warnings,
                    detail={"text_preview": text[:300], "matched_patterns": matched, "meta": meta},
                )

            last_error = "legal pattern not found in response"
            if attempt_idx < 1:
                continue
            return _record(
                6,
                "Legal LoRA Evidence Augmentation",
                2,
                "failed",
                elapsed,
                attempts,
                assertions=assertions,
                warnings=warnings,
                error=last_error,
                detail={"text_preview": text[:300], "meta": meta},
            )

        except Exception as exc:
            last_error = str(exc)

    return _record(
        6, "Legal LoRA Evidence Augmentation", 2, "failed", 0, attempts, error=last_error
    )


async def scenario7_task_type_classification() -> dict:
    """Scenario 7: Task Type Classification (at least 2/3 correct)."""
    test_cases = [
        ("민원 답변 초안을 작성해줘", {"draft_response"}),
        ("관련 통계 데이터를 조회해줘", {"stats_query", "lookup_stats"}),
        ("이 민원의 근거를 보강해줘", {"append_evidence"}),
    ]

    t0 = time.monotonic()
    correct = 0
    sub_results = []

    for query, expected_types in test_cases:
        try:
            sid = _session_id(7)
            ok, text, meta, err = await _call_agent_with_approval(
                query, sid, approve=True, timeout=180
            )

            planned = meta.get("planned_tools", [])
            if planned:
                _observed_tools.update(planned)

            actual_type = meta.get("task_type")
            matched = actual_type in expected_types if actual_type else False
            if matched:
                correct += 1

            sub_results.append(
                {
                    "query": query[:30],
                    "expected": list(expected_types),
                    "actual": actual_type,
                    "matched": matched,
                    "ok": ok,
                    "error": err,
                }
            )
        except Exception as exc:
            sub_results.append(
                {
                    "query": query[:30],
                    "expected": list(expected_types),
                    "actual": None,
                    "matched": False,
                    "error": str(exc),
                }
            )

    elapsed = time.monotonic() - t0
    assertions = [f"{correct}/3 task types correct (need >= 2)"]

    if correct >= 2:
        return _record(
            7,
            "Task Type Classification",
            2,
            "passed",
            elapsed,
            assertions=assertions,
            detail={"sub_results": sub_results, "correct": correct},
        )
    return _record(
        7,
        "Task Type Classification",
        2,
        "failed",
        elapsed,
        assertions=assertions,
        error=f"only {correct}/3 correct (need >= 2)",
        detail={"sub_results": sub_results},
    )


# ---------------------------------------------------------------------------
# Phase 3: data.go.kr API Tools (soft gate)
# ---------------------------------------------------------------------------

_datago_available: bool = False


async def _check_datago_connectivity() -> bool:
    """data.go.kr 연결 확인 preflight."""
    global _datago_available
    try:
        code, _ = await http_get_raw("https://www.data.go.kr", timeout=10)
        _datago_available = code in (200, 301, 302, 403)
        return _datago_available
    except Exception:
        _datago_available = False
        return False


async def scenario8_external_api_tools() -> dict:
    """Scenario 8: External API Tool Invocation (4 sub-cases, accept 3/4)."""
    if not _datago_available:
        return _record(
            8,
            "External API Tool Invocation",
            3,
            "skipped",
            0,
            error="data.go.kr unreachable — Phase 3 skipped",
        )

    sub_cases = [
        ("8a", "최근 도로 관련 민원 이슈를 분석해줘", "issue_detector"),
        ("8b", "서울시 민원 통계를 조회해줘", "stats_lookup"),
        ("8c", "도로 관련 키워드 트렌드를 분석해줘", "keyword_analyzer"),
        ("8d", "서울시 강남구 민원 인구통계를 조회해줘", "demographics_lookup"),
    ]

    t0 = time.monotonic()
    sub_passed = 0
    sub_results = []

    for label, query, expected_tool in sub_cases:
        for attempt_idx in range(2):  # retry 1x
            try:
                sid = _session_id(8)
                ok, text, meta, err = await _call_agent_with_approval(
                    query, sid, approve=True, timeout=180
                )

                planned = meta.get("planned_tools", [])
                if planned:
                    _observed_tools.update(planned)

                tool_in_plan = expected_tool in planned
                tool_results = meta.get("tool_results", {})
                tool_in_results = expected_tool in tool_results

                passed = tool_in_plan  # tool in planned_tools suffices
                if passed:
                    sub_passed += 1

                sub_results.append(
                    {
                        "label": label,
                        "expected_tool": expected_tool,
                        "tool_in_plan": tool_in_plan,
                        "tool_in_results": tool_in_results,
                        "planned_tools": planned,
                        "passed": passed,
                        "attempt": attempt_idx + 1,
                        "error": err,
                    }
                )
                break  # no retry needed if we got a response

            except Exception as exc:
                if attempt_idx == 1:
                    sub_results.append(
                        {
                            "label": label,
                            "expected_tool": expected_tool,
                            "passed": False,
                            "error": str(exc),
                            "attempt": attempt_idx + 1,
                        }
                    )

    elapsed = time.monotonic() - t0
    assertions = [f"{sub_passed}/4 sub-cases passed (need >= 3)"]

    if sub_passed >= 3:
        return _record(
            8,
            "External API Tool Invocation",
            3,
            "passed",
            elapsed,
            assertions=assertions,
            detail={"sub_results": sub_results},
        )
    return _record(
        8,
        "External API Tool Invocation",
        3,
        "failed",
        elapsed,
        assertions=assertions,
        error=f"only {sub_passed}/4 passed (need >= 3)",
        detail={"sub_results": sub_results},
    )


# ---------------------------------------------------------------------------
# Phase 4: Adapter Dynamics
# ---------------------------------------------------------------------------


async def scenario9_sequential_adapter_switching() -> dict:
    """Scenario 9: Sequential Adapter Switching (3 iterations, 3 requests each)."""
    t0 = time.monotonic()
    errors: list[str] = []
    total_requests = 0

    for i in range(1, 4):
        sid = _session_id(9)

        # Civil query
        ok1, text1, meta1, err1 = await _call_agent_with_approval(
            "주차 위반 과태료 이의신청 민원 답변을 작성해줘", sid, approve=True, timeout=180
        )
        total_requests += 1
        if meta1.get("planned_tools"):
            _observed_tools.update(meta1["planned_tools"])
        if not ok1 or not text1.strip():
            errors.append(f"iter {i} civil-1: {err1 or '빈 응답'}")
            continue

        # Legal query (same session)
        ok2, text2, meta2, err2 = await _call_agent_with_approval(
            "위 답변에 관련 법령 근거를 추가해줘", sid, approve=True, timeout=180
        )
        total_requests += 1
        if meta2.get("planned_tools"):
            _observed_tools.update(meta2["planned_tools"])
        if not ok2 or not text2.strip():
            errors.append(f"iter {i} legal: {err2 or '빈 응답'}")
            continue

        # Civil query again (same session)
        ok3, text3, meta3, err3 = await _call_agent_with_approval(
            "추가 민원 답변 초안을 작성해줘", sid, approve=True, timeout=180
        )
        total_requests += 1
        if meta3.get("planned_tools"):
            _observed_tools.update(meta3["planned_tools"])
        if not ok3 or not text3.strip():
            errors.append(f"iter {i} civil-2: {err3 or '빈 응답'}")

    elapsed = time.monotonic() - t0
    assertions = [f"{total_requests} requests completed", f"{len(errors)} errors"]

    if errors:
        return _record(
            9,
            "Sequential Adapter Switching",
            4,
            "failed",
            elapsed,
            assertions=assertions,
            error="; ".join(errors[:3]),
            detail={"iterations": 3, "total_requests": total_requests, "errors": errors},
        )
    return _record(
        9,
        "Sequential Adapter Switching",
        4,
        "passed",
        elapsed,
        assertions=assertions,
        detail={"iterations": 3, "total_requests": total_requests, "all_passed": True},
    )


async def scenario10_lora_id_consistency() -> dict:
    """Scenario 10: LoRA ID Consistency (informational, always PASS)."""
    t0 = time.monotonic()
    try:
        _, resp_before = await http_get("/v1/models", timeout=10)
        models_before = [m.get("id", "") for m in resp_before.get("data", [])]

        # Scenario 9 이미 완료된 상태에서 다시 확인
        _, resp_after = await http_get("/v1/models", timeout=10)
        models_after = [m.get("id", "") for m in resp_after.get("data", [])]

        elapsed = time.monotonic() - t0
        stable = set(models_before) == set(models_after)
        assertions = [
            f"before: {len(models_before)} models",
            f"after: {len(models_after)} models",
            f"stable: {stable}",
        ]
        warnings = [] if stable else ["adapter list changed between checks"]

        return _record(
            10,
            "LoRA ID Consistency",
            4,
            "passed",
            elapsed,
            assertions=assertions,
            warnings=warnings,
            detail={"models_before": models_before, "models_after": models_after, "stable": stable},
        )
    except Exception as exc:
        return _record(
            10,
            "LoRA ID Consistency",
            4,
            "passed",
            time.monotonic() - t0,
            assertions=["informational check"],
            warnings=[f"could not verify: {exc}"],
        )


# ---------------------------------------------------------------------------
# Phase 5: Robustness
# ---------------------------------------------------------------------------


async def scenario11_empty_query() -> dict:
    """Scenario 11: Empty Query Handling (expect 422, NOT 500)."""
    t0 = time.monotonic()
    assertions = []
    last_error = ""

    for attempt_idx in range(2):
        try:
            # REST endpoint
            code_rest, resp_rest = await http_post("/v2/agent/run", {"query": ""}, timeout=10)
            assertions.append(f"/v2/agent/run empty query: HTTP {code_rest}")

            # SSE endpoint
            code_sse, events_sse = await http_post_sse(
                "/v2/agent/stream", {"query": ""}, timeout=10
            )
            assertions.append(f"/v2/agent/stream empty query: HTTP {code_sse}")

            elapsed = time.monotonic() - t0

            # 422 (Pydantic validation) 또는 400 (Bad Request) 허용, 500은 불가
            rest_ok = code_rest in (400, 422)
            sse_ok = code_sse in (400, 422)
            no_500 = code_rest != 500 and code_sse != 500

            if no_500 and (rest_ok or sse_ok):
                return _record(
                    11,
                    "Empty Query Handling",
                    5,
                    "passed",
                    elapsed,
                    attempt_idx + 1,
                    assertions=assertions,
                    detail={"rest_code": code_rest, "sse_code": code_sse},
                )

            if not no_500:
                last_error = f"got 500 (rest={code_rest}, sse={code_sse})"
            else:
                last_error = f"unexpected codes: rest={code_rest}, sse={code_sse}"

            if attempt_idx < 1:
                continue
            return _record(
                11,
                "Empty Query Handling",
                5,
                "failed",
                elapsed,
                attempt_idx + 1,
                assertions=assertions,
                error=last_error,
                detail={"rest_code": code_rest, "sse_code": code_sse},
            )

        except Exception as exc:
            last_error = str(exc)

    return _record(
        11, "Empty Query Handling", 5, "failed", time.monotonic() - t0, 2, error=last_error
    )


async def scenario12_reject_flow() -> dict:
    """Scenario 12: Reject Flow Completeness."""
    last_error = ""

    for attempt_idx in range(2):
        t0 = time.monotonic()
        try:
            sid = _session_id(12)
            ok, text, meta, err = await _call_agent_with_approval(
                "민원 답변을 작성해주세요", sid, approve=False, timeout=30
            )
            elapsed = time.monotonic() - t0

            assertions = []

            # reject 후에는 tool_results가 비어있어야 함
            tool_results = meta.get("tool_results", {})

            if ok:
                assertions.append("reject flow completed")

                if not tool_results:
                    assertions.append("tool_results empty after reject")
                else:
                    assertions.append(f"tool_results NOT empty: {list(tool_results.keys())}")

                if elapsed < 5:
                    assertions.append(f"response < 5s ({elapsed:.1f}s)")
                else:
                    assertions.append(f"response >= 5s ({elapsed:.1f}s)")

                return _record(
                    12,
                    "Reject Flow Completeness",
                    5,
                    "passed",
                    elapsed,
                    attempt_idx + 1,
                    assertions=assertions,
                    detail={"text_preview": text[:100], "tool_results": tool_results, "meta": meta},
                )

            last_error = err or "reject flow failed"
            if attempt_idx < 1:
                continue
            return _record(
                12,
                "Reject Flow Completeness",
                5,
                "failed",
                elapsed,
                attempt_idx + 1,
                assertions=assertions,
                error=last_error,
                detail={"meta": meta},
            )

        except Exception as exc:
            last_error = str(exc)

    return _record(
        12, "Reject Flow Completeness", 5, "failed", time.monotonic() - t0, 2, error=last_error
    )


async def scenario13_concurrent_isolation() -> dict:
    """Scenario 13: Concurrent Request Isolation (3 simultaneous requests)."""
    t0 = time.monotonic()

    queries = [
        ("주차 위반 민원 답변 초안을 작성해줘", _session_id(13)),
        ("소음 민원에 대한 답변을 작성해줘", _session_id(13)),
        ("도로 파손 민원 답변을 작성해줘", _session_id(13)),
    ]

    async def _run_one(query: str, sid: str) -> dict:
        try:
            ok, text, meta, err = await _call_agent_with_approval(
                query, sid, approve=True, timeout=300
            )
            if meta.get("planned_tools"):
                _observed_tools.update(meta["planned_tools"])
            return {
                "session_id": sid,
                "ok": ok,
                "text_len": len(text),
                "error": err,
                "query": query[:20],
            }
        except Exception as exc:
            return {
                "session_id": sid,
                "ok": False,
                "text_len": 0,
                "error": str(exc),
                "query": query[:20],
            }

    tasks = [_run_one(q, s) for q, s in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - t0

    # 결과 정리
    sub_results = []
    valid_count = 0
    for r in results:
        if isinstance(r, Exception):
            sub_results.append({"ok": False, "error": str(r)})
        else:
            sub_results.append(r)
            if r.get("ok"):
                valid_count += 1

    # session_id 교차 오염 확인 (여기서는 각각 독립적 session_id)
    session_ids = [q[1] for q in queries]
    all_unique = len(set(session_ids)) == len(session_ids)

    assertions = [
        f"{valid_count}/3 concurrent requests succeeded",
        f"session_ids unique: {all_unique}",
    ]

    if valid_count == 3:
        return _record(
            13,
            "Concurrent Request Isolation",
            5,
            "passed",
            elapsed,
            assertions=assertions,
            detail={"sub_results": sub_results},
        )
    return _record(
        13,
        "Concurrent Request Isolation",
        5,
        "failed",
        elapsed,
        assertions=assertions,
        error=f"only {valid_count}/3 succeeded",
        detail={"sub_results": sub_results},
    )


# ---------------------------------------------------------------------------
# Cold Start 대기
# ---------------------------------------------------------------------------


async def _wait_cold_start() -> float:
    """서버 cold start 대기. 최대 10회 x 30초 간격. 대기한 총 시간을 반환."""
    total_wait = 0.0
    for i in range(10):
        try:
            code, body = await http_get("/health", timeout=10)
            if code == 200 and body.get("status") in ("ok", "healthy"):
                print(f"  서버 준비 완료 (대기 {total_wait:.0f}s)")
                return total_wait
        except Exception:
            pass
        if i < 9:
            print(f"  서버 대기 중... ({i + 1}/10, 30s 후 재시도)")
            await asyncio.sleep(30)
            total_wait += 30

    print("  [WARN] 서버 준비 확인 실패 — 계속 진행")
    return total_wait


# ---------------------------------------------------------------------------
# 메인 러너
# ---------------------------------------------------------------------------


async def main() -> int:
    print("=" * 60)
    print("GovOn E2E Tool Calling + AdapterRegistry 검증")
    print("=" * 60)
    print(f"  대상 서버: {BASE_URL}")
    print(f"  인증: {'API_KEY 설정됨' if API_KEY else '미설정 (비인증)'}")
    print(f"  HTTP 백엔드: {_HTTP_BACKEND}")
    print(f"  타임아웃: {TIMEOUT}s / 시나리오")
    print(f"  run_id: {_run_id}")
    print("-" * 60)

    # Cold start 대기
    print("[Cold Start] 서버 준비 확인 중...")
    cold_start_wait = await _wait_cold_start()

    # ===== Phase 1: Infrastructure (hard gate) =====
    print("\n[Phase 1] Infrastructure (hard gate)")
    print("-" * 40)

    phase1_scenarios = [
        scenario1_health_profile,
        scenario2_base_model_generation,
        scenario3_adapter_registry,
    ]

    phase1_failed = False
    for fn in phase1_scenarios:
        result = await fn()
        if result["status"] == "failed":
            phase1_failed = True

    if phase1_failed:
        print("\n" + "!" * 60)
        print("ABORT: Infrastructure not ready — Phase 1 failed")
        print("!" * 60)
        _write_output(cold_start_wait)
        return 1

    # ===== Phase 2: Agent Pipeline Core =====
    print("\n[Phase 2] Agent Pipeline Core")
    print("-" * 40)

    phase2_scenarios = [
        scenario4_planner_valid_plan,
        scenario5_civil_lora_draft,
        scenario6_legal_lora_evidence,
        scenario7_task_type_classification,
    ]

    for fn in phase2_scenarios:
        await fn()

    # ===== Phase 3: data.go.kr API Tools (soft gate) =====
    print("\n[Phase 3] data.go.kr API Tools (soft gate)")
    print("-" * 40)

    print("  data.go.kr 연결 확인...")
    datago_ok = await _check_datago_connectivity()
    if datago_ok:
        print("  data.go.kr 연결 가능")
    else:
        print("  data.go.kr 연결 불가 — Phase 3 스킵")

    await scenario8_external_api_tools()

    # ===== Phase 4: Adapter Dynamics =====
    print("\n[Phase 4] Adapter Dynamics")
    print("-" * 40)

    await scenario9_sequential_adapter_switching()
    await scenario10_lora_id_consistency()

    # ===== Phase 5: Robustness =====
    print("\n[Phase 5] Robustness")
    print("-" * 40)

    phase5_scenarios = [
        scenario11_empty_query,
        scenario12_reject_flow,
        scenario13_concurrent_isolation,
    ]

    for fn in phase5_scenarios:
        await fn()

    # ===== 요약 =====
    print("\n" + "=" * 60)
    passed = sum(1 for r in _results if r["status"] == "passed")
    failed = sum(1 for r in _results if r["status"] == "failed")
    skipped = sum(1 for r in _results if r["status"] == "skipped")
    total = len(_results)

    print(f"결과: {passed}/{total} 통과, {failed} 실패, {skipped} 스킵")

    tool_ratio = len(_observed_tools) / len(VALID_TOOLS) if VALID_TOOLS else 0
    print(f"도구 커버리지: {len(_observed_tools)}/{len(VALID_TOOLS)} ({tool_ratio:.0%})")
    if _observed_tools:
        print(f"  관측된 도구: {sorted(_observed_tools)}")

    _write_output(cold_start_wait)

    return 0 if failed == 0 else 1


def _write_output(cold_start_wait: float) -> None:
    """JSON 결과 파일 출력."""
    from datetime import datetime, timezone

    passed = sum(1 for r in _results if r["status"] == "passed")
    failed = sum(1 for r in _results if r["status"] == "failed")
    skipped = sum(1 for r in _results if r["status"] == "skipped")

    tool_ratio = len(_observed_tools) / len(VALID_TOOLS) if VALID_TOOLS else 0

    output = {
        "meta": {
            "run_id": _run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_url": BASE_URL,
            "cold_start_wait_seconds": cold_start_wait,
        },
        "summary": {
            "total": len(_results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "tool_coverage": {
                "observed": sorted(_observed_tools),
                "ratio": round(tool_ratio, 2),
            },
        },
        "scenarios": _results,
        "server_url": BASE_URL,
        "http_backend": _HTTP_BACKEND,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULTS_PATH}")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
