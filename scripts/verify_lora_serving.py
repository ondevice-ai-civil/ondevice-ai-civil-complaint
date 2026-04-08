#!/usr/bin/env python3
"""GovOn Legal LoRA 어댑터 서빙 통합 검증 스크립트.

HuggingFace Space에 배포된 govon-runtime 서버에 대해
legal/public_admin adapter Multi-LoRA 서빙 동작을 검증한다.

사용법:
    GOVON_RUNTIME_URL=https://<space-url>.hf.space python3 scripts/verify_lora_serving.py
    GOVON_RUNTIME_URL=https://<space-url>.hf.space API_KEY=<key> python3 scripts/verify_lora_serving.py

엔드포인트 참고 (src/inference/api_server.py):
    GET  /health              — 서버 상태 확인 (status: "healthy")
    POST /v1/completions      — OpenAI-compatible (vLLM 직접 제공)
    POST /v1/generate         — GovOn 레거시 생성 엔드포인트
    POST /v2/agent/run        — LangGraph agent (REST, interrupt까지 실행)
    POST /v2/agent/stream     — LangGraph agent (SSE 스트리밍)
    GET  /v1/models           — OpenAI-compatible 모델 목록 (vLLM 직접 제공)

AgentRunRequest 필드:
    query: str          — 사용자 입력 (필수)
    session_id: str     — 세션 식별자 (선택)
    stream: bool        — 스트리밍 여부 (기본값 False)
    force_tools: list   — 강제 실행 도구 목록 (선택)
    max_tokens: int     — 최대 토큰 수 (기본값 512)
    temperature: float  — 온도 (기본값 0.7)
    use_rag: bool       — RAG 사용 여부 (기본값 True)
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
RESULTS_PATH = "verify_results.json"

logger = logging.getLogger(__name__)

# 법령 관련 패턴 (Scenario 4 검증용) — regex 기반, 단일 문자 제외
LEGAL_PATTERNS = [
    r"제\s*\d+\s*조",
    r"제\s*\d+\s*항",
    r"법률",
    r"시행령",
    r"조례",
    r"판례",
    r"대법원",
]

_results: list[dict] = []


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

    async def http_get(path: str) -> tuple[int, dict]:
        url = BASE_URL + path
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, headers=_build_headers())
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"_raw": resp.text[:200]}

    async def http_post(path: str, body: dict) -> tuple[int, dict]:
        url = BASE_URL + path
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, json=body, headers=_build_headers())
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"_raw": resp.text[:200]}

    async def http_post_sse(path: str, body: dict) -> tuple[int, list[dict]]:
        """SSE 스트리밍 POST. 청크를 수집하여 파싱된 이벤트 목록을 반환한다."""
        url = BASE_URL + path
        h = _build_headers()
        h["Accept"] = "text/event-stream"
        events: list[dict] = []
        status_code = 0
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
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

except ImportError:
    import urllib.error
    import urllib.request

    _HTTP_BACKEND = "urllib"

    def _build_headers() -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    async def http_get(path: str) -> tuple[int, dict]:
        url = BASE_URL + path
        req = urllib.request.Request(url, headers=_build_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {}

    async def http_post(path: str, body: dict) -> tuple[int, dict]:
        url = BASE_URL + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=_build_headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {}

    async def http_post_sse(path: str, body: dict) -> tuple[int, list[dict]]:
        """urllib fallback: SSE 스트리밍을 동기 방식으로 읽는다."""
        url = BASE_URL + path
        data = json.dumps(body).encode()
        h = _build_headers()
        h["Accept"] = "text/event-stream"
        req = urllib.request.Request(url, data=data, headers=h, method="POST")
        events: list[dict] = []
        status_code = 0
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
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


# ---------------------------------------------------------------------------
# 결과 기록 / 출력 헬퍼
# ---------------------------------------------------------------------------


def _record(
    scenario_num: int,
    name: str,
    passed: bool,
    elapsed: float,
    error: Optional[str] = None,
    detail: Optional[Any] = None,
) -> dict:
    tag = "[PASS]" if passed else "[FAIL]"
    suffix = f"({elapsed:.2f}s)"
    if passed:
        print(f"{tag} Scenario {scenario_num}: {name} {suffix}")
    else:
        print(f"{tag} Scenario {scenario_num}: {name} — {error} {suffix}")

    entry = {
        "scenario": scenario_num,
        "name": name,
        "passed": passed,
        "elapsed_s": round(elapsed, 3),
        "error": error,
        "detail": detail,
    }
    _results.append(entry)
    return entry


def _extract_text_from_events(events: list[dict]) -> str:
    """SSE 이벤트 목록에서 최종 텍스트를 추출한다.

    v2/agent/stream 이벤트 구조:
      - synthesis 노드: {"node": "synthesis", "final_text": "..."}
      - v1/agent/stream 이벤트: {"text": "...", "finished": true}
    """
    # synthesis 노드 final_text 우선
    for ev in reversed(events):
        if ev.get("node") == "synthesis" and ev.get("final_text"):
            return ev["final_text"]
    # v1 스트리밍 호환: finished=true인 마지막 이벤트의 text
    for ev in reversed(events):
        if ev.get("finished") and ev.get("text"):
            return ev["text"]
    # 전체 이벤트에서 non-empty text를 이어붙인다 (fallback)
    chunks = [ev.get("text", "") or ev.get("final_text", "") for ev in events]
    return "".join(c for c in chunks if c)


def _contains_legal_keyword(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in LEGAL_PATTERNS)


# ---------------------------------------------------------------------------
# 시나리오 구현
# ---------------------------------------------------------------------------


async def scenario1_health_check() -> dict:
    """Scenario 1: Health Check."""
    t0 = time.monotonic()
    try:
        status_code, body = await http_get("/health")
        elapsed = time.monotonic() - t0

        if status_code != 200:
            return _record(1, "Health Check", False, elapsed, f"HTTP {status_code}", {"body": body})

        # api_server.py: /health는 "status": "healthy" 반환
        srv_status = body.get("status", "")
        if srv_status not in ("ok", "healthy"):
            return _record(
                1,
                "Health Check",
                False,
                elapsed,
                f"status 필드가 ok/healthy가 아님: {srv_status!r}",
                {"body": body},
            )

        return _record(1, "Health Check", True, elapsed, detail={"status": srv_status})
    except Exception as exc:
        return _record(1, "Health Check", False, time.monotonic() - t0, str(exc))


async def scenario2_base_model_generation() -> dict:
    """Scenario 2: Base Model Generation (OpenAI-compatible /v1/completions).

    vLLM이 /v1/completions 엔드포인트를 직접 노출한다.
    GovOn api_server.py에 구현되어 있지 않으므로 vLLM 레이어 엔드포인트를 사용한다.
    서버가 /v1/completions를 지원하지 않으면 /v1/generate 레거시로 fallback한다.
    """
    t0 = time.monotonic()
    body_completions = {
        "model": BASE_MODEL,
        "prompt": "대한민국 수도는 어디입니까?",
        "max_tokens": 64,
        "temperature": 0.0,
    }
    try:
        status_code, resp = await http_post("/v1/completions", body_completions)
        elapsed = time.monotonic() - t0

        # vLLM /v1/completions 응답 구조 확인
        if status_code == 200:
            choices = resp.get("choices", [])
            if choices and choices[0].get("text") is not None:
                text = choices[0]["text"]
                return _record(
                    2,
                    "Base Model Generation",
                    True,
                    elapsed,
                    detail={"endpoint": "/v1/completions", "text_preview": text[:100]},
                )
            return _record(
                2, "Base Model Generation", False, elapsed, "choices[0].text 없음", {"resp": resp}
            )

        # /v1/completions 미지원 시 /v1/generate 레거시로 fallback
        body_legacy = {
            "prompt": "대한민국 수도는 어디입니까?",
            "max_tokens": 64,
            "temperature": 0.0,
            "use_rag": False,
        }
        status_code2, resp2 = await http_post("/v1/generate", body_legacy)
        elapsed2 = time.monotonic() - t0
        if status_code2 == 200 and resp2.get("text"):
            return _record(
                2,
                "Base Model Generation",
                True,
                elapsed2,
                detail={"endpoint": "/v1/generate (fallback)", "text_preview": resp2["text"][:100]},
            )

        return _record(
            2,
            "Base Model Generation",
            False,
            elapsed2,
            f"/v1/completions HTTP {status_code}, /v1/generate HTTP {status_code2}",
            {"completions_resp": resp, "generate_resp": resp2},
        )
    except Exception as exc:
        return _record(2, "Base Model Generation", False, time.monotonic() - t0, str(exc))


async def _call_agent(
    message: str,
    session_id: str,
    use_stream: bool = True,
) -> tuple[bool, str, Optional[str]]:
    """에이전트 엔드포인트를 호출하고 (성공여부, 응답텍스트, 에러) 를 반환한다.

    v2/agent/stream (SSE) → v2/agent/run (REST) 순으로 시도한다.
    use_rag=False를 기본으로 전달하여 LoRA 경로를 강제한다.
    """
    body = {"query": message, "session_id": session_id, "use_rag": False}

    # v2/agent/stream 시도 (SSE)
    if use_stream:
        try:
            status_code, events = await http_post_sse("/v2/agent/stream", body)
            if status_code == 200 and events:
                text = _extract_text_from_events(events)
                if text:
                    return True, text, None
                # 이벤트는 수신했지만 text가 없는 경우 — error 이벤트 확인
                for ev in events:
                    if ev.get("status") == "error":
                        return False, "", ev.get("error", "unknown error")
                # __interrupt__ 또는 awaiting_approval 이벤트 → 자동 승인 후 최종 텍스트 수집
                # LangGraph interrupt()는 "__interrupt__" 노드로 emit됨
                awaiting = next(
                    (
                        ev
                        for ev in events
                        if ev.get("status") == "awaiting_approval"
                        or ev.get("node") == "__interrupt__"
                    ),
                    None,
                )
                if awaiting:
                    thread_id = awaiting.get("thread_id") or session_id
                    try:
                        approve_code, approve_resp = await http_post(
                            f"/v2/agent/approve?thread_id={thread_id}&approved=true", {}
                        )
                        if approve_code == 200:
                            final_text = approve_resp.get("text", "") or approve_resp.get(
                                "final_text", ""
                            )
                            if final_text:
                                return True, final_text, None
                            return False, "", f"approve 200 but text 없음: {approve_resp}"
                        return False, "", f"approve HTTP {approve_code}: {approve_resp}"
                    except Exception as approve_exc:
                        return False, "", f"approve 호출 실패: {approve_exc}"
                return False, "", f"SSE 이벤트 수신했으나 text 없음 (events={len(events)})"
        except Exception as exc:
            logger.warning("Stream error: %s", exc)  # fallback to /v2/agent/run

    # v2/agent/run 시도 (REST)
    try:
        status_code, resp = await http_post("/v2/agent/run", body)
        if status_code == 200:
            text = resp.get("text", "") or resp.get("final_text", "")
            if resp.get("status") == "error":
                return False, text, resp.get("error", "agent run error")
            if text:
                return True, text, None
            # awaiting_approval 상태 — 실제 텍스트 생성 없음으로 failure 처리
            if resp.get("status") == "awaiting_approval":
                return (
                    False,
                    "",
                    f"awaiting_approval: 텍스트 미생성 (thread_id={resp.get('thread_id')})",
                )
            return False, "", f"text 없음, status={resp.get('status')}"
        return False, "", f"HTTP {status_code}: {resp}"
    except Exception as exc:
        return False, "", str(exc)


# Scenario 3/4 공유 세션 ID (동일 run에서 같은 세션 사용)
_RUN_SESSION_ID = str(uuid4())


async def scenario3_public_admin_lora() -> dict:
    """Scenario 3: Public Admin LoRA — draft_response (v2/agent/stream)."""
    t0 = time.monotonic()
    try:
        ok, text, err = await _call_agent(
            message="주차 위반 과태료 이의신청 민원에 대한 답변 초안을 작성해줘",
            session_id=_RUN_SESSION_ID,
        )
        elapsed = time.monotonic() - t0
        if not ok:
            return _record(
                3,
                "Public Admin LoRA (draft_response)",
                False,
                elapsed,
                err,
                {"text_preview": text[:200] if text else ""},
            )
        if not text.strip():
            return _record(
                3, "Public Admin LoRA (draft_response)", False, elapsed, "응답 텍스트가 비어있음"
            )
        return _record(
            3,
            "Public Admin LoRA (draft_response)",
            True,
            elapsed,
            detail={"text_preview": text[:200]},
        )
    except Exception as exc:
        return _record(
            3, "Public Admin LoRA (draft_response)", False, time.monotonic() - t0, str(exc)
        )


async def scenario4_legal_lora() -> dict:
    """Scenario 4: Legal LoRA — draft_response (v2/agent/stream).

    독립 세션에서 민원 답변 초안 요청 후 동일 세션에서 법령 근거 보강을 요청한다.
    응답에 법령/조항 관련 패턴이 포함되어 있는지 확인한다.
    """
    t0 = time.monotonic()
    session_id = str(uuid4())
    try:
        # 동일 세션에서 public_admin 요청 먼저 (draft_response는 이전 답변 컨텍스트 필요)
        ok_admin, _, err_admin = await _call_agent(
            message="건축 허가 신청 민원에 대한 답변 초안을 작성해줘",
            session_id=session_id,
        )
        if not ok_admin:
            elapsed = time.monotonic() - t0
            return _record(
                4,
                "Legal LoRA (draft_response)",
                False,
                elapsed,
                f"public_admin 선행 요청 실패: {err_admin}",
            )

        ok, text, err = await _call_agent(
            message="위 답변에 관련 법령과 판례 근거를 보강해줘",
            session_id=session_id,
        )
        elapsed = time.monotonic() - t0
        if not ok:
            return _record(
                4,
                "Legal LoRA (draft_response)",
                False,
                elapsed,
                err,
                {"text_preview": text[:200] if text else ""},
            )
        if not text.strip():
            return _record(
                4, "Legal LoRA (draft_response)", False, elapsed, "응답 텍스트가 비어있음"
            )

        has_legal = _contains_legal_keyword(text)
        matched = [p for p in LEGAL_PATTERNS if re.search(p, text)]
        detail = {
            "has_legal_keyword": has_legal,
            "matched_patterns": matched,
            "text_preview": text[:300],
        }
        if not has_legal:
            return _record(
                4,
                "Legal LoRA (draft_response)",
                False,
                elapsed,
                f"법령 패턴 미발견 ({LEGAL_PATTERNS[:3]}...)",
                detail,
            )
        return _record(4, "Legal LoRA (draft_response)", True, elapsed, detail=detail)
    except Exception as exc:
        return _record(4, "Legal LoRA (draft_response)", False, time.monotonic() - t0, str(exc))


async def scenario5_sequential_multi_lora_switching() -> dict:
    """Scenario 5: Sequential Multi-LoRA Switching (public_admin → legal x3).

    public_admin 요청 → legal 요청을 3회 반복하여 LoRA 전환 오류가 없는지 확인한다.
    반복마다 별도의 UUID 세션 ID를 사용한다.
    """
    t0 = time.monotonic()
    errors: list[str] = []
    iterations = 3

    for i in range(1, iterations + 1):
        session_id = str(uuid4())

        # public_admin 요청
        ok, text, err = await _call_agent(
            message="행정처분 이의신청 민원 답변 초안을 작성해줘",
            session_id=session_id,
        )
        if not ok or not text.strip():
            errors.append(f"iter {i} public_admin: {err or '빈 응답'}")
            continue

        # legal 요청 (동일 세션)
        ok2, text2, err2 = await _call_agent(
            message="위 답변에 관련 법령 근거를 추가해줘",
            session_id=session_id,
        )
        if not ok2 or not text2.strip():
            errors.append(f"iter {i} legal: {err2 or '빈 응답'}")

    elapsed = time.monotonic() - t0
    if errors:
        return _record(
            5,
            "Sequential Multi-LoRA Switching",
            False,
            elapsed,
            "; ".join(errors),
            {"iterations": iterations, "errors": errors},
        )
    return _record(
        5,
        "Sequential Multi-LoRA Switching",
        True,
        elapsed,
        detail={"iterations": iterations, "all_passed": True},
    )


async def scenario6_lora_id_consistency() -> dict:
    """Scenario 6: LoRA ID Consistency Check (정보성).

    /v1/models (vLLM OpenAI-compatible)에서 civil/legal 어댑터 노출 여부를 확인한다.
    vLLM은 버전/설정에 따라 LoRA 어댑터를 /v1/models에 노출하지 않을 수 있으므로,
    미감지 시 FAIL이 아닌 WARNING으로 기록하고 전체 결과에 영향을 주지 않는다.
    """
    t0 = time.monotonic()
    try:
        status_code, health = await http_get("/health")
        elapsed = time.monotonic() - t0

        if status_code != 200:
            return _record(
                6, "LoRA ID Consistency Check", False, elapsed, f"/health HTTP {status_code}"
            )

        detail: dict = {"health_status": health.get("status")}

        # /health feature_flags / agents_loaded 정보 기록
        detail["agents_loaded"] = health.get("agents_loaded", [])
        detail["model"] = health.get("model", "")
        detail["feature_flags"] = health.get("feature_flags", {})

        public_admin_found = False
        legal_found = False

        # /v1/models 시도 (vLLM OpenAI-compatible)
        # HF Hub 경로("govon-civil-adapter")에 "civil" 문자열이 포함되므로 경로 기반으로 검색
        try:
            models_status, models_resp = await http_get("/v1/models")
            if models_status == 200:
                model_ids = [m.get("id", "") for m in models_resp.get("data", [])]
                detail["v1_models"] = model_ids
                public_admin_found = any("civil" in mid for mid in model_ids)
                legal_found = any("legal" in mid for mid in model_ids)
                detail["public_admin_adapter_in_models"] = public_admin_found
                detail["legal_adapter_in_models"] = legal_found
        except Exception as exc:
            logger.warning("Failed to fetch /v1/models: %s", exc)
            detail["v1_models"] = "unavailable"

        # vLLM이 /v1/models에 어댑터를 노출하지 않을 수 있으므로 정보성 기록만 수행
        if not public_admin_found or not legal_found:
            missing = []
            if not public_admin_found:
                missing.append("public_admin")
            if not legal_found:
                missing.append("legal")
            detail["warning"] = f"어댑터 미감지 (vLLM 버전에 따라 정상): {', '.join(missing)}"
            logger.warning(detail["warning"])

        return _record(6, "LoRA ID Consistency Check", True, time.monotonic() - t0, detail=detail)
    except Exception as exc:
        return _record(6, "LoRA ID Consistency Check", False, time.monotonic() - t0, str(exc))


# ---------------------------------------------------------------------------
# 메인 러너
# ---------------------------------------------------------------------------


async def main() -> int:
    print("GovOn Legal LoRA 서빙 통합 검증")
    print(f"  대상 서버: {BASE_URL}")
    print(f"  인증: {'API_KEY 설정됨' if API_KEY else '미설정 (비인증)'}")
    print(f"  HTTP 백엔드: {_HTTP_BACKEND}")
    print(f"  타임아웃: {TIMEOUT}s / 시나리오")
    print("-" * 60)

    scenarios = [
        scenario1_health_check,
        scenario2_base_model_generation,
        scenario3_public_admin_lora,
        scenario4_legal_lora,
        scenario5_sequential_multi_lora_switching,
        scenario6_lora_id_consistency,
    ]

    for fn in scenarios:
        await fn()

    print("-" * 60)
    passed = sum(1 for r in _results if r["passed"])
    failed = len(_results) - passed
    print(f"결과: {passed}/{len(_results)} 통과, {failed} 실패")

    # JSON 결과 저장
    output = {
        "server_url": BASE_URL,
        "http_backend": _HTTP_BACKEND,
        "total": len(_results),
        "passed": passed,
        "failed": failed,
        "scenarios": _results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {RESULTS_PATH}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
