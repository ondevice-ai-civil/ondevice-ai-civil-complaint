"""시나리오 베이스 유틸리티."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..config import LEGAL_PATTERNS, VALID_TOOLS
from ..flow_tracker import PipelineFlowTracker
from ..http_client import http_post, http_post_sse
from ..logger import E2ELogger


def session_id(scenario_num: int) -> str:
    return f"e2e-{scenario_num}-{uuid4().hex[:8]}"


def extract_text_from_events(events: list[dict]) -> str:
    """SSE 이벤트 목록에서 최종 텍스트를 추출한다."""
    for ev in reversed(events):
        if ev.get("node") == "synthesis" and ev.get("final_text"):
            return ev["final_text"]
    for ev in reversed(events):
        if ev.get("finished") and ev.get("text"):
            return ev["text"]
    chunks = [ev.get("text", "") or ev.get("final_text", "") for ev in events]
    return "".join(c for c in chunks if c)


def contains_legal_keyword(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in LEGAL_PATTERNS)


async def call_agent_with_approval(
    query: str,
    sid: str,
    approve: bool = True,
    timeout: float = 180,
    tracker: Optional[PipelineFlowTracker] = None,
) -> tuple[bool, str, dict, Optional[str]]:
    """에이전트 SSE 스트리밍으로 호출 -> awaiting_approval까지 파싱 -> approve/reject.

    Returns: (success, text, metadata_dict, error)
    """
    body = {"query": query, "session_id": sid}
    meta: dict[str, Any] = {
        "planned_tools": [],
        "task_type": None,
        "goal": None,
        "reason": None,
        "tool_results": {},
        "adapter_mode": None,
        "tool_args": {},
    }

    # SSE 스트리밍 시도
    try:
        status_code, events = await http_post_sse("/v2/agent/stream", body, timeout=timeout)
        if status_code != 200:
            raise RuntimeError(f"SSE HTTP {status_code}")

        # flow tracking
        if tracker:
            for ev in events:
                tracker.track_event(ev)

        # 노드별 메타데이터 수집
        awaiting = None
        for ev in events:
            if ev.get("status") == "awaiting_approval" or ev.get("node") == "__interrupt__":
                awaiting = ev
                break
            _extract_meta_from_event(ev, meta)

        if awaiting:
            _extract_meta_from_event(awaiting, meta)
            thread_id = awaiting.get("thread_id") or sid

            approve_code, approve_resp = await http_post(
                f"/v2/agent/approve?thread_id={thread_id}&approved={'true' if approve else 'false'}",
                {},
                timeout=timeout,
            )
            if approve_code != 200:
                return False, "", meta, f"approve HTTP {approve_code}: {approve_resp}"

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

        # awaiting 없이 최종 텍스트가 있는 경우
        text = extract_text_from_events(events)
        for ev in events:
            _extract_meta_from_event(ev, meta)
        if text:
            return True, text, meta, None

        for ev in events:
            if ev.get("status") == "error":
                return False, "", meta, ev.get("error", "unknown error")

        return False, "", meta, f"SSE 이벤트 수신했으나 text/awaiting 없음 (events={len(events)})"

    except Exception as sse_exc:
        pass  # fallback to REST

    # REST fallback: /v2/agent/run
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
            thread_id = resp.get("thread_id") or sid
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


def _extract_meta_from_event(ev: dict, meta: dict) -> None:
    """SSE 이벤트에서 메타데이터를 추출하여 meta dict에 병합한다."""
    approval_req = ev.get("approval_request", {})
    if not isinstance(approval_req, dict):
        approval_req = {}

    if not meta["planned_tools"]:
        meta["planned_tools"] = (
            approval_req.get("planned_tools") or ev.get("planned_tools") or meta["planned_tools"]
        )
    if not meta.get("task_type"):
        meta["task_type"] = approval_req.get("task_type") or ev.get("task_type")
    if not meta.get("goal"):
        meta["goal"] = approval_req.get("goal") or ev.get("goal")
    if not meta.get("reason"):
        meta["reason"] = approval_req.get("reason") or ev.get("reason")
    if ev.get("tool_results") and not meta["tool_results"]:
        meta["tool_results"] = ev["tool_results"]
    if ev.get("adapter_mode") and not meta["adapter_mode"]:
        meta["adapter_mode"] = ev["adapter_mode"]
    if ev.get("tool_args") and not meta["tool_args"]:
        meta["tool_args"] = ev["tool_args"]
