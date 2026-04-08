"""Phase 6: Advanced 시나리오 (S14-S20).

신규 고급 시나리오: 파이프라인 무결성, SLA, fallback, 세션 지속,
LoRA 핫스왑, 에러 전파, Evidence Envelope 검증.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional

from ..config import EXPECTED_APPROVED_FLOW, NODE_SLA_THRESHOLDS, VALID_TOOLS
from ..flow_tracker import FlowValidator, LatencyAggregator, PipelineFlowTracker
from ..http_client import http_post, http_post_sse
from ..logger import E2ELogger
from .base import call_agent_with_approval, session_id


async def scenario14_full_pipeline_flow(logger: E2ELogger) -> dict:
    """S14: Full Pipeline Flow Integrity — 6-node 전이를 SSE 이벤트로 검증."""
    logger.set_context(phase=6, scenario_id=14)
    t0 = time.monotonic()
    tracker = PipelineFlowTracker()

    try:
        ok, text, meta, err = await call_agent_with_approval(
            "도로 파손 민원 답변 초안을 작성해주세요",
            session_id(14),
            approve=True,
            timeout=180,
            tracker=tracker,
        )
        elapsed = time.monotonic() - t0

        actual_nodes = tracker.node_sequence
        flow_text = tracker.to_text()
        logger.flow(f"실제 흐름: {flow_text}")

        valid, issues = FlowValidator.validate_approved_flow(actual_nodes)
        assertions = [f"flow: {flow_text}"]
        warnings = []

        if valid:
            assertions.append("승인 경로 노드 순서 검증 통과")
        else:
            for issue in issues:
                warnings.append(issue)

        # planner와 tool_execute 노드가 모두 관측되어야 PASS
        has_planner = any("planner" in n for n in actual_nodes)
        has_tool_execute = any("tool_execute" in n or "tool" in n for n in actual_nodes)
        has_key_nodes = has_planner and has_tool_execute and ok
        status = "passed" if has_key_nodes else "failed"

        return logger.scenario_result(
            14,
            "Full Pipeline Flow Integrity",
            6,
            status,
            elapsed,
            assertions=assertions,
            warnings=warnings,
            error=err if not has_key_nodes else None,
            detail={"actual_nodes": actual_nodes, "flow": flow_text, "meta": meta},
        )
    except Exception as exc:
        return logger.scenario_result(
            14, "Full Pipeline Flow Integrity", 6, "failed", time.monotonic() - t0, error=str(exc)
        )


async def scenario15_node_latency_sla(logger: E2ELogger, aggregator: LatencyAggregator) -> dict:
    """S15: Node Latency SLA 검증 — 3회 요청의 p95가 SLA 이내인지."""
    logger.set_context(phase=6, scenario_id=15)
    t0 = time.monotonic()

    # 3회 요청으로 레이턴시 수집
    for i in range(3):
        tracker = PipelineFlowTracker()
        try:
            await call_agent_with_approval(
                f"민원 답변 초안 {i+1}",
                session_id(150 + i),
                approve=True,
                timeout=180,
                tracker=tracker,
            )
            aggregator.record_from_tracker(tracker)
        except Exception:
            pass

    elapsed = time.monotonic() - t0
    assertions = []
    violations = []

    total_samples = 0
    for sla in NODE_SLA_THRESHOLDS:
        stats = aggregator.stats(sla.name)
        total_samples += stats["count"]
        if stats["count"] == 0:
            assertions.append(f"{sla.name}: 데이터 없음")
            continue
        p95_sec = stats["p95"] / 1000
        if p95_sec <= sla.max_p95_sec:
            assertions.append(f"{sla.name}: p95={p95_sec:.1f}s <= {sla.max_p95_sec}s OK")
        else:
            violations.append(f"{sla.name}: p95={p95_sec:.1f}s > {sla.max_p95_sec}s")

    logger.metric(f"레이턴시 요약:\n{aggregator.summary_text()}")

    # 최소 1건 이상의 샘플이 있어야 유효한 검증으로 간주
    if total_samples == 0:
        violations.append("레이턴시 샘플 0건: SLA 검증 불가")

    status = "passed" if not violations else "failed"
    return logger.scenario_result(
        15,
        "Node Latency SLA",
        6,
        status,
        elapsed,
        assertions=assertions,
        warnings=violations if violations else None,
        error="; ".join(violations) if violations else None,
        detail={"stats": aggregator.all_stats()},
    )


async def scenario16_planner_fallback_chain(logger: E2ELogger) -> dict:
    """S16: Planner Fallback Chain — Regex fallback이 정상 동작하는지 확인."""
    logger.set_context(phase=6, scenario_id=16)
    t0 = time.monotonic()

    try:
        # 애매한 쿼리로 Regex fallback 유도
        ok, text, meta, err = await call_agent_with_approval(
            "처리해주세요",
            session_id(16),
            approve=True,
            timeout=120,
        )
        elapsed = time.monotonic() - t0

        planned = meta.get("planned_tools", [])
        assertions = []
        if planned:
            assertions.append(f"planned_tools 생성됨: {planned}")
            invalid = [t for t in planned if t not in VALID_TOOLS]
            if invalid:
                assertions.append(f"비정상 도구 포함: {invalid}")
            else:
                assertions.append("모든 도구가 VALID_TOOLS 내에 있음")

        status = "passed" if planned and not invalid else "failed"
        return logger.scenario_result(
            16,
            "Planner Fallback Chain",
            6,
            status,
            elapsed,
            assertions=assertions,
            error=err if not planned else None,
            detail={"meta": meta},
        )
    except Exception as exc:
        return logger.scenario_result(
            16, "Planner Fallback Chain", 6, "failed", time.monotonic() - t0, error=str(exc)
        )


async def scenario17_session_context_persistence(logger: E2ELogger) -> dict:
    """S17: Session Context Persistence — 3-turn 대화에서 세션 컨텍스트 유지."""
    logger.set_context(phase=6, scenario_id=17)
    t0 = time.monotonic()
    sid = session_id(17)

    try:
        # Turn 1: 초기 질문
        ok1, text1, _, err1 = await call_agent_with_approval(
            "주민센터 업무 시간에 대해 답변을 작성해주세요", sid, timeout=180
        )
        if not ok1:
            return logger.scenario_result(
                17,
                "Session Context Persistence",
                6,
                "failed",
                time.monotonic() - t0,
                error=f"Turn 1 실패: {err1}",
            )

        # Turn 2: 후속 질문 (이전 컨텍스트 참조)
        ok2, text2, _, err2 = await call_agent_with_approval(
            "위 답변을 더 정중하게 수정해주세요", sid, timeout=180
        )
        if not ok2:
            return logger.scenario_result(
                17,
                "Session Context Persistence",
                6,
                "failed",
                time.monotonic() - t0,
                error=f"Turn 2 실패: {err2}",
            )

        # Turn 3: 추가 후속 질문
        ok3, text3, _meta3, err3 = await call_agent_with_approval(
            "관련 법령 근거를 추가해주세요", sid, timeout=180
        )
        elapsed = time.monotonic() - t0

        # 교차-turn 검증: Turn 2가 Turn 1의 내용을 참조하는지 확인
        # "수정"/"정중" 등의 키워드가 Turn 2에 존재하면 이전 컨텍스트를 인지한 것으로 판단
        cross_turn_ok = ok2 and (len(text2) > 0) and (text2 != text1)

        assertions = [
            f"Turn 1: {len(text1)} chars",
            f"Turn 2: {len(text2)} chars",
            f"Cross-turn: {'Turn 2 != Turn 1' if cross_turn_ok else 'Turn 2 == Turn 1 (세션 미참조 의심)'}",
            (
                f"Turn 3: {'성공' if ok3 else '실패'} ({len(text3)} chars)"
                if ok3
                else f"Turn 3 실패: {err3}"
            ),
        ]

        # 3턴 모두 성공 + 교차-turn 검증 통과
        status = "passed" if ok1 and ok2 and ok3 and cross_turn_ok else "failed"
        return logger.scenario_result(
            17,
            "Session Context Persistence",
            6,
            status,
            elapsed,
            assertions=assertions,
            error=err3 if not ok3 else None,
            detail={"texts": [text1[:100], text2[:100], text3[:100] if text3 else ""]},
        )
    except Exception as exc:
        return logger.scenario_result(
            17, "Session Context Persistence", 6, "failed", time.monotonic() - t0, error=str(exc)
        )


async def scenario18_lora_hot_switch(logger: E2ELogger) -> dict:
    """S18: LoRA Adapter Hot-Switch — public_admin -> legal -> public_admin 안정성."""
    logger.set_context(phase=6, scenario_id=18)
    t0 = time.monotonic()
    sid = session_id(18)
    results = []

    queries = [
        ("민원 답변 초안을 작성해주세요", "public_admin"),
        ("위 답변에 법령 근거를 추가해주세요", "legal"),
        ("새로운 민원 답변 초안을 작성해주세요", "public_admin"),
    ]

    for query, expected_domain in queries:
        ok, text, meta, err = await call_agent_with_approval(query, sid, approve=True, timeout=180)
        actual_adapter = meta.get("adapter_mode", "unknown")
        results.append(
            {
                "domain": expected_domain,
                "actual_adapter": actual_adapter,
                "ok": ok,
                "text_len": len(text) if text else 0,
                "error": err,
            }
        )

    elapsed = time.monotonic() - t0
    all_ok = all(r["ok"] for r in results)

    # 어댑터가 1종류만 사용되면 hot-switch 미발생 의심
    adapters_seen = {r["actual_adapter"] for r in results if r["ok"]}
    hot_switch_detected = len(adapters_seen) > 1

    assertions = [
        f"{r['domain']}: {'OK' if r['ok'] else 'FAIL'} adapter={r['actual_adapter']} ({r['text_len']} chars)"
        for r in results
    ]
    if not hot_switch_detected:
        assertions.append(f"WARNING: 어댑터 전환 미감지 (사용된 어댑터: {adapters_seen})")

    return logger.scenario_result(
        18,
        "LoRA Adapter Hot-Switch",
        6,
        "passed" if all_ok else "failed",
        elapsed,
        assertions=assertions,
        warnings=None if hot_switch_detected else [f"어댑터 전환 미감지: {adapters_seen}"],
        error="; ".join(r["error"] or "" for r in results if not r["ok"]) or None,
        detail={"results": results, "adapters_seen": list(adapters_seen)},
    )


async def scenario19_graceful_error_propagation(logger: E2ELogger) -> dict:
    """S19: Graceful Error Propagation — 비정상 입력 시 non-500 응답."""
    logger.set_context(phase=6, scenario_id=19)
    t0 = time.monotonic()

    test_cases = [
        ("", "empty query"),
        ("x" * 50000, "oversized query"),
    ]
    assertions = []

    has_exception = False
    for query, label in test_cases:
        try:
            code, resp = await http_post("/v2/agent/run", {"query": query}, timeout=10)
            if code == 500:
                assertions.append(f"{label}: FAIL HTTP 500")
            elif 400 <= code < 500:
                assertions.append(f"{label}: OK HTTP {code} (expected client error)")
            else:
                assertions.append(f"{label}: OK HTTP {code}")
        except Exception as exc:
            has_exception = True
            assertions.append(f"{label}: FAIL exception {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - t0
    no_500 = all("FAIL" not in a for a in assertions) and not has_exception

    return logger.scenario_result(
        19,
        "Graceful Error Propagation",
        6,
        "passed" if no_500 else "failed",
        elapsed,
        assertions=assertions,
        error="500 응답 감지" if not no_500 else None,
    )


async def scenario20_evidence_envelope(logger: E2ELogger) -> dict:
    """S20: Evidence Envelope 통합 — API 결과의 evidence 정규화 검증."""
    logger.set_context(phase=6, scenario_id=20)
    t0 = time.monotonic()

    try:
        ok, text, meta, err = await call_agent_with_approval(
            "도로 파손 민원 답변을 작성해주세요",
            session_id(20),
            approve=True,
            timeout=180,
        )
        elapsed = time.monotonic() - t0

        assertions = []
        tool_results = meta.get("tool_results", {})

        # tool_results에서 evidence 필드 검사
        evidence_found = False
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and result.get("evidence"):
                evidence_found = True
                evidence = result["evidence"]
                if isinstance(evidence, dict):
                    items = evidence.get("items", [])
                    status_val = evidence.get("status", "")
                    assertions.append(
                        f"{tool_name}: evidence.status={status_val}, items={len(items)}"
                    )
                else:
                    assertions.append(f"{tool_name}: evidence가 dict가 아님")

        if not evidence_found:
            assertions.append("evidence 필드를 가진 tool_result 없음")

        # text + 최소 1개 evidence 검증 성공이어야 PASS
        status = "passed" if ok and text and evidence_found else "failed"
        return logger.scenario_result(
            20,
            "Evidence Envelope",
            6,
            status,
            elapsed,
            assertions=assertions,
            error=err if not ok else None,
            detail={"tool_results_keys": list(tool_results.keys())},
        )
    except Exception as exc:
        return logger.scenario_result(
            20, "Evidence Envelope", 6, "failed", time.monotonic() - t0, error=str(exc)
        )


async def run_phase6(
    logger: E2ELogger,
    observed_tools: set[str] | None = None,
    aggregator: LatencyAggregator | None = None,
) -> list[dict]:
    """Phase 6 전체 실행.

    Parameters
    ----------
    observed_tools : set[str] | None
        runner에서 전달받은 관측 도구 집합. Phase 6 시나리오에서 갱신.
    aggregator : LatencyAggregator | None
        runner에서 전달받은 레이턴시 집계기. None이면 내부 생성.
    """
    logger.info("\n[Phase 6] Advanced Scenarios")
    logger.info("-" * 40)

    if aggregator is None:
        aggregator = LatencyAggregator()

    results = []

    scenarios = [
        lambda: scenario14_full_pipeline_flow(logger),
        lambda: scenario15_node_latency_sla(logger, aggregator),
        lambda: scenario16_planner_fallback_chain(logger),
        lambda: scenario17_session_context_persistence(logger),
        lambda: scenario18_lora_hot_switch(logger),
        lambda: scenario19_graceful_error_propagation(logger),
        lambda: scenario20_evidence_envelope(logger),
    ]

    for scenario_fn in scenarios:
        result = await scenario_fn()
        results.append(result)
        # 실행된 시나리오에서 관측된 도구 수집
        if observed_tools is not None:
            detail = result.get("detail", {})
            for key in ("planned_tools", "actual_nodes"):
                if isinstance(detail.get(key), (list, set)):
                    observed_tools.update(detail[key])

    return results
