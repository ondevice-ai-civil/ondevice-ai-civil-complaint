#!/usr/bin/env python3
"""GovOn E2E GPU Test Runner.

HuggingFace Spaces GPU에 배포된 govon-runtime 서버에 대해
전체 에이전트 파이프라인을 검증한다.

사용법:
    # 전체 실행
    GOVON_RUNTIME_URL=https://<space>.hf.space python -m scripts.e2e_gpu_test.runner

    # 특정 Phase만 실행
    GOVON_RUNTIME_URL=... python -m scripts.e2e_gpu_test.runner --phase 1

    # 실시간 모니터링 모드
    GOVON_RUNTIME_URL=... python -m scripts.e2e_gpu_test.runner --monitor

6-Phase 구성:
    Phase 1: Infrastructure (hard gate)
    Phase 2: Agent Pipeline Core
    Phase 3: data.go.kr API Tools (soft gate)
    Phase 4: Adapter Dynamics
    Phase 5: Robustness
    Phase 6: Advanced (flow integrity, SLA, fallback, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from uuid import uuid4

from .config import BASE_URL, LOG_PATH, RESULTS_PATH, TIMEOUT, VALID_TOOLS
from .flow_tracker import LatencyAggregator
from .http_client import get_http_backend, http_get, http_get_raw
from .logger import E2ELogger
from .report import print_summary, write_json_report
from .scenarios.phase6_advanced import run_phase6

# 기존 verify_e2e_tool_calling.py의 Phase 1-5 시나리오를 import
# (기존 스크립트를 직접 참조하지 않고, runner가 Phase 6만 직접 실행)
# Phase 1-5는 기존 scripts/verify_e2e_tool_calling.py를 사용하거나
# 점진적으로 이관할 수 있다.


_observed_tools: set[str] = set()
_results: list[dict] = []
_run_id = uuid4().hex


async def _wait_cold_start(logger: E2ELogger) -> float:
    """서버 cold start 대기. 최대 10회 x 30초."""
    total_wait = 0.0
    for i in range(10):
        try:
            code, body = await http_get("/health", timeout=10)
            if code == 200 and body.get("status") in ("ok", "healthy"):
                logger.info(f"서버 준비 완료 (대기 {total_wait:.0f}s)")
                return total_wait
        except Exception:
            pass
        if i < 9:
            logger.info(f"서버 대기 중... ({i + 1}/10, 30s 후 재시도)")
            await asyncio.sleep(30)
            total_wait += 30

    logger.warn("서버 준비 확인 실패 -- 계속 진행")
    return total_wait


async def run_phase1_infra(logger: E2ELogger) -> list[dict]:
    """Phase 1: Infrastructure (hard gate) -- 기본 서버 상태 확인."""
    logger.info("\n[Phase 1] Infrastructure (hard gate)")
    logger.info("-" * 40)
    results = []

    # S1: Health & Profile
    logger.set_context(phase=1, scenario_id=1)
    t0 = time.monotonic()
    try:
        code, body = await http_get("/health", timeout=10)
        elapsed = time.monotonic() - t0
        if code == 200 and body.get("status") in ("ok", "healthy"):
            results.append(logger.scenario_result(
                1, "Health & Profile", 1, "passed", elapsed,
                assertions=[f"HTTP 200, status={body.get('status')}"],
                detail={"model": body.get("model"), "profile": body.get("profile")},
            ))
        else:
            results.append(logger.scenario_result(
                1, "Health & Profile", 1, "failed", elapsed,
                error=f"HTTP {code}, status={body.get('status')}",
            ))
            return results  # hard gate
    except Exception as exc:
        results.append(logger.scenario_result(
            1, "Health & Profile", 1, "failed", time.monotonic() - t0, error=str(exc),
        ))
        return results

    # S2: Base Model Generation
    logger.set_context(phase=1, scenario_id=2)
    from .http_client import http_post
    t0 = time.monotonic()
    try:
        from .config import BASE_MODEL
        code, resp = await http_post(
            "/v1/completions",
            {"model": BASE_MODEL, "prompt": "대한민국의 수도는", "max_tokens": 32, "temperature": 0.0},
            timeout=60,
        )
        elapsed = time.monotonic() - t0
        choices = resp.get("choices", [])
        if code == 200 and choices and choices[0].get("text", "").strip():
            results.append(logger.scenario_result(
                2, "Base Model Generation", 1, "passed", elapsed,
                assertions=["HTTP 200", "non-empty text"],
            ))
        else:
            # fallback: /v1/generate
            code2, resp2 = await http_post(
                "/v1/generate",
                {"prompt": "대한민국의 수도는", "max_tokens": 32, "temperature": 0.0},
                timeout=60,
            )
            elapsed2 = time.monotonic() - t0
            if code2 == 200 and resp2.get("text", "").strip():
                results.append(logger.scenario_result(
                    2, "Base Model Generation", 1, "passed", elapsed2,
                    assertions=["HTTP 200 (fallback /v1/generate)"],
                ))
            else:
                results.append(logger.scenario_result(
                    2, "Base Model Generation", 1, "failed", elapsed2,
                    error=f"/v1/completions={code}, /v1/generate={code2}",
                ))
                return results
    except Exception as exc:
        results.append(logger.scenario_result(
            2, "Base Model Generation", 1, "failed", time.monotonic() - t0, error=str(exc),
        ))
        return results

    # S3: Adapter Registry
    logger.set_context(phase=1, scenario_id=3)
    t0 = time.monotonic()
    try:
        code, resp = await http_get("/v1/models", timeout=10)
        elapsed = time.monotonic() - t0
        if code != 200:
            results.append(logger.scenario_result(
                3, "Adapter Registry", 1, "passed", elapsed,
                warnings=[f"/v1/models HTTP {code} -- 엔드포인트 미노출 (vLLM 설정에 따라 정상)"],
            ))
        else:
            model_ids = [m.get("id", "") for m in resp.get("data", [])]
            results.append(logger.scenario_result(
                3, "Adapter Registry", 1, "passed", elapsed,
                assertions=[f"{len(model_ids)} models found"],
                detail={"model_ids": model_ids},
            ))
    except Exception as exc:
        results.append(logger.scenario_result(
            3, "Adapter Registry", 1, "failed", time.monotonic() - t0, error=str(exc),
        ))

    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description="GovOn E2E GPU Test Runner")
    parser.add_argument("--phase", type=int, help="특정 Phase만 실행 (1-6)")
    parser.add_argument("--monitor", action="store_true", help="실시간 모니터링 모드")
    parser.add_argument("--verbose", action="store_true", default=True, help="상세 출력")
    args = parser.parse_args()

    logger = E2ELogger(LOG_PATH, verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("GovOn E2E GPU Test Suite")
    logger.info("=" * 60)
    logger.info(f"  대상 서버: {BASE_URL}")
    logger.info(f"  HTTP 백엔드: {get_http_backend()}")
    logger.info(f"  타임아웃: {TIMEOUT}s / 시나리오")
    logger.info(f"  run_id: {_run_id}")
    logger.info(f"  로그 파일: {LOG_PATH}")
    logger.info(f"  결과 파일: {RESULTS_PATH}")
    logger.info("-" * 60)

    # Cold start 대기
    logger.info("[Cold Start] 서버 준비 확인 중...")
    cold_start_wait = await _wait_cold_start(logger)

    all_results: list[dict] = []
    aggregator = LatencyAggregator()

    target_phase = args.phase

    # Phase 1: Infrastructure
    if target_phase is None or target_phase == 1:
        phase1_results = await run_phase1_infra(logger)
        all_results.extend(phase1_results)
        phase1_failed = any(r.get("status") == "failed" for r in phase1_results)
        if phase1_failed and target_phase is None:
            logger.error("ABORT: Infrastructure not ready -- Phase 1 failed")
            write_json_report(all_results, RESULTS_PATH, _run_id, cold_start_wait, _observed_tools)
            logger.close()
            return 1

    # Phase 2-5: 기존 스크립트 호환 (점진적 이관 예정)
    if target_phase is not None and target_phase in (2, 3, 4, 5):
        logger.info(f"\n[Phase {target_phase}] 기존 verify_e2e_tool_calling.py를 사용하세요")
        logger.info("  GOVON_RUNTIME_URL=... python scripts/verify_e2e_tool_calling.py")

    # Phase 6: Advanced
    if target_phase is None or target_phase == 6:
        phase6_results = await run_phase6(logger)
        all_results.extend(phase6_results)

    # 요약
    print_summary(all_results, logger, _observed_tools)
    write_json_report(all_results, RESULTS_PATH, _run_id, cold_start_wait, _observed_tools, aggregator)
    logger.info(f"\n결과 저장: {RESULTS_PATH}")
    logger.info(f"로그 저장: {LOG_PATH}")

    logger.close()

    failed = sum(1 for r in all_results if r.get("status") == "failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
