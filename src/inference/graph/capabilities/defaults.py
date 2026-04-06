"""Capability timeout/retry 기본값 모듈.

Issue #163: capability별 timeout과 retry 기본값을 중앙 집중 관리.
환경변수 GOVON_TOOL_TIMEOUT_{CAPABILITY_NAME} 으로 오버라이드 가능.

예: GOVON_TOOL_TIMEOUT_RAG_SEARCH=20  -> rag_search timeout을 20초로 변경
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from loguru import logger


@dataclass(frozen=True)
class CapabilityDefaults:
    """capability별 timeout/retry 기본값."""

    timeout_sec: float
    max_retries: int


# -----------------------------------------------------------------------
# 기본값 정의 (코드베이스 capability metadata에서 추출)
# -----------------------------------------------------------------------

_DEFAULTS: Dict[str, CapabilityDefaults] = {
    "rag_search": CapabilityDefaults(timeout_sec=15.0, max_retries=0),
    "api_lookup": CapabilityDefaults(timeout_sec=10.0, max_retries=1),
    "draft_civil_response": CapabilityDefaults(timeout_sec=30.0, max_retries=0),
    "append_evidence": CapabilityDefaults(timeout_sec=15.0, max_retries=0),
    "issue_detector": CapabilityDefaults(timeout_sec=15.0, max_retries=0),
    "stats_lookup": CapabilityDefaults(timeout_sec=15.0, max_retries=0),
    "keyword_analyzer": CapabilityDefaults(timeout_sec=10.0, max_retries=0),
    "demographics_lookup": CapabilityDefaults(timeout_sec=15.0, max_retries=0),
}


def get_timeout(capability_name: str) -> float:
    """capability의 timeout(초)을 반환한다.

    환경변수 ``GOVON_TOOL_TIMEOUT_{CAPABILITY_NAME_UPPER}`` 가 설정되어 있으면
    해당 값을 사용하고, 없으면 기본값을 반환한다.

    Parameters
    ----------
    capability_name : str
        capability 이름 (예: "rag_search").

    Returns
    -------
    float
        timeout 초. 알 수 없는 capability는 10.0초.
    """
    env_key = f"GOVON_TOOL_TIMEOUT_{capability_name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        try:
            val = float(env_val)
            if val > 0:
                return val
            logger.warning(
                f"GOVON_TOOL_TIMEOUT_{capability_name.upper()} 값이 양수가 아닙니다: {env_val}"
            )
        except ValueError:
            logger.warning(f"{env_key} 값을 숫자로 파싱할 수 없습니다: {env_val!r}")

    defaults = _DEFAULTS.get(capability_name)
    return defaults.timeout_sec if defaults else 10.0


def get_max_retries(capability_name: str) -> int:
    """capability의 최대 재시도 횟수를 반환한다.

    Parameters
    ----------
    capability_name : str
        capability 이름.

    Returns
    -------
    int
        최대 재시도 횟수. 알 수 없는 capability는 0.
    """
    defaults = _DEFAULTS.get(capability_name)
    return defaults.max_retries if defaults else 0


def get_all_defaults() -> Dict[str, CapabilityDefaults]:
    """등록된 모든 capability 기본값을 반환한다."""
    return dict(_DEFAULTS)
