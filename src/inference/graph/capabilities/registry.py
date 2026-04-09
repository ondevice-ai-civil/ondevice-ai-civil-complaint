"""tool metadata registry — MVP capability의 단일 소스.

Issue #416: tool metadata registry 및 LangGraph executor binding 정리.

이 모듈은 다음을 보장한다:
- ReAct agent가 읽는 metadata와 executor binding이 같은 소스에서 나온다
- approval prompt와 session log가 동일한 capability identifier를 사용한다
- 비MVP capability가 registry 수준에서 차단된다
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List

from .api_lookup import ApiLookupCapability
from .base import CapabilityBase, CapabilityMetadata
from .demographics_lookup import DemographicsLookupCapability
from .issue_detector import IssueDetectorCapability
from .keyword_analyzer import KeywordAnalyzerCapability
from .stats_lookup import StatsLookupCapability


class ToolType(str, Enum):
    """MVP 내부 capability 카탈로그."""

    API_LOOKUP = "api_lookup"
    ISSUE_DETECTOR = "issue_detector"
    STATS_LOOKUP = "stats_lookup"
    KEYWORD_ANALYZER = "keyword_analyzer"
    DEMOGRAPHICS_LOOKUP = "demographics_lookup"


def get_all_metadata(
    registry: Dict[str, CapabilityBase],
) -> List[Dict[str, Any]]:
    """registry에 등록된 모든 capability의 metadata를 dict 목록으로 반환한다.

    ReAct agent가 tool 목록을 구성할 때 사용한다.

    Parameters
    ----------
    registry : Dict[str, CapabilityBase]
        capability name -> CapabilityBase 인스턴스 매핑.

    Returns
    -------
    List[Dict[str, Any]]
        각 capability의 metadata dict 목록.
    """
    result: List[Dict[str, Any]] = []
    for name, cap in registry.items():
        meta = cap.metadata
        result.append(
            {
                "name": meta.name,
                "description": meta.description,
                "approval_summary": meta.approval_summary,
                "provider": meta.provider,
                "timeout_sec": meta.timeout_sec,
                "parameters": meta.parameters,
            }
        )
    return result


def is_mvp_capability(name: str) -> bool:
    """주어진 이름이 MVP capability인지 확인한다."""
    mvp_ids = frozenset(t.value for t in ToolType)
    return name in mvp_ids
