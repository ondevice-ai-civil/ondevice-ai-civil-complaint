"""tool metadata registry — MVP capability의 단일 소스.

Issue #416: tool metadata registry 및 LangGraph executor binding 정리.

이 모듈은 다음을 보장한다:
- planner가 읽는 metadata와 executor binding이 같은 소스에서 나온다
- approval prompt와 session log가 동일한 capability identifier를 사용한다
- 비MVP capability가 registry 수준에서 차단된다
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.inference.tool_router import ToolType

from .api_lookup import ApiLookupCapability
from .append_evidence import AppendEvidenceCapability
from .base import CapabilityBase, CapabilityMetadata
from .demographics_lookup import DemographicsLookupCapability
from .draft_civil_response import DraftCivilResponseCapability
from .issue_detector import IssueDetectorCapability
from .keyword_analyzer import KeywordAnalyzerCapability
from .rag_search import RagSearchCapability
from .stats_lookup import StatsLookupCapability

# ---------------------------------------------------------------------------
# MVP capability stable identifiers (session log, approval prompt에서 사용)
# ToolType enum에서 파생하여 단일 소스를 유지한다.
# ---------------------------------------------------------------------------

MVP_CAPABILITY_IDS: frozenset[str] = frozenset(t.value for t in ToolType)


def get_mvp_capability_ids() -> frozenset[str]:
    """MVP capability stable identifier 집합을 반환한다.

    plan_validator, planner_adapter 등에서 화이트리스트로 사용한다.
    """
    return MVP_CAPABILITY_IDS


def build_mvp_registry(
    *,
    rag_search_fn: Callable[..., Any],
    api_lookup_action: Any = None,
    draft_civil_response_fn: Callable[..., Any],
    append_evidence_fn: Callable[..., Any],
    rag_low_confidence_threshold: float = 0.3,
) -> Dict[str, CapabilityBase]:
    """MVP 4개 capability를 CapabilityBase 인스턴스로 구성한 registry를 반환한다.

    모든 capability가 CapabilityBase를 구현하므로,
    RegistryExecutorAdapter.get_tool_metadata()가 일관된 metadata를 반환한다.

    Parameters
    ----------
    rag_search_fn : Callable
        ``async (query, context, session) -> dict`` 형태의 RAG 검색 함수.
    api_lookup_action : Any, optional
        ``MinwonAnalysisAction`` 인스턴스. None이면 빈 결과 반환.
    draft_civil_response_fn : Callable
        ``async (query, context, session) -> dict`` 형태의 민원 답변 생성 함수.
    append_evidence_fn : Callable
        ``async (query, context, session) -> dict`` 형태의 근거 보강 함수.
    rag_low_confidence_threshold : float
        RAG 검색 저신뢰도 임계값. 기본값 0.3.

    Returns
    -------
    Dict[str, CapabilityBase]
        capability name -> CapabilityBase 인스턴스 매핑.
    """
    return {
        "rag_search": RagSearchCapability(
            execute_fn=rag_search_fn,
            low_confidence_threshold=rag_low_confidence_threshold,
        ),
        "api_lookup": ApiLookupCapability(action=api_lookup_action),
        "draft_civil_response": DraftCivilResponseCapability(
            execute_fn=draft_civil_response_fn,
        ),
        "append_evidence": AppendEvidenceCapability(execute_fn=append_evidence_fn),
        "issue_detector": IssueDetectorCapability(action=api_lookup_action),
        "stats_lookup": StatsLookupCapability(action=api_lookup_action),
        "keyword_analyzer": KeywordAnalyzerCapability(action=api_lookup_action),
        "demographics_lookup": DemographicsLookupCapability(action=api_lookup_action),
    }


def get_all_metadata(
    registry: Dict[str, CapabilityBase],
) -> List[Dict[str, Any]]:
    """registry에 등록된 모든 capability의 metadata를 dict 목록으로 반환한다.

    planner가 tool 목록을 구성할 때 사용한다.

    Parameters
    ----------
    registry : Dict[str, CapabilityBase]
        build_mvp_registry()가 반환한 registry.

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
            }
        )
    return result


def is_mvp_capability(name: str) -> bool:
    """주어진 이름이 MVP capability인지 확인한다."""
    return name in MVP_CAPABILITY_IDS
