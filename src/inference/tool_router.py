"""GovOn MVP용 task routing 모듈."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ToolType(str, Enum):
    """MVP 내부 capability 카탈로그."""

    RAG_SEARCH = "rag_search"
    API_LOOKUP = "api_lookup"
    DRAFT_CIVIL_RESPONSE = "draft_civil_response"
    APPEND_EVIDENCE = "append_evidence"


ToolName = ToolType | str


def tool_name(tool: ToolName) -> str:
    return tool.value if isinstance(tool, ToolType) else str(tool)


@dataclass
class ToolStep:
    tool: ToolName
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: Optional[str] = None

    @property
    def step_id(self) -> str:
        return tool_name(self.tool)


@dataclass
class ExecutionPlan:
    steps: List[ToolStep]
    reason: str

    @property
    def tool_names(self) -> List[str]:
        return [step.step_id for step in self.steps]


_EVIDENCE_PATTERNS = [
    r"근거",
    r"출처",
    r"왜\s*이렇게",
    r"이유",
    r"링크",
]

_DRAFT_PATTERNS = [
    r"답변",
    r"회신",
    r"초안",
    r"작성",
    r"다시\s*써",
    r"수정",
    r"정중",
    r"공손",
]

_LOOKUP_PATTERNS = [
    r"유사.*사례",
    r"비슷한.*민원",
    r"민원.*통계",
    r"민원.*현황",
    r"동향",
    r"이슈",
    r"보고서",
    r"트렌드",
    r"국민신문고",
]

_RAG_PATTERNS = [
    r"법령",
    r"규정",
    r"매뉴얼",
    r"공지",
    r"문서",
    r"근거",
]


class ToolRouter:
    """자연어 요청을 MVP capability 조합으로 정규화한다."""

    def __init__(self) -> None:
        self._evidence_patterns = [re.compile(p) for p in _EVIDENCE_PATTERNS]
        self._draft_patterns = [re.compile(p) for p in _DRAFT_PATTERNS]
        self._lookup_patterns = [re.compile(p) for p in _LOOKUP_PATTERNS]
        self._rag_patterns = [re.compile(p) for p in _RAG_PATTERNS]

    def plan(
        self,
        query: str,
        has_context: bool = False,
        force_tools: Optional[List[ToolName]] = None,
    ) -> ExecutionPlan:
        if force_tools:
            return ExecutionPlan(
                steps=[ToolStep(tool=tool) for tool in force_tools],
                reason=f"강제 지정된 capability: {[tool_name(tool) for tool in force_tools]}",
            )

        needs_evidence = self._match_patterns(query, self._evidence_patterns)
        needs_draft = self._match_patterns(query, self._draft_patterns)
        needs_lookup = self._match_patterns(query, self._lookup_patterns)
        needs_rag = self._match_patterns(query, self._rag_patterns)

        if needs_evidence:
            return ExecutionPlan(
                steps=[
                    ToolStep(tool=ToolType.RAG_SEARCH),
                    ToolStep(tool=ToolType.API_LOOKUP),
                    ToolStep(tool=ToolType.APPEND_EVIDENCE, depends_on=ToolType.API_LOOKUP.value),
                ],
                reason="근거/출처 보강 요청으로 판단",
            )

        if needs_lookup and not needs_draft and not needs_rag:
            return ExecutionPlan(
                steps=[ToolStep(tool=ToolType.API_LOOKUP)],
                reason="외부 민원분석/통계/보고서 조회 요청으로 판단",
            )

        if needs_draft or has_context or needs_rag:
            return ExecutionPlan(
                steps=[
                    ToolStep(tool=ToolType.RAG_SEARCH),
                    ToolStep(tool=ToolType.API_LOOKUP),
                    ToolStep(
                        tool=ToolType.DRAFT_CIVIL_RESPONSE,
                        depends_on=ToolType.API_LOOKUP.value,
                    ),
                ],
                reason="민원 답변 작성 또는 수정 작업으로 판단",
            )

        return ExecutionPlan(
            steps=[
                ToolStep(tool=ToolType.RAG_SEARCH),
                ToolStep(tool=ToolType.API_LOOKUP),
                ToolStep(
                    tool=ToolType.DRAFT_CIVIL_RESPONSE,
                    depends_on=ToolType.API_LOOKUP.value,
                ),
            ],
            reason="명시적 신호가 약해 기본 drafting loop로 처리",
        )

    @staticmethod
    def _match_patterns(text: str, patterns: List[re.Pattern]) -> bool:
        return any(pattern.search(text) for pattern in patterns)
