"""Tool 라우팅 모듈.

사용자 요청을 분석하여 어떤 tool(classify, search, generate)을
어떤 순서로 실행할지 결정한다.

Issue: #393
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class ToolType(str, Enum):
    """실행 가능한 tool 타입."""

    CLASSIFY = "classify"
    SEARCH = "search"
    GENERATE = "generate"
    API_LOOKUP = "api_lookup"


@dataclass
class ToolStep:
    """실행 계획의 단일 단계."""

    tool: ToolType
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: Optional[str] = None  # 이전 단계 결과에 의존하는 경우

    @property
    def step_id(self) -> str:
        return self.tool.value


@dataclass
class ExecutionPlan:
    """tool 실행 계획."""

    steps: List[ToolStep]
    reason: str  # 이 계획을 선택한 이유

    @property
    def tool_names(self) -> List[str]:
        return [s.tool.value for s in self.steps]

    def __repr__(self) -> str:
        return f"ExecutionPlan(steps={self.tool_names}, reason={self.reason!r})"


# 분류 관련 키워드 패턴
_CLASSIFY_PATTERNS = [
    r"분류",
    r"카테고리",
    r"어떤\s*종류",
    r"유형.*판단",
    r"어디.*부서",
    r"담당.*부서",
    r"어느.*분야",
]

# 검색 관련 키워드 패턴
_SEARCH_PATTERNS = [
    r"검색",
    r"찾아",
    r"유사.*민원",
    r"비슷한.*사례",
    r"관련.*사례",
    r"조회",
    r"이전.*처리",
    r"선례",
    r"참고.*사례",
]

# 생성(답변 작성) 관련 키워드 패턴
_GENERATE_PATTERNS = [
    r"답변.*작성",
    r"답변.*해줘",
    r"답변.*생성",
    r"초안",
    r"작성.*해줘",
    r"회신",
    r"응답.*작성",
    r"처리.*방안",
    r"안내문",
]

# 외부 API 조회 관련 키워드 패턴 (data.go.kr 민원 분석 API)
_API_LOOKUP_PATTERNS = [
    r"민원.*분석",
    r"유사.*사례.*조회",
    r"공공.*데이터",
    r"data\.go\.kr",
    r"민원.*통계",
    r"민원.*현황",
    r"타\s*기관",
    r"국민신문고",
    r"트렌드",
]


class ToolRouter:
    """사용자 요청을 분석하여 실행 계획을 수립한다.

    키워드 기반 패턴 매칭으로 필요한 tool과 실행 순서를 결정한다.
    classify -> search -> generate 순서를 기본으로 하되,
    명시적 요청이 있으면 해당 단계만 실행한다.
    """

    def __init__(self) -> None:
        self._classify_patterns = [re.compile(p) for p in _CLASSIFY_PATTERNS]
        self._search_patterns = [re.compile(p) for p in _SEARCH_PATTERNS]
        self._generate_patterns = [re.compile(p) for p in _GENERATE_PATTERNS]
        self._api_lookup_patterns = [re.compile(p) for p in _API_LOOKUP_PATTERNS]

    def plan(
        self,
        query: str,
        has_context: bool = False,
        force_tools: Optional[List[ToolType]] = None,
    ) -> ExecutionPlan:
        """요청을 분석하여 실행 계획을 수립한다.

        Parameters
        ----------
        query : str
            사용자 요청 텍스트.
        has_context : bool
            기존 세션 컨텍스트(분류/검색 결과)가 있는지 여부.
        force_tools : Optional[List[ToolType]]
            강제로 실행할 tool 목록. 지정 시 패턴 매칭을 건너뛴다.

        Returns
        -------
        ExecutionPlan
            실행할 tool 단계 목록과 선택 사유.
        """
        if force_tools:
            steps = [ToolStep(tool=t) for t in force_tools]
            return ExecutionPlan(
                steps=steps,
                reason=f"강제 지정된 tool: {[t.value for t in force_tools]}",
            )

        needs_classify = self._match_patterns(query, self._classify_patterns)
        needs_search = self._match_patterns(query, self._search_patterns)
        needs_generate = self._match_patterns(query, self._generate_patterns)
        needs_api_lookup = self._match_patterns(query, self._api_lookup_patterns)

        steps: List[ToolStep] = []
        reasons: List[str] = []

        # 아무 패턴도 매칭되지 않으면 전체 파이프라인 실행 (민원 텍스트로 간주)
        if not needs_classify and not needs_search and not needs_generate and not needs_api_lookup:
            steps = [
                ToolStep(tool=ToolType.CLASSIFY),
                ToolStep(tool=ToolType.SEARCH, depends_on="classify"),
                ToolStep(tool=ToolType.GENERATE, depends_on="search"),
            ]
            return ExecutionPlan(
                steps=steps,
                reason="명시적 키워드 없음 - 전체 파이프라인(classify -> search -> generate) 실행",
            )

        # 매칭된 tool을 순서대로 추가
        if needs_classify:
            steps.append(ToolStep(tool=ToolType.CLASSIFY))
            reasons.append("분류 키워드 감지")

        if needs_search:
            dep = "classify" if needs_classify else None
            steps.append(ToolStep(tool=ToolType.SEARCH, depends_on=dep))
            reasons.append("검색 키워드 감지")

        if needs_api_lookup:
            # classify 이후에 배치 (classify가 있으면 depends_on "classify")
            dep = "classify" if needs_classify else None
            steps.append(ToolStep(tool=ToolType.API_LOOKUP, depends_on=dep))
            reasons.append("API 조회 키워드 감지")

        if needs_generate:
            # generate는 마지막 단계에 의존
            last_dep: Optional[str] = None
            if needs_api_lookup:
                last_dep = "api_lookup"
            elif needs_search:
                last_dep = "search"
            elif needs_classify:
                last_dep = "classify"
            steps.append(ToolStep(tool=ToolType.GENERATE, depends_on=last_dep))
            reasons.append("생성 키워드 감지")

        # 검색만 요청된 경우에도 generate는 추가하지 않음
        # 분류만 요청된 경우에도 나머지 단계는 추가하지 않음

        return ExecutionPlan(
            steps=steps,
            reason=" / ".join(reasons),
        )

    @staticmethod
    def _match_patterns(text: str, patterns: List[re.Pattern]) -> bool:
        """텍스트가 패턴 목록 중 하나 이상과 매칭되는지 확인."""
        return any(p.search(text) for p in patterns)
