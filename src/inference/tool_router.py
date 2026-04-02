"""Tool 라우팅 모듈.

사용자 요청을 분석하여 어떤 tool(classify, search_similar,
generate_public_doc, generate_civil_response)을 어떤 순서로 실행할지
결정한다.

Issue: #393
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class ToolType(str, Enum):
    """기본 내장 tool 타입.

    런타임 registry는 추가 문자열 tool 이름도 허용한다.
    이 Enum은 기본 제공되는 built-in tool catalog를 표현한다.
    """

    CLASSIFY = "classify"
    SEARCH_SIMILAR = "search_similar"
    GENERATE_PUBLIC_DOC = "generate_public_doc"
    GENERATE_CIVIL_RESPONSE = "generate_civil_response"
    API_LOOKUP = "api_lookup"


ToolName = ToolType | str


def tool_name(tool: ToolName) -> str:
    """ToolType 또는 사용자 정의 tool 이름을 문자열로 정규화한다."""
    return tool.value if isinstance(tool, ToolType) else str(tool)


@dataclass
class ToolStep:
    """실행 계획의 단일 단계."""

    tool: ToolName
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: Optional[str] = None  # 이전 단계 결과에 의존하는 경우

    @property
    def step_id(self) -> str:
        return tool_name(self.tool)


@dataclass
class ExecutionPlan:
    """tool 실행 계획."""

    steps: List[ToolStep]
    reason: str  # 이 계획을 선택한 이유

    @property
    def tool_names(self) -> List[str]:
        return [s.step_id for s in self.steps]

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

# 공문서 생성 관련 키워드 패턴
_GENERATE_PUBLIC_DOC_PATTERNS = [
    r"공문",
    r"공문서",
    r"보도자료",
    r"연설문",
    r"공지사항",
    r"정책보고서",
    r"기안문",
    r"문안",
]

# 민원 답변 생성 관련 키워드 패턴
_GENERATE_CIVIL_RESPONSE_PATTERNS = [
    r"답변.*작성",
    r"답변.*해줘",
    r"답변.*생성",
    r"민원.*회신",
    r"회신.*작성",
    r"응답.*작성",
    r"처리.*방안",
    r"안내문",
    r"회신서",
]

# 외부 API 조회 관련 키워드 패턴 (data.go.kr 민원 분석 API)
_API_LOOKUP_PATTERNS = [
    r"민원.*분석",
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
    classify -> search_similar -> generate_civil_response 순서를 기본으로 하되,
    명시적 요청이 있으면 해당 단계만 실행한다.
    """

    def __init__(self) -> None:
        self._classify_patterns = [re.compile(p) for p in _CLASSIFY_PATTERNS]
        self._search_patterns = [re.compile(p) for p in _SEARCH_PATTERNS]
        self._generate_public_doc_patterns = [
            re.compile(p) for p in _GENERATE_PUBLIC_DOC_PATTERNS
        ]
        self._generate_civil_response_patterns = [
            re.compile(p) for p in _GENERATE_CIVIL_RESPONSE_PATTERNS
        ]
        self._api_lookup_patterns = [re.compile(p) for p in _API_LOOKUP_PATTERNS]

    def plan(
        self,
        query: str,
        has_context: bool = False,
        force_tools: Optional[List[ToolName]] = None,
    ) -> ExecutionPlan:
        """요청을 분석하여 실행 계획을 수립한다.

        Parameters
        ----------
        query : str
            사용자 요청 텍스트.
        has_context : bool
            기존 세션 컨텍스트(분류/검색 결과)가 있는지 여부.
        force_tools : Optional[List[ToolName]]
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
                reason=f"강제 지정된 tool: {[tool_name(t) for t in force_tools]}",
            )

        needs_classify = self._match_patterns(query, self._classify_patterns)
        needs_search_similar = self._match_patterns(query, self._search_patterns)
        public_doc_score = self._match_count(query, self._generate_public_doc_patterns)
        civil_response_score = self._match_count(query, self._generate_civil_response_patterns)
        needs_generate_public_doc = public_doc_score > 0
        needs_generate_civil_response = civil_response_score > 0
        needs_api_lookup = self._match_patterns(query, self._api_lookup_patterns)

        if needs_generate_public_doc and needs_generate_civil_response:
            if public_doc_score >= civil_response_score:
                needs_generate_civil_response = False
            else:
                needs_generate_public_doc = False

        steps: List[ToolStep] = []
        reasons: List[str] = []

        # 아무 패턴도 매칭되지 않으면 전체 파이프라인 실행 (민원 텍스트로 간주)
        if (
            not needs_classify
            and not needs_search_similar
            and not needs_generate_public_doc
            and not needs_generate_civil_response
            and not needs_api_lookup
        ):
            steps = [
                ToolStep(tool=ToolType.CLASSIFY),
                ToolStep(tool=ToolType.SEARCH_SIMILAR, depends_on="classify"),
                ToolStep(
                    tool=ToolType.GENERATE_CIVIL_RESPONSE,
                    depends_on="search_similar",
                ),
            ]
            return ExecutionPlan(
                steps=steps,
                reason=(
                    "명시적 키워드 없음 - 전체 파이프라인"
                    "(classify -> search_similar -> generate_civil_response) 실행"
                ),
            )

        # 매칭된 tool을 순서대로 추가
        if needs_classify:
            steps.append(ToolStep(tool=ToolType.CLASSIFY))
            reasons.append("분류 키워드 감지")

        if needs_search_similar:
            dep = "classify" if needs_classify else None
            steps.append(ToolStep(tool=ToolType.SEARCH_SIMILAR, depends_on=dep))
            reasons.append("유사 사례 검색 키워드 감지")

        if needs_api_lookup:
            # classify 이후에 배치 (classify가 있으면 depends_on "classify")
            dep = "classify" if needs_classify else None
            steps.append(ToolStep(tool=ToolType.API_LOOKUP, depends_on=dep))
            reasons.append("API 조회 키워드 감지")

        if needs_generate_public_doc or needs_generate_civil_response:
            # 생성 tool은 마지막 단계에 의존
            last_dep: Optional[str] = None
            if needs_api_lookup:
                last_dep = "api_lookup"
            elif needs_search_similar:
                last_dep = "search_similar"
            elif needs_classify:
                last_dep = "classify"

            if needs_generate_public_doc:
                steps.append(ToolStep(tool=ToolType.GENERATE_PUBLIC_DOC, depends_on=last_dep))
                reasons.append("공문서 생성 키워드 감지")

            if needs_generate_civil_response:
                steps.append(
                    ToolStep(tool=ToolType.GENERATE_CIVIL_RESPONSE, depends_on=last_dep)
                )
                reasons.append("민원 답변 생성 키워드 감지")

        return ExecutionPlan(
            steps=steps,
            reason=" / ".join(reasons),
        )

    @staticmethod
    def _match_patterns(text: str, patterns: List[re.Pattern]) -> bool:
        """텍스트가 패턴 목록 중 하나 이상과 매칭되는지 확인."""
        return any(p.search(text) for p in patterns)

    @staticmethod
    def _match_count(text: str, patterns: List[re.Pattern]) -> int:
        """텍스트와 매칭되는 패턴 개수를 반환한다."""
        return sum(1 for pattern in patterns if pattern.search(text))
