"""Action 추상 베이스 클래스 모듈.

AgentLoop에서 외부 API나 서비스를 호출하는 Action의
공통 인터페이스와 결과 타입을 정의한다.

Issue: #394
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from ..session_context import SessionContext


@dataclass
class Citation:
    """출처 정보.

    API 응답이나 검색 결과의 출처를 표현한다.

    Parameters
    ----------
    title : str
        출처 제목.
    url : str
        출처 URL. 없으면 빈 문자열.
    date : str
        작성/등록 날짜. 없으면 빈 문자열.
    snippet : str
        본문 요약(발췌). 없으면 빈 문자열.
    metadata : Dict[str, Any]
        추가 메타데이터.
    """

    title: str
    url: str = ""
    date: str = ""
    snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "date": self.date,
            "snippet": self.snippet,
            "metadata": self.metadata,
        }


@dataclass
class ActionResult:
    """Action 실행 결과.

    BaseAction.execute()의 반환값으로,
    AgentLoop의 ToolFunction 반환 형식(dict)과 호환된다.

    Parameters
    ----------
    success : bool
        실행 성공 여부.
    data : Dict[str, Any]
        성공 시 페이로드.
    error : Optional[str]
        실패 시 오류 메시지.
    source : str
        결과 출처 식별자 (예: "data.go.kr").
    citations : List[Citation]
        참조된 출처 목록.
    context_text : str
        LLM 프롬프트에 삽입할 컨텍스트 텍스트.
    """

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    source: str = ""
    citations: List[Citation] = field(default_factory=list)
    context_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """AgentLoop ToolFunction 반환 형식(dict)으로 변환."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "source": self.source,
            "citations": [c.to_dict() for c in self.citations],
            "context_text": self.context_text,
        }


class BaseAction(ABC):
    """Action 추상 베이스 클래스.

    AgentLoop의 ToolFunction 시그니처(query, context, session) -> dict와
    호환되는 비동기 callable을 제공한다.

    Parameters
    ----------
    action_name : str
        Action 식별자. 로깅에 사용된다.
    """

    def __init__(self, action_name: str) -> None:
        self._action_name = action_name

    async def __call__(
        self,
        query: str,
        context: Dict[str, Any],
        session: SessionContext,
    ) -> Dict[str, Any]:
        """AgentLoop ToolFunction 시그니처 호환 진입점.

        1. validate()로 사전 검증.
        2. 검증 통과 시 execute() 실행.
        3. ActionResult를 dict로 변환해 반환.
        4. 성공/실패를 로깅한다.

        Parameters
        ----------
        query : str
            사용자 요청 텍스트.
        context : Dict[str, Any]
            AgentLoop 누적 컨텍스트(이전 tool 결과 포함).
        session : SessionContext
            현재 세션 컨텍스트.

        Returns
        -------
        Dict[str, Any]
            ActionResult.to_dict() 결과.
        """
        # 1. 사전 검증
        validation_error = self.validate(query, context, session)
        if validation_error:
            logger.warning(f"[{self._action_name}] 검증 실패: {validation_error}")
            result = ActionResult(
                success=False,
                error=validation_error,
                source=self._action_name,
            )
            return result.to_dict()

        # 2. 실행
        try:
            result = await self.execute(query, context, session)
        except Exception as exc:
            logger.error(
                f"[{self._action_name}] execute() 예외 발생: {exc}",
                exc_info=True,
            )
            result = ActionResult(
                success=False,
                error=f"Action 실행 중 오류: {exc}",
                source=self._action_name,
            )

        # 3. 로깅
        if result.success:
            logger.info(
                f"[{self._action_name}] 성공 "
                f"citations={len(result.citations)} "
                f"context_text_len={len(result.context_text)}"
            )
        else:
            logger.warning(f"[{self._action_name}] 실패: {result.error}")

        return result.to_dict()

    def validate(
        self,
        query: str,
        context: Dict[str, Any],
        session: SessionContext,
    ) -> Optional[str]:
        """실행 전 사전 검증. 오류가 있으면 오류 메시지 문자열을 반환한다.

        기본 구현: 빈 쿼리 검사. 서브클래스에서 super() 호출 후 추가 검증 가능.

        Parameters
        ----------
        query : str
            사용자 요청 텍스트.
        context : Dict[str, Any]
            AgentLoop 누적 컨텍스트.
        session : SessionContext
            현재 세션 컨텍스트.

        Returns
        -------
        Optional[str]
            검증 실패 메시지. None이면 검증 통과.
        """
        if not query or not query.strip():
            return "쿼리가 비어 있습니다."
        return None

    @abstractmethod
    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: SessionContext,
    ) -> ActionResult:
        """Action 실제 실행 로직. 서브클래스에서 구현한다.

        Parameters
        ----------
        query : str
            사용자 요청 텍스트.
        context : Dict[str, Any]
            AgentLoop 누적 컨텍스트.
        session : SessionContext
            현재 세션 컨텍스트.

        Returns
        -------
        ActionResult
            실행 결과.
        """
        ...
