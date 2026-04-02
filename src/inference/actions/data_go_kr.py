"""data.go.kr 민원 분석 API Action 모듈.

공공데이터포털(data.go.kr)의 민원분석정보조회 API를 호출하여
유사 민원 사례를 검색하고 LLM 컨텍스트로 변환한다.

API 문서: https://www.data.go.kr/data/15025759/openapi.do
Issue: #394
"""

import os
from typing import Any, Dict, List, Optional

from loguru import logger

from ..session_context import SessionContext
from .base import ActionResult, BaseAction, Citation

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_BASE_URL = "http://apis.data.go.kr/1140100/minAnalsInfoView5"
_ENDPOINT_SIMILAR = "/minSimilarInfo5"

# 분류 카테고리 → 한글 검색어 매핑
_CATEGORY_KO: Dict[str, str] = {
    "environment": "환경",
    "traffic": "교통",
    "welfare": "복지",
    "safety": "안전",
    "tax": "세금",
    "housing": "주거",
    "education": "교육",
    "health": "보건",
    "culture": "문화",
    "economy": "경제",
    "civil": "민원",
    "administrative": "행정",
    "infrastructure": "기반시설",
    "public_order": "치안",
    "labor": "노동",
}


class MinwonAnalysisAction(BaseAction):
    """공공데이터포털 민원분석정보조회 API Action.

    data.go.kr의 민원분석정보조회 API를 호출하여
    유사 민원 사례를 가져오고 AgentLoop 컨텍스트에 제공한다.

    Parameters
    ----------
    api_key : Optional[str]
        공공데이터포털 API 인증키. None이면 DATA_GO_KR_API_KEY 환경변수에서 로드.
    ret_count : int
        반환할 유사 사례 수. 기본값 5.
    min_score : int
        최소 유사도 점수. 기본값 2.
    timeout : float
        HTTP 요청 타임아웃(초). 기본값 10.0.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        ret_count: int = 5,
        min_score: int = 2,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(action_name="minwon_analysis")
        self._api_key = api_key or os.getenv("DATA_GO_KR_API_KEY", "")
        self._ret_count = ret_count
        self._min_score = min_score
        self._timeout = timeout

    def validate(
        self,
        query: str,
        context: Dict[str, Any],
        session: SessionContext,
    ) -> Optional[str]:
        """API 키와 쿼리 길이를 검증한다.

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
            검증 실패 메시지. None이면 통과.
        """
        base_error = super().validate(query, context, session)
        if base_error:
            return base_error

        if not self._api_key:
            return "DATA_GO_KR_API_KEY 환경변수가 설정되지 않았습니다."

        if len(query.strip()) < 2:
            return "쿼리가 너무 짧습니다 (최소 2자 이상)."

        if not _HTTPX_AVAILABLE:
            return "httpx 패키지가 설치되지 않았습니다. pip install httpx>=0.27.0"

        return None

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: SessionContext,
    ) -> ActionResult:
        """유사 민원 사례를 조회하고 ActionResult로 반환한다.

        1. _enrich_query로 분류 카테고리를 반영한 검색어 생성.
        2. _call_similar_api로 API 호출.
        3. 결과를 파싱하여 ActionResult 생성.

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
            유사 민원 사례와 LLM 컨텍스트가 포함된 결과.
        """
        payload = await self.fetch_similar_cases(query, context)
        items = payload["results"]

        if items is None:
            return ActionResult(
                success=False,
                error="민원 분석 API 호출에 실패했습니다.",
                source="data.go.kr",
            )

        if not items:
            return ActionResult(
                success=True,
                data={"results": [], "query": payload["query"], "count": 0},
                source="data.go.kr",
                context_text="",
            )

        return ActionResult(
            success=True,
            data={
                "results": items,
                "query": payload["query"],
                "count": len(items),
            },
            source="data.go.kr",
            citations=payload["citations"],
            context_text=payload["context_text"],
        )

    async def fetch_similar_cases(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """유사 민원 사례 검색에 필요한 payload를 구성한다.

        search_similar tool과 api_lookup action이 같은 minSimilarInfo5 호출 경로를
        공유할 수 있도록 공개 helper로 제공한다.
        """
        search_query = self._enrich_query(query, context)
        logger.debug(f"[minwon_analysis] 보강된 검색어: {search_query!r}")
        items = await self._call_similar_api(search_query)

        return {
            "query": search_query,
            "results": items,
            "count": len(items or []),
            "context_text": self._build_context_text(items or [], query) if items else "",
            "citations": self._build_citations(items or []),
        }

    async def _call_similar_api(self, search_query: str) -> Optional[List[Dict[str, Any]]]:
        """공공데이터포털 유사민원정보 API를 호출한다.

        Parameters
        ----------
        search_query : str
            API에 전달할 검색어.

        Returns
        -------
        Optional[List[Dict[str, Any]]]
            성공 시 아이템 목록, 실패 시 None.
        """
        url = _BASE_URL + _ENDPOINT_SIMILAR
        params = {
            "serviceKey": self._api_key,
            "startPos": 1,
            "retCount": self._ret_count,
            "target": "qna,qna_origin",
            "minScore": self._min_score,
            "dataType": "json",
            "searchWord": search_query,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                body = response.json()
        except httpx.TimeoutException as exc:
            logger.warning(f"[minwon_analysis] API 타임아웃: {exc}")
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning(
                f"[minwon_analysis] HTTP 오류 {exc.response.status_code}: {exc}"
            )
            return None
        except Exception as exc:
            logger.error(f"[minwon_analysis] API 호출 오류: {exc}", exc_info=True)
            return None

        # resultCode 검사
        result_code = str(body.get("resultCode", "00"))
        if result_code not in ("00", "0", "200"):
            logger.warning(
                f"[minwon_analysis] API resultCode={result_code}: "
                f"{body.get('resultMsg', '')}"
            )
            return None

        return self._parse_similar_items(body)

    def _parse_similar_items(self, raw_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """API 응답에서 아이템 목록을 추출한다.

        배열 형식과 단일 dict 래핑 형식을 모두 처리한다.

        Parameters
        ----------
        raw_body : Dict[str, Any]
            API 전체 응답 JSON.

        Returns
        -------
        List[Dict[str, Any]]
            파싱된 아이템 목록.
        """
        # 최상위 키 탐색: body → items → item 또는 직접 items
        body = raw_body.get("body") or raw_body.get("response", {}).get("body") or raw_body
        items_raw = body.get("items") if isinstance(body, dict) else None

        if items_raw is None:
            logger.debug("[minwon_analysis] 응답에 'items' 키 없음 — 빈 결과 반환")
            return []

        # 배열 vs dict 래핑 처리
        if isinstance(items_raw, list):
            return items_raw
        if isinstance(items_raw, dict):
            item = items_raw.get("item")
            if item is None:
                return []
            if isinstance(item, list):
                return item
            if isinstance(item, dict):
                return [item]

        logger.warning(f"[minwon_analysis] 예상치 못한 items 형식: {type(items_raw)}")
        return []

    def _build_context_text(self, items: List[Dict[str, Any]], query: str) -> str:
        """아이템 목록을 LLM 프롬프트용 컨텍스트 텍스트로 변환한다.

        Parameters
        ----------
        items : List[Dict[str, Any]]
            API에서 반환된 아이템 목록.
        query : str
            원본 사용자 쿼리.

        Returns
        -------
        str
            LLM 프롬프트에 삽입할 텍스트.
        """
        if not items:
            return ""

        lines = [f"### 공공데이터포털 유사 민원 사례 (검색어: {query})\n"]
        for i, item in enumerate(items[:5], 1):
            title = item.get("title", item.get("qnaTitle", ""))
            content = item.get("content", item.get("qnaContent", item.get("question", "")))
            answer = item.get("answer", item.get("qnaAnswer", ""))
            category = item.get("category", item.get("minCategory", ""))
            date = item.get("regDate", item.get("date", ""))

            lines.append(f"{i}. [{category}] {title}")
            if date:
                lines.append(f"   (등록일: {date})")
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"   민원: {preview}")
            if answer:
                ans_preview = answer[:200] + "..." if len(answer) > 200 else answer
                lines.append(f"   답변: {ans_preview}")
            lines.append("")

        return "\n".join(lines)

    def _build_citations(self, items: List[Dict[str, Any]]) -> List[Citation]:
        """아이템 목록에서 Citation 객체 목록을 생성한다.

        Parameters
        ----------
        items : List[Dict[str, Any]]
            API에서 반환된 아이템 목록.

        Returns
        -------
        List[Citation]
            Citation 객체 목록.
        """
        citations = []
        for item in items:
            title = item.get("title", item.get("qnaTitle", ""))
            url = item.get("url", item.get("detailUrl", ""))
            date = item.get("regDate", item.get("date", ""))
            content = item.get("content", item.get("qnaContent", item.get("question", "")))
            snippet = content[:150] + "..." if len(content) > 150 else content

            # 제목 없는 항목은 스킵
            if not title:
                continue

            citations.append(
                Citation(
                    title=title,
                    url=url,
                    date=date,
                    snippet=snippet,
                    metadata={k: v for k, v in item.items() if k not in ("content", "answer")},
                )
            )
        return citations

    def _enrich_query(self, query: str, context: Dict[str, Any]) -> str:
        """classify 결과의 카테고리를 반영하여 검색어를 보강한다.

        Parameters
        ----------
        query : str
            원본 사용자 쿼리.
        context : Dict[str, Any]
            AgentLoop 누적 컨텍스트. classify 결과가 있으면 카테고리를 추가.

        Returns
        -------
        str
            보강된 검색어.
        """
        classify_data = context.get("classify", {})
        if not classify_data:
            return query

        classification = classify_data.get("classification") or {}
        category = classification.get("category", "")
        if not category:
            return query

        ko_category = _CATEGORY_KO.get(category.lower(), "")
        if ko_category and ko_category not in query:
            return f"{ko_category} {query}"

        return query

    # ---------------------------------------------------------------------------
    # 보조 API 메서드 (시그니처만 정의)
    # ---------------------------------------------------------------------------

    async def get_top_keywords(
        self,
        period: str = "monthly",
        category: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """인기 민원 키워드를 조회한다. (미구현)

        Parameters
        ----------
        period : str
            조회 주기 ("daily" | "weekly" | "monthly").
        category : Optional[str]
            필터할 카테고리.
        """
        raise NotImplementedError("get_top_keywords는 아직 구현되지 않았습니다.")

    async def get_classification_info(
        self,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """민원 분류 정보를 조회한다. (미구현)

        Parameters
        ----------
        query : str
            분류할 민원 텍스트.
        """
        raise NotImplementedError("get_classification_info는 아직 구현되지 않았습니다.")

    async def get_trend(
        self,
        category: str,
        start_date: str,
        end_date: str,
    ) -> Optional[Dict[str, Any]]:
        """민원 트렌드를 조회한다. (미구현)

        Parameters
        ----------
        category : str
            트렌드를 조회할 카테고리.
        start_date : str
            시작 날짜 (YYYYMMDD).
        end_date : str
            종료 날짜 (YYYYMMDD).
        """
        raise NotImplementedError("get_trend는 아직 구현되지 않았습니다.")
