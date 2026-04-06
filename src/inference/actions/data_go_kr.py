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
    _HttpxTimeoutError = httpx.TimeoutException
    _HttpxStatusError = httpx.HTTPStatusError
except ImportError:
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False
    _HttpxTimeoutError = type(None)  # 절대 매치되지 않는 타입
    _HttpxStatusError = type(None)


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_BASE_URL = "http://apis.data.go.kr/1140100/minAnalsInfoView5"
_ENDPOINT_SIMILAR = "/minSimilarInfo5"


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
        ret_count: Optional[int] = None,
        min_score: Optional[int] = None,
    ) -> Dict[str, Any]:
        """유사 민원 사례 검색에 필요한 payload를 구성한다.

        api_lookup capability 내부에서 minSimilarInfo5 호출 경로를
        공용으로 재사용할 수 있도록 공개 helper로 제공한다.

        Parameters
        ----------
        ret_count : Optional[int]
            반환 건수 오버라이드.
        min_score : Optional[int]
            최소 유사도 오버라이드.
        """
        search_query = self._enrich_query(query, context)
        logger.debug(f"[minwon_analysis] 보강된 검색어: {search_query!r}")
        items = await self._call_similar_api(search_query, ret_count=ret_count, min_score=min_score)

        return {
            "query": search_query,
            "results": items,
            "count": len(items or []),
            "context_text": self._build_context_text(items or [], query) if items else "",
            "citations": self._build_citations(items or []),
        }

    async def _call_similar_api(
        self,
        search_query: str,
        ret_count: Optional[int] = None,
        min_score: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """공공데이터포털 유사민원정보 API를 호출한다.

        Parameters
        ----------
        search_query : str
            API에 전달할 검색어.
        ret_count : Optional[int]
            반환 건수 오버라이드. None이면 인스턴스 기본값.
        min_score : Optional[int]
            최소 유사도 오버라이드. None이면 인스턴스 기본값.

        Returns
        -------
        Optional[List[Dict[str, Any]]]
            성공 시 아이템 목록, 실패 시 None.
        """
        url = _BASE_URL + _ENDPOINT_SIMILAR
        params = {
            "serviceKey": self._api_key,
            "startPos": 1,
            "retCount": ret_count if ret_count is not None else self._ret_count,
            "target": "qna,qna_origin",
            "minScore": min_score if min_score is not None else self._min_score,
            "dataType": "json",
            "searchword": search_query,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                body = response.json()
        except _HttpxTimeoutError as exc:
            logger.warning(f"[minwon_analysis] API 타임아웃: {exc}")
            return None
        except _HttpxStatusError as exc:
            logger.warning(f"[minwon_analysis] HTTP 오류 {exc.response.status_code}: {exc}")
            return None
        except Exception as exc:
            logger.error(f"[minwon_analysis] API 호출 오류: {exc}", exc_info=True)
            return None

        # 실제 API는 최상위 배열([]) 또는 returnObject 래핑으로 응답
        if isinstance(body, list):
            return body

        if not isinstance(body, dict):
            logger.warning(f"[minwon_analysis] 예상치 못한 응답 타입: {type(body)}")
            return None

        # returnObject 래핑
        if "returnObject" in body:
            obj = body["returnObject"]
            return obj if isinstance(obj, list) else []

        # 에러 응답 검사 — 성공 코드만 통과
        _SUCCESS_CODES = {"00", "0", "200", ""}
        code = str(body.get("code", body.get("resultCode", "00")))
        if code not in _SUCCESS_CODES:
            logger.warning(
                f"[minwon_analysis] API 에러 (code={code}): {body.get('msg', body.get('resultMsg', ''))}"
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
            title = item.get("title") or item.get("qnaTitle") or ""
            content = item.get("content") or item.get("qnaContent") or item.get("question") or ""
            answer = item.get("answer") or item.get("qnaAnswer") or ""
            category = (
                item.get("category") or item.get("minCategory") or item.get("main_sub_name") or ""
            )
            date = item.get("regDate") or item.get("date") or item.get("create_date") or ""

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
            title = item.get("title") or item.get("qnaTitle") or ""
            url = item.get("url") or item.get("detailUrl") or ""
            date = item.get("regDate") or item.get("date") or item.get("create_date") or ""
            content = item.get("content") or item.get("qnaContent") or item.get("question") or ""
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
        """세션 요약이나 최근 assistant 응답을 반영해 검색어를 보강한다.

        Parameters
        ----------
        query : str
            원본 사용자 쿼리.
        context : Dict[str, Any]
            AgentLoop 누적 컨텍스트.

        Returns
        -------
        str
            보강된 검색어.
        """
        query_variants = context.get("query_variants", {})
        if isinstance(query_variants, dict):
            prepared_query = str(query_variants.get("api_lookup", "")).strip()
            if prepared_query:
                return prepared_query

        session_context = str(context.get("session_context", "")).strip()
        if session_context:
            recent_summary = " ".join(session_context.splitlines()[-2:]).strip()
            if recent_summary and recent_summary not in query:
                return f"{query} {recent_summary[:120]}".strip()
        return query

    # ---------------------------------------------------------------------------
    # 공통 API 호출 헬퍼
    # ---------------------------------------------------------------------------

    async def _call_api(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """공통 API 호출 + 응답 파싱.

        Parameters
        ----------
        endpoint : str
            _BASE_URL 뒤에 붙는 엔드포인트 경로.
        params : Dict[str, Any]
            쿼리 파라미터 (serviceKey, dataType 자동 추가).

        Returns
        -------
        Optional[List[Dict[str, Any]]]
            성공 시 아이템 목록, 실패 시 None.
        """
        if not _HTTPX_AVAILABLE:
            logger.warning("[minwon_analysis] httpx 미설치")
            return None

        url = _BASE_URL + endpoint
        params["serviceKey"] = self._api_key
        params["dataType"] = "json"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                body = response.json()
        except httpx.TimeoutException as exc:
            logger.warning(f"[minwon_analysis] API 타임아웃 ({endpoint}): {exc}")
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning(
                f"[minwon_analysis] HTTP 오류 ({endpoint}) " f"{exc.response.status_code}: {exc}"
            )
            return None
        except Exception as exc:
            logger.error(
                f"[minwon_analysis] API 호출 오류 ({endpoint}): {exc}",
                exc_info=True,
            )
            return None

        # 최상위 배열
        if isinstance(body, list):
            return body

        # dict 래핑
        if isinstance(body, dict):
            if "returnObject" in body:
                obj = body["returnObject"]
                return obj if isinstance(obj, list) else []
            # 에러 코드 화이트리스트 (기존 _call_similar_api와 동일)
            code = str(body.get("code", body.get("resultCode", "00")))
            if code not in ("00", "0", "200", ""):
                logger.warning(
                    f"[minwon_analysis] API 에러 ({endpoint}): code={code}, "
                    f"msg={body.get('msg', body.get('resultMsg', ''))}"
                )
                return None
            # body > items 경로 파싱 시도
            return self._parse_similar_items(body)

        return None

    # ---------------------------------------------------------------------------
    # 이슈 탐지 API (issue_detector)
    # ---------------------------------------------------------------------------

    async def get_rising_keywords(
        self,
        analysis_time: str,
        max_result: int = 10,
        target: str = "pttn,dfpt,saeol",
        main_sub_code: str = "1140100",
    ) -> Optional[List[Dict[str, Any]]]:
        """급증키워드를 조회한다.

        Parameters
        ----------
        analysis_time : str
            분석 시점 (예: "2021050614").
        max_result : int
            최대 결과 수.
        target : str
            대상 채널.
        main_sub_code : str
            기관 코드.
        """
        return await self._call_api(
            "/minRisingKeyword5",
            {
                "analysisTime": analysis_time,
                "maxResult": max_result,
                "target": target,
                "mainSubCode": main_sub_code,
            },
        )

    async def get_today_topics(
        self,
        search_date: str,
        top_n: int = 5,
        target: str = "pttn,dfpt,saeol",
    ) -> Optional[List[Dict[str, Any]]]:
        """오늘 이슈 토픽을 조회한다.

        Parameters
        ----------
        search_date : str
            검색 날짜 (예: "20210506").
        top_n : int
            상위 N개.
        target : str
            대상 채널.
        """
        return await self._call_api(
            "/minTodayTopicInfo5",
            {
                "searchDate": search_date,
                "todayTopicTopN": top_n,
                "target": target,
            },
        )

    async def get_top_keywords_by_period(
        self,
        analysis_time: str,
        period: str = "MONTHLY",
        range_count: int = 1,
        max_result: int = 5,
        target: str = "pttn,dfpt,saeol",
        main_sub_code: str = "1140100",
    ) -> Optional[List[Dict[str, Any]]]:
        """기간별 최다 키워드를 조회한다.

        Parameters
        ----------
        analysis_time : str
            분석 시작 시점 (예: "20210301").
        period : str
            기간 단위 ("DAILY" | "WEEKLY" | "MONTHLY").
        range_count : int
            기간 범위 수.
        max_result : int
            최대 결과 수.
        target : str
            대상 채널.
        main_sub_code : str
            기관 코드.
        """
        return await self._call_api(
            "/minDFTopNKeyword5",
            {
                "target": target,
                "period": period,
                "analysisTime": analysis_time,
                "rangeCount": range_count,
                "maxResult": max_result,
                "mainSubCode": main_sub_code,
            },
        )

    # ---------------------------------------------------------------------------
    # 통계 API (stats_lookup)
    # ---------------------------------------------------------------------------

    async def get_statistics(
        self,
        date_from: str,
        date_to: str,
        period: str = "DAILY",
        target: str = "pttn,dfpt,saeol",
        sort_by: str = "NAME",
        sort_order: str = "false",
    ) -> Optional[List[Dict[str, Any]]]:
        """맞춤형 통계를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        period : str
            기간 단위.
        target : str
            대상 채널.
        sort_by : str
            정렬 기준.
        sort_order : str
            정렬 순서 ("true" 오름차순, "false" 내림차순).
        """
        return await self._call_api(
            "/minStaticsInfo5",
            {
                "target": target,
                "dateFrom": date_from,
                "dateTo": date_to,
                "period": period,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            },
        )

    async def get_trend(
        self,
        date_from: str,
        date_to: str,
        period: str = "DAILY",
        target: str = "pttn,dfpt,saeol",
        sort_by: str = "NAME",
        sort_order: str = "false",
    ) -> Optional[List[Dict[str, Any]]]:
        """민원 트렌드(시계열)를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜시간 (YYYYMMDDHH).
        date_to : str
            종료 날짜시간 (YYYYMMDDHH).
        period : str
            기간 단위.
        target : str
            대상 채널.
        sort_by : str
            정렬 기준.
        sort_order : str
            정렬 순서.
        """
        return await self._call_api(
            "/minTimeSeriseView5",
            {
                "target": target,
                "dateFrom": date_from,
                "dateTo": date_to,
                "period": period,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            },
        )

    async def get_doc_count(
        self,
        date_from: str,
        date_to: str,
        searchword: str,
        target: str = "pttn,dfpt,saeol",
        min_score: int = 70,
        omit_duplicate: bool = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """민원 건수를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        searchword : str
            검색어.
        target : str
            대상 채널.
        min_score : int
            최소 유사도 점수.
        omit_duplicate : bool
            중복 제거 여부.
        """
        return await self._call_api(
            "/minSearchDocCnt5",
            {
                "dateFrom": date_from,
                "dateTo": date_to,
                "target": target,
                "minScore": min_score,
                "searchword": searchword,
                "omitDuplicate": str(omit_duplicate).lower(),
            },
        )

    async def get_org_ranking(
        self,
        date_from: str,
        date_to: str,
        top_n: int = 5,
        target: str = "pttn,dfpt,saeol",
        sort_by: str = "VALUE",
        sort_order: str = "false",
    ) -> Optional[List[Dict[str, Any]]]:
        """기관별 민원 순위를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        top_n : int
            상위 N개.
        target : str
            대상 채널.
        sort_by : str
            정렬 기준.
        sort_order : str
            정렬 순서.
        """
        return await self._call_api(
            "/minMofacetInfo5",
            {
                "topN": top_n,
                "sortBy": sort_by,
                "sortOrder": sort_order,
                "target": target,
                "dateFrom": date_from,
                "dateTo": date_to,
            },
        )

    async def get_region_ranking(
        self,
        date_from: str,
        date_to: str,
        top_n: int = 5,
        target: str = "pttn,dfpt,saeol",
        sort_by: str = "VALUE",
        sort_order: str = "false",
    ) -> Optional[List[Dict[str, Any]]]:
        """지역별 민원 순위를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        top_n : int
            상위 N개.
        target : str
            대상 채널.
        sort_by : str
            정렬 기준.
        sort_order : str
            정렬 순서.
        """
        return await self._call_api(
            "/minMrfacetInfo5",
            {
                "topN": top_n,
                "sortBy": sort_by,
                "sortOrder": sort_order,
                "dateFrom": date_from,
                "dateTo": date_to,
                "target": target,
            },
        )

    # ---------------------------------------------------------------------------
    # 키워드 분석 API (keyword_analyzer)
    # ---------------------------------------------------------------------------

    async def get_core_keywords(
        self,
        date_from: str,
        date_to: str,
        result_count: int = 5,
        target: str = "pttn,dfpt,saeol",
    ) -> Optional[List[Dict[str, Any]]]:
        """핵심 키워드를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        result_count : int
            결과 수.
        target : str
            대상 채널.
        """
        return await self._call_api(
            "/minTopNKeyword5",
            {
                "target": target,
                "dateFrom": date_from,
                "dateTo": date_to,
                "resultCount": result_count,
            },
        )

    async def get_related_words(
        self,
        date_from: str,
        date_to: str,
        searchword: str,
        result_count: int = 5,
        target: str = "pttn,dfpt,saeol",
    ) -> Optional[List[Dict[str, Any]]]:
        """연관어를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        searchword : str
            검색어.
        result_count : int
            결과 수.
        target : str
            대상 채널.
        """
        return await self._call_api(
            "/minWdcloudInfo5",
            {
                "target": target,
                "searchword": searchword,
                "dateFrom": date_from,
                "dateTo": date_to,
                "resultCount": result_count,
            },
        )

    # ---------------------------------------------------------------------------
    # 인구통계 API (demographics_lookup)
    # ---------------------------------------------------------------------------

    async def get_gender_stats(
        self,
        date_from: str,
        date_to: str,
        searchword: str,
        target: str = "pttn",
    ) -> Optional[List[Dict[str, Any]]]:
        """성별 통계를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        searchword : str
            검색어.
        target : str
            대상 채널.
        """
        return await self._call_api(
            "/minPttnStstGndrInfo5",
            {
                "dateFrom": date_from,
                "dateTo": date_to,
                "target": target,
                "searchword": searchword,
            },
        )

    async def get_age_stats(
        self,
        date_from: str,
        date_to: str,
        searchword: str,
        target: str = "pttn",
    ) -> Optional[List[Dict[str, Any]]]:
        """연령별 통계를 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        searchword : str
            검색어.
        target : str
            대상 채널.
        """
        return await self._call_api(
            "/minPttnStstAgeInfo5",
            {
                "dateFrom": date_from,
                "dateTo": date_to,
                "target": target,
                "searchword": searchword,
            },
        )

    async def get_population_ratio(
        self,
        date_from: str,
        date_to: str,
        top_n: int = 5,
        target: str = "pttn,saeol,dfpt",
        period: str = "DAILY",
        sort_by: str = "VALUE",
        sort_order: str = "false",
        date_type: str = "C",
        search_type: str = "REGION",
    ) -> Optional[List[Dict[str, Any]]]:
        """인구대비 민원 비율을 조회한다.

        Parameters
        ----------
        date_from : str
            시작 날짜 (YYYYMMDD).
        date_to : str
            종료 날짜 (YYYYMMDD).
        top_n : int
            상위 N개.
        target : str
            대상 채널.
        period : str
            기간 단위.
        sort_by : str
            정렬 기준.
        sort_order : str
            정렬 순서.
        date_type : str
            날짜 유형 ("C" 접수일, "R" 등록일).
        search_type : str
            검색 유형 ("REGION" 지역별).
        """
        return await self._call_api(
            "/minMrPopltnRtInfo5",
            {
                "target": target,
                "dateFrom": date_from,
                "dateTo": date_to,
                "dateType": date_type,
                "topN": top_n,
                "period": period,
                "sortBy": sort_by,
                "sortOrder": sort_order,
                "searchType": search_type,
            },
        )
