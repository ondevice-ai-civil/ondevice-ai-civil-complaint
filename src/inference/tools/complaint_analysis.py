"""
민원분석 API Tool: 공공데이터포털 민원분석 API를 표준 tool 인터페이스로 래핑.

GovOn shell agent가 ``complaint_analysis`` 이름으로 호출하여
공공데이터포털의 민원분석 정보를 조회한다.

API: GET https://apis.data.go.kr/1140100/minAnalsInfoView5
인증: serviceKey 파라미터 (환경변수 DATA_GO_KR_API_KEY)

Issue: #394
"""

from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, List, Optional

import httpx
from loguru import logger
from pydantic import Field, field_validator

from src.inference.tools.base import BaseTool, ToolInput, ToolOutput


# ---------------------------------------------------------------------------
# 입출력 스키마
# ---------------------------------------------------------------------------


class ComplaintAnalysisInput(ToolInput):
    """민원분석 API 입력 스키마.

    Attributes
    ----------
    keyword : str
        검색 키워드 (필수).
    num_of_rows : int
        한 페이지 결과 수 (기본 10, 최대 100).
    page_no : int
        페이지 번호 (기본 1).
    """

    keyword: str = Field(..., min_length=1, max_length=500, description="검색 키워드")
    num_of_rows: int = Field(default=10, gt=0, le=100, description="한 페이지 결과 수")
    page_no: int = Field(default=1, gt=0, description="페이지 번호")

    @field_validator("keyword")
    @classmethod
    def keyword_must_not_be_blank(cls, v: str) -> str:
        """공백만으로 이루어진 keyword를 거부한다."""
        if not v.strip():
            raise ValueError("keyword는 공백만으로 구성될 수 없습니다.")
        return v.strip()


# ---------------------------------------------------------------------------
# 민원분석 Tool
# ---------------------------------------------------------------------------

_API_BASE_URL = "https://apis.data.go.kr/1140100/minAnalsInfoView5"
_TIMEOUT_SECONDS = 10


class ComplaintAnalysisTool(BaseTool):
    """공공데이터포털 민원분석 API를 표준 tool 인터페이스로 제공한다.

    환경변수 ``DATA_GO_KR_API_KEY`` 에 설정된 서비스 키로 인증하며,
    httpx.AsyncClient를 사용하여 비동기로 API를 호출한다.
    """

    name: ClassVar[str] = "complaint_analysis"
    description: ClassVar[str] = (
        "공공데이터포털 민원분석 API를 호출하여 민원 분석 정보를 조회합니다. "
        "키워드를 입력하면 관련 민원분석 결과를 반환합니다."
    )

    def _get_input_schema(self) -> Dict[str, Any]:
        return ComplaintAnalysisInput.model_json_schema()

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        """민원분석 API를 호출하고 결과를 ToolOutput으로 반환한다."""
        if not isinstance(tool_input, ComplaintAnalysisInput):
            return ToolOutput(
                success=False,
                error="잘못된 입력 타입입니다. ComplaintAnalysisInput이 필요합니다.",
            )

        inp: ComplaintAnalysisInput = tool_input

        # API 키 확인
        api_key = os.getenv("DATA_GO_KR_API_KEY")
        if not api_key:
            return ToolOutput(
                success=False,
                error="DATA_GO_KR_API_KEY 환경변수가 설정되지 않았습니다.",
            )

        # API 호출
        params = {
            "serviceKey": api_key,
            "keyword": inp.keyword,
            "numOfRows": str(inp.num_of_rows),
            "pageNo": str(inp.page_no),
            "type": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.get(_API_BASE_URL, params=params)
        except httpx.TimeoutException:
            logger.warning(f"민원분석 API 타임아웃 (keyword={inp.keyword})")
            return ToolOutput(
                success=False,
                error="민원분석 API 요청이 시간 초과되었습니다. 잠시 후 다시 시도해 주세요.",
            )
        except httpx.HTTPError as e:
            logger.error(f"민원분석 API HTTP 오류: {e}")
            return ToolOutput(
                success=False,
                error=f"민원분석 API 통신 오류가 발생했습니다: {type(e).__name__}",
            )

        # HTTP 상태 코드 확인
        if response.status_code != 200:
            logger.warning(
                f"민원분석 API 비정상 응답: status={response.status_code}, "
                f"keyword={inp.keyword}"
            )
            return ToolOutput(
                success=False,
                error=f"민원분석 API 오류 (HTTP {response.status_code})",
            )

        # 응답 파싱
        try:
            body = response.json()
        except Exception:
            logger.error("민원분석 API 응답 JSON 파싱 실패")
            return ToolOutput(
                success=False,
                error="민원분석 API 응답을 파싱할 수 없습니다.",
            )

        items, total_count, page_no, num_of_rows = _parse_api_response(body)

        # LLM 컨텍스트에 적합한 형태로 변환
        results = _transform_items(items)

        return ToolOutput(
            success=True,
            data={
                "keyword": inp.keyword,
                "total_count": total_count,
                "page_no": page_no,
                "num_of_rows": num_of_rows,
                "results": results,
                "result_count": len(results),
            },
        )


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _parse_api_response(body: dict) -> tuple[List[dict], int, int, int]:
    """공공데이터포털 API 응답을 유연하게 파싱한다.

    표준 형식(response > body > items)을 우선 시도하되,
    구조가 다를 경우에도 graceful하게 처리한다.

    Returns
    -------
    tuple[List[dict], int, int, int]
        (items 리스트, totalCount, pageNo, numOfRows)
    """
    # 표준 공공데이터포털 응답 구조
    resp = body.get("response", body)
    resp_body = resp.get("body", resp)

    total_count = resp_body.get("totalCount", 0)
    page_no = resp_body.get("pageNo", 1)
    num_of_rows = resp_body.get("numOfRows", 10)

    # items 추출: 다양한 구조 대응
    raw_items = resp_body.get("items", [])

    if isinstance(raw_items, dict):
        # {"items": {"item": [...]}} 형태
        item_list = raw_items.get("item", [])
        if isinstance(item_list, dict):
            item_list = [item_list]
        items = item_list if isinstance(item_list, list) else []
    elif isinstance(raw_items, list):
        # {"items": [{...}, {...}]} 형태
        # 각 원소가 {"item": {...}} 구조일 수도 있음
        items = []
        for entry in raw_items:
            if isinstance(entry, dict) and "item" in entry and len(entry) == 1:
                items.append(entry["item"])
            else:
                items.append(entry)
    else:
        items = []

    return items, total_count, page_no, num_of_rows


def _transform_items(items: List[dict]) -> List[Dict[str, Any]]:
    """API 응답 항목들을 LLM 컨텍스트에 적합한 구조화된 형태로 변환한다.

    공공데이터포털 민원분석 API 필드명이 다양할 수 있으므로
    핵심 필드를 유연하게 추출한다.
    """
    results: List[Dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        result: Dict[str, Any] = {
            "title": _extract_field(item, ["sj", "title", "minSj", "complaintTitle"]),
            "content": _extract_field(
                item, ["cn", "content", "minCn", "complaintContent", "dtlCn"]
            ),
            "category": _extract_field(
                item, ["ctgry", "category", "minCtgry", "complaintCategory", "upperCtgryNm"]
            ),
            "agency": _extract_field(
                item, ["insttNm", "agency", "orgNm", "chargerInsttNm"]
            ),
            "date": _extract_field(
                item, ["regDt", "date", "registDt", "creatDt", "rceptDt"]
            ),
            "status": _extract_field(
                item, ["sttus", "status", "processSttus", "resultCode"]
            ),
            "url": _extract_field(item, ["url", "dtlUrl", "linkUrl"]),
        }

        # 원본 항목에서 추가 메타데이터 수집 (변환되지 않은 필드)
        known_keys = set()
        for candidates in [
            ["sj", "title", "minSj", "complaintTitle"],
            ["cn", "content", "minCn", "complaintContent", "dtlCn"],
            ["ctgry", "category", "minCtgry", "complaintCategory", "upperCtgryNm"],
            ["insttNm", "agency", "orgNm", "chargerInsttNm"],
            ["regDt", "date", "registDt", "creatDt", "rceptDt"],
            ["sttus", "status", "processSttus", "resultCode"],
            ["url", "dtlUrl", "linkUrl"],
        ]:
            known_keys.update(candidates)

        extras = {k: v for k, v in item.items() if k not in known_keys and v}
        if extras:
            result["metadata"] = extras

        results.append(result)

    return results


def _extract_field(item: dict, candidates: List[str]) -> str:
    """후보 필드명 목록에서 첫 번째로 존재하는 값을 반환한다."""
    for key in candidates:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
