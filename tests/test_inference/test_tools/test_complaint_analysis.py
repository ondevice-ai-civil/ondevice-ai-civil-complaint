"""
ComplaintAnalysisTool 단위 테스트.

공공데이터포털 민원분석 API를 mock하여
tool 인터페이스의 입출력 계약과 에러 처리를 검증한다.

Issue: #394
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# 무거운 의존성 mock 등록
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("faiss", MagicMock())

import httpx
import pytest

from src.inference.tools.base import ToolInput, ToolOutput
from src.inference.tools.complaint_analysis import (
    ComplaintAnalysisInput,
    ComplaintAnalysisTool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    """ComplaintAnalysisTool 인스턴스."""
    return ComplaintAnalysisTool()


@pytest.fixture
def sample_api_response():
    """공공데이터포털 표준 API 응답 mock 데이터."""
    return {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE"},
            "body": {
                "items": [
                    {
                        "sj": "도로 파손 관련 민원 분석",
                        "cn": "최근 3개월간 도로 파손 민원이 증가하고 있습니다.",
                        "ctgry": "도로/교통",
                        "insttNm": "국토교통부",
                        "regDt": "2025-01-15",
                        "sttus": "완료",
                        "url": "https://example.go.kr/detail/1234",
                    },
                    {
                        "sj": "보도블록 파손 민원 현황",
                        "cn": "보도블록 파손 관련 민원이 전년 대비 20% 증가했습니다.",
                        "ctgry": "도로/교통",
                        "insttNm": "행정안전부",
                        "regDt": "2025-02-01",
                        "sttus": "진행중",
                    },
                ],
                "totalCount": 42,
                "pageNo": 1,
                "numOfRows": 10,
            },
        }
    }


def _make_mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> httpx.Response:
    """httpx.Response mock 객체를 생성한다."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data or {}
    return response


# ---------------------------------------------------------------------------
# 1. 성공 케이스
# ---------------------------------------------------------------------------


class TestComplaintAnalysisSuccess:
    @pytest.mark.asyncio
    async def test_successful_api_call(self, tool, sample_api_response):
        """httpx 응답 mock -> 정상 결과 반환."""
        mock_response = _make_mock_response(200, sample_api_response)

        with patch.dict("os.environ", {"DATA_GO_KR_API_KEY": "test-api-key"}):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                inp = ComplaintAnalysisInput(keyword="도로 파손")
                output = await tool.run(inp)

        assert output.success is True
        assert output.tool_name == "complaint_analysis"
        assert output.data["keyword"] == "도로 파손"
        assert output.data["total_count"] == 42
        assert output.data["result_count"] == 2

        first = output.data["results"][0]
        assert first["title"] == "도로 파손 관련 민원 분석"
        assert "도로 파손 민원이 증가" in first["content"]
        assert first["category"] == "도로/교통"
        assert first["agency"] == "국토교통부"
        assert first["date"] == "2025-01-15"
        assert first["url"] == "https://example.go.kr/detail/1234"


# ---------------------------------------------------------------------------
# 2. 타임아웃 케이스
# ---------------------------------------------------------------------------


class TestComplaintAnalysisTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, tool):
        """httpx.TimeoutException mock -> success=False, 적절한 에러 메시지."""
        with patch.dict("os.environ", {"DATA_GO_KR_API_KEY": "test-api-key"}):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                inp = ComplaintAnalysisInput(keyword="테스트")
                output = await tool.run(inp)

        assert output.success is False
        assert "시간 초과" in output.error


# ---------------------------------------------------------------------------
# 3. API 실패 케이스
# ---------------------------------------------------------------------------


class TestComplaintAnalysisAPIFailure:
    @pytest.mark.asyncio
    async def test_http_500_returns_error(self, tool):
        """HTTP 500 mock -> success=False, 적절한 에러 메시지."""
        mock_response = _make_mock_response(500)

        with patch.dict("os.environ", {"DATA_GO_KR_API_KEY": "test-api-key"}):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                inp = ComplaintAnalysisInput(keyword="테스트")
                output = await tool.run(inp)

        assert output.success is False
        assert "500" in output.error

    @pytest.mark.asyncio
    async def test_http_error_returns_error(self, tool):
        """httpx.HTTPError 발생 시 success=False."""
        with patch.dict("os.environ", {"DATA_GO_KR_API_KEY": "test-api-key"}):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(
                    side_effect=httpx.ConnectError("connection refused")
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                inp = ComplaintAnalysisInput(keyword="테스트")
                output = await tool.run(inp)

        assert output.success is False
        assert "통신 오류" in output.error


# ---------------------------------------------------------------------------
# 4. API 키 미설정
# ---------------------------------------------------------------------------


class TestComplaintAnalysisNoAPIKey:
    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self, tool):
        """DATA_GO_KR_API_KEY 미설정 시 에러."""
        with patch.dict("os.environ", {}, clear=True):
            # 기존 환경변수에서 DATA_GO_KR_API_KEY 제거 보장
            import os

            orig = os.environ.pop("DATA_GO_KR_API_KEY", None)
            try:
                inp = ComplaintAnalysisInput(keyword="테스트")
                output = await tool.run(inp)
            finally:
                if orig is not None:
                    os.environ["DATA_GO_KR_API_KEY"] = orig

        assert output.success is False
        assert "DATA_GO_KR_API_KEY" in output.error


# ---------------------------------------------------------------------------
# 5. 입력 검증
# ---------------------------------------------------------------------------


class TestComplaintAnalysisInputValidation:
    def test_empty_keyword_rejected(self):
        """빈 keyword 거부."""
        with pytest.raises(Exception):
            ComplaintAnalysisInput(keyword="")

    def test_blank_keyword_rejected(self):
        """공백만 있는 keyword 거부."""
        with pytest.raises(Exception):
            ComplaintAnalysisInput(keyword="   ")

    def test_valid_input(self):
        """정상 입력 검증."""
        inp = ComplaintAnalysisInput(keyword="도로 파손", num_of_rows=20, page_no=2)
        assert inp.keyword == "도로 파손"
        assert inp.num_of_rows == 20
        assert inp.page_no == 2

    def test_default_values(self):
        """기본값 검증."""
        inp = ComplaintAnalysisInput(keyword="테스트")
        assert inp.num_of_rows == 10
        assert inp.page_no == 1

    def test_num_of_rows_max_100(self):
        """num_of_rows 최대 100 제한."""
        with pytest.raises(Exception):
            ComplaintAnalysisInput(keyword="테스트", num_of_rows=101)

    @pytest.mark.asyncio
    async def test_wrong_input_type(self, tool):
        """잘못된 입력 타입 거부."""
        output = await tool.run(ToolInput())
        assert output.success is False
        assert "잘못된 입력 타입" in output.error


# ---------------------------------------------------------------------------
# 6. 스키마 검증
# ---------------------------------------------------------------------------


class TestComplaintAnalysisSchema:
    def test_get_schema(self, tool):
        """get_schema() 반환값 검증."""
        schema = tool.get_schema()
        assert schema["name"] == "complaint_analysis"
        assert "description" in schema
        assert "parameters" in schema

        params = schema["parameters"]
        assert "properties" in params
        assert "keyword" in params["properties"]
        assert "num_of_rows" in params["properties"]
        assert "page_no" in params["properties"]

    def test_tool_name_and_description(self, tool):
        """tool 이름과 설명 검증."""
        assert tool.name == "complaint_analysis"
        assert "민원분석" in tool.description


# ---------------------------------------------------------------------------
# 7. 응답 파싱 유연성 테스트
# ---------------------------------------------------------------------------


class TestComplaintAnalysisResponseParsing:
    @pytest.mark.asyncio
    async def test_nested_item_format(self, tool):
        """items가 {"item": [...]} 형태인 경우 파싱."""
        nested_response = {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE"},
                "body": {
                    "items": {
                        "item": [
                            {
                                "sj": "중첩 구조 테스트",
                                "cn": "중첩 items 구조입니다.",
                                "ctgry": "테스트",
                            }
                        ]
                    },
                    "totalCount": 1,
                    "pageNo": 1,
                    "numOfRows": 10,
                },
            }
        }
        mock_response = _make_mock_response(200, nested_response)

        with patch.dict("os.environ", {"DATA_GO_KR_API_KEY": "test-api-key"}):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                inp = ComplaintAnalysisInput(keyword="테스트")
                output = await tool.run(inp)

        assert output.success is True
        assert output.data["result_count"] == 1
        assert output.data["results"][0]["title"] == "중첩 구조 테스트"

    @pytest.mark.asyncio
    async def test_elapsed_time_recorded(self, tool, sample_api_response):
        """실행 소요 시간이 기록되는지 확인."""
        mock_response = _make_mock_response(200, sample_api_response)

        with patch.dict("os.environ", {"DATA_GO_KR_API_KEY": "test-api-key"}):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                inp = ComplaintAnalysisInput(keyword="성능 테스트")
                output = await tool.run(inp)

        assert output.elapsed_ms is not None
        assert output.elapsed_ms >= 0
