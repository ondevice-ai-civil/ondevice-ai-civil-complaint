# TDD: Red-Green-Refactor cycle로 구현됨
# TDD Phase: RED - 테스트 먼저 작성
"""ResponseFormatter 단위 테스트.

응답 포맷 표준화의 핵심 기능을 테스트한다:
- 성공/실패 응답 표준 JSON 래핑
- 에러 코드 체계
- 메타데이터 (request_id, timestamp, latency_ms)
- thought 블록 제거
"""

import time
import uuid

import pytest

from src.inference.response_formatter import (
    ErrorCode,
    ResponseFormatter,
    StandardResponse,
)


@pytest.fixture
def formatter() -> ResponseFormatter:
    return ResponseFormatter()


class TestSuccessResponse:
    """성공 응답 포맷."""

    def test_success_response_structure(self, formatter: ResponseFormatter):
        response = formatter.success(
            data={"text": "답변입니다"},
            request_id="req-123",
        )
        assert response.success is True
        assert response.data == {"text": "답변입니다"}
        assert response.error is None

    def test_success_response_has_metadata(self, formatter: ResponseFormatter):
        response = formatter.success(data={}, request_id="req-123")
        assert response.metadata.request_id == "req-123"
        assert response.metadata.timestamp is not None

    def test_success_response_latency(self, formatter: ResponseFormatter):
        start = time.time()
        response = formatter.success(data={}, request_id="req-1", start_time=start)
        assert response.metadata.latency_ms >= 0


class TestErrorResponse:
    """에러 응답 포맷."""

    def test_error_response_structure(self, formatter: ResponseFormatter):
        response = formatter.error(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="잘못된 입력입니다",
            request_id="req-err-1",
        )
        assert response.success is False
        assert response.error.code == "VALIDATION_ERROR"
        assert response.error.message == "잘못된 입력입니다"
        assert response.data is None

    def test_error_response_has_metadata(self, formatter: ResponseFormatter):
        response = formatter.error(
            error_code=ErrorCode.MODEL_ERROR,
            message="모델 오류",
            request_id="req-err-2",
        )
        assert response.metadata.request_id == "req-err-2"


class TestErrorCodes:
    """에러 코드 체계."""

    def test_validation_error_code(self):
        assert ErrorCode.VALIDATION_ERROR == "VALIDATION_ERROR"

    def test_model_error_code(self):
        assert ErrorCode.MODEL_ERROR == "MODEL_ERROR"

    def test_search_error_code(self):
        assert ErrorCode.SEARCH_ERROR == "SEARCH_ERROR"

    def test_rate_limit_error_code(self):
        assert ErrorCode.RATE_LIMIT_ERROR == "RATE_LIMIT_ERROR"

    def test_internal_error_code(self):
        assert ErrorCode.INTERNAL_ERROR == "INTERNAL_ERROR"

    def test_auth_error_code(self):
        assert ErrorCode.AUTH_ERROR == "AUTH_ERROR"


class TestThoughtBlockRemoval:
    """thought 블록 제거."""

    def test_removes_think_block(self, formatter: ResponseFormatter):
        raw = "<think>내부 추론 과정...</think>실제 답변입니다."
        cleaned = formatter.clean_response(raw)
        assert "내부 추론 과정" not in cleaned
        assert cleaned == "실제 답변입니다."

    def test_removes_nested_think_block(self, formatter: ResponseFormatter):
        raw = "<think>생각 중\n여러 줄\n추론</think>답변"
        cleaned = formatter.clean_response(raw)
        assert cleaned == "답변"

    def test_no_think_block_unchanged(self, formatter: ResponseFormatter):
        raw = "일반적인 답변입니다."
        cleaned = formatter.clean_response(raw)
        assert cleaned == "일반적인 답변입니다."

    def test_strips_whitespace_after_removal(self, formatter: ResponseFormatter):
        raw = "<think>추론</think>  \n  답변"
        cleaned = formatter.clean_response(raw)
        assert cleaned == "답변"


class TestStandardResponseSerialization:
    """StandardResponse JSON 직렬화."""

    def test_to_dict(self, formatter: ResponseFormatter):
        response = formatter.success(data={"key": "value"}, request_id="req-1")
        d = response.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is True
        assert "metadata" in d

    def test_auto_request_id_generation(self, formatter: ResponseFormatter):
        response = formatter.success(data={})
        assert response.metadata.request_id is not None
        # UUID 형식인지 확인
        uuid.UUID(response.metadata.request_id)
