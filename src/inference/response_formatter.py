"""응답 포맷 표준화.

성공/실패 응답을 표준 JSON 구조로 래핑한다:
- 에러 코드 체계
- 응답 메타데이터 (request_id, timestamp, latency_ms)
- thought 블록 제거
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger


class ErrorCode:
    """에러 코드 상수."""

    VALIDATION_ERROR: str = "VALIDATION_ERROR"
    MODEL_ERROR: str = "MODEL_ERROR"
    SEARCH_ERROR: str = "SEARCH_ERROR"
    RATE_LIMIT_ERROR: str = "RATE_LIMIT_ERROR"
    INTERNAL_ERROR: str = "INTERNAL_ERROR"
    AUTH_ERROR: str = "AUTH_ERROR"


@dataclass
class ResponseMetadata:
    """응답 메타데이터."""

    request_id: str
    timestamp: datetime
    latency_ms: Optional[float] = None


@dataclass
class ErrorInfo:
    """에러 정보."""

    code: str
    message: str


@dataclass
class StandardResponse:
    """표준 응답 래퍼."""

    success: bool
    metadata: ResponseMetadata
    data: Optional[Dict[str, Any]] = None
    error: Optional[ErrorInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환한다."""
        result: Dict[str, Any] = {
            "success": self.success,
            "metadata": {
                "request_id": self.metadata.request_id,
                "timestamp": self.metadata.timestamp.isoformat(),
                "latency_ms": self.metadata.latency_ms,
            },
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = {
                "code": self.error.code,
                "message": self.error.message,
            }
        return result


# thought 블록 정규식 (단일 라인 + 멀티 라인)
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

class ResponseFormatter:
    """응답 포맷 표준화기."""

    def success(
        self,
        data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> StandardResponse:
        """성공 응답을 생성한다."""
        rid = request_id or str(uuid.uuid4())
        latency = (time.time() - start_time) * 1000 if start_time else None
        metadata = ResponseMetadata(
            request_id=rid,
            timestamp=datetime.now(timezone.utc),
            latency_ms=latency,
        )
        return StandardResponse(success=True, metadata=metadata, data=data)

    def error(
        self,
        error_code: str,
        message: str,
        request_id: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> StandardResponse:
        """에러 응답을 생성한다."""
        rid = request_id or str(uuid.uuid4())
        latency = (time.time() - start_time) * 1000 if start_time else None
        metadata = ResponseMetadata(
            request_id=rid,
            timestamp=datetime.now(timezone.utc),
            latency_ms=latency,
        )
        return StandardResponse(
            success=False,
            metadata=metadata,
            error=ErrorInfo(code=error_code, message=message),
        )

    def clean_response(self, raw_text: str) -> str:
        """thought 블록을 제거하고 응답을 정리한다."""
        cleaned = _THINK_PATTERN.sub("", raw_text)
        return cleaned.strip()
