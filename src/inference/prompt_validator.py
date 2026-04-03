"""프롬프트 입력 검증기.

사용자 프롬프트의 유효성을 검증하고 안전하게 정규화한다:
- 빈 프롬프트 거부
- 최대 길이 제한
- EXAONE 특수 토큰 기반 프롬프트 인젝션 탐지
- 유니코드 NFKC 정규화
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from loguru import logger

MAX_PROMPT_LENGTH: int = 4096

# EXAONE 특수 토큰 패턴
_EXAONE_SPECIAL_TOKENS = [
    "[|user|]",
    "[|assistant|]",
    "[|endofturn|]",
    "[|system|]",
    "[|begin_of_text|]",
    "[|end_of_text|]",
]

_INJECTION_PATTERN = re.compile("|".join(re.escape(token) for token in _EXAONE_SPECIAL_TOKENS))


@dataclass
class PromptValidationResult:
    """프롬프트 검증 결과."""

    is_valid: bool
    sanitized_prompt: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class PromptValidator:
    """프롬프트 입력 검증기."""

    def __init__(self, max_length: int = MAX_PROMPT_LENGTH) -> None:
        self._max_length = max_length

    def validate(self, prompt: str) -> PromptValidationResult:
        """프롬프트를 검증하고 정규화된 결과를 반환한다."""
        # 1. 유니코드 NFKC 정규화
        normalized = unicodedata.normalize("NFKC", prompt)

        # 2. 빈 프롬프트 검사
        if not normalized.strip():
            logger.warning("빈 프롬프트 거부")
            return PromptValidationResult(
                is_valid=False,
                sanitized_prompt="",
                error_code="EMPTY_PROMPT",
                error_message="프롬프트가 비어 있습니다.",
            )

        # 3. 최대 길이 검사
        if len(normalized) > self._max_length:
            logger.warning(f"프롬프트 길이 초과: {len(normalized)} > {self._max_length}")
            return PromptValidationResult(
                is_valid=False,
                sanitized_prompt="",
                error_code="PROMPT_TOO_LONG",
                error_message=f"프롬프트가 최대 길이({self._max_length})를 초과합니다.",
            )

        # 4. 프롬프트 인젝션 탐지
        if _INJECTION_PATTERN.search(normalized):
            logger.warning("프롬프트 인젝션 패턴 탐지")
            return PromptValidationResult(
                is_valid=False,
                sanitized_prompt="",
                error_code="PROMPT_INJECTION",
                error_message="프롬프트에 허용되지 않는 특수 토큰이 포함되어 있습니다.",
            )

        # 5. 통과
        return PromptValidationResult(
            is_valid=True,
            sanitized_prompt=normalized,
        )
