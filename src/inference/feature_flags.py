"""Feature Flag 관리 모듈.

환경변수 기반 Feature Flag와 X-Feature-Flag 헤더를 통한 요청별 오버라이드를 지원한다.
"""

import os
from dataclasses import asdict, dataclass
from typing import Optional

from loguru import logger


@dataclass(frozen=True)
class FeatureFlags:
    """런타임 Feature Flag 설정."""

    model_version: str = "v2_lora"  # v1_lora | v2_lora

    @classmethod
    def from_env(cls) -> "FeatureFlags":
        """환경변수에서 Feature Flag를 로드한다."""
        flags = cls(
            model_version=os.getenv("MODEL_VERSION", "v2_lora"),
        )
        logger.info(f"Feature Flags 로드: {flags}")
        return flags

    def override_from_header(self, header_value: Optional[str]) -> "FeatureFlags":
        """X-Feature-Flag 헤더에서 런타임 오버라이드.

        형식: 'MODEL_VERSION=v1_lora'
        원본 인스턴스는 변경되지 않으며 새 인스턴스를 반환한다.
        """
        if not header_value:
            return self

        overrides: dict = {}
        for pair in header_value.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip().upper()
            value = value.strip()

            if key == "MODEL_VERSION":
                if value in ("v1_lora", "v2_lora"):
                    overrides["model_version"] = value

        if overrides:
            current = asdict(self)
            current.update(overrides)
            return FeatureFlags(**current)
        return self
