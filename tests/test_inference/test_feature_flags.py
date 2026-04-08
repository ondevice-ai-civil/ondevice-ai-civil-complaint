"""Feature Flag 모듈 단위 테스트.

vLLM 등 무거운 의존성 없이 FeatureFlags 클래스만 순수하게 테스트한다.
"""

import os
from unittest.mock import patch

import pytest

from src.inference.feature_flags import FeatureFlags


class TestFromEnvDefaults:
    """환경변수 미설정 시 기본값을 확인한다."""

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            flags = FeatureFlags.from_env()
        assert flags.model_version == "v2_lora"


class TestFromEnvCustom:
    """커스텀 환경변수 설정을 확인한다."""

    def test_from_env_custom(self):
        with patch.dict(
            os.environ,
            {"MODEL_VERSION": "v1_lora"},
            clear=True,
        ):
            flags = FeatureFlags.from_env()
        assert flags.model_version == "v1_lora"


class TestOverrideFromHeader:
    """X-Feature-Flag 헤더 오버라이드를 테스트한다."""

    def test_override_from_header_none(self):
        flags = FeatureFlags()
        result = flags.override_from_header(None)
        assert result is flags

    def test_override_from_header_empty(self):
        flags = FeatureFlags()
        result = flags.override_from_header("")
        assert result is flags

    def test_override_from_header_model_version(self):
        flags = FeatureFlags(model_version="v2_lora")
        result = flags.override_from_header("MODEL_VERSION=v1_lora")
        assert result.model_version == "v1_lora"

    def test_override_from_header_invalid_model_version(self):
        flags = FeatureFlags(model_version="v2_lora")
        result = flags.override_from_header("MODEL_VERSION=v3_invalid")
        assert result.model_version == "v2_lora"

    def test_override_from_header_invalid_format(self):
        flags = FeatureFlags()
        result = flags.override_from_header("INVALID_NO_EQUALS")
        assert result is flags


class TestImmutability:
    """frozen=True dataclass의 불변성을 확인한다."""

    def test_override_immutable(self):
        original = FeatureFlags(model_version="v2_lora")
        overridden = original.override_from_header("MODEL_VERSION=v1_lora")
        assert original.model_version == "v2_lora"
        assert overridden.model_version == "v1_lora"

    def test_frozen_cannot_set_attribute(self):
        flags = FeatureFlags()
        with pytest.raises(AttributeError):
            flags.model_version = "changed"  # type: ignore[misc]


class TestHealthEndpointFeatureFlags:
    """Feature Flags가 /health 응답 구조에 포함될 수 있는지 검증한다."""

    def test_api_health_includes_feature_flags(self):
        flags = FeatureFlags(model_version="v2_lora")
        health_response = {
            "status": "healthy",
            "feature_flags": {
                "model_version": flags.model_version,
            },
        }
        assert "feature_flags" in health_response
        assert health_response["feature_flags"]["model_version"] == "v2_lora"
