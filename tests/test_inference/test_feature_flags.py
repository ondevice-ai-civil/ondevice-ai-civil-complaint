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
        assert flags.use_rag_pipeline is True
        assert flags.model_version == "v2_lora"


class TestFromEnvCustom:
    """커스텀 환경변수 설정을 확인한다."""

    def test_from_env_custom(self):
        with patch.dict(
            os.environ,
            {"USE_RAG_PIPELINE": "false", "MODEL_VERSION": "v1_lora"},
            clear=True,
        ):
            flags = FeatureFlags.from_env()
        assert flags.use_rag_pipeline is False
        assert flags.model_version == "v1_lora"


class TestFromEnvTruthyFalsy:
    """다양한 truthy/falsy 값을 테스트한다."""

    @pytest.mark.parametrize("value", ["true", "1", "yes"])
    def test_from_env_various_truthy(self, value: str):
        with patch.dict(os.environ, {"USE_RAG_PIPELINE": value}, clear=True):
            flags = FeatureFlags.from_env()
        assert flags.use_rag_pipeline is True

    @pytest.mark.parametrize("value", ["false", "0", "no"])
    def test_from_env_various_falsy(self, value: str):
        with patch.dict(os.environ, {"USE_RAG_PIPELINE": value}, clear=True):
            flags = FeatureFlags.from_env()
        assert flags.use_rag_pipeline is False


class TestOverrideFromHeader:
    """X-Feature-Flag 헤더 오버라이드를 테스트한다."""

    def test_override_from_header_none(self):
        flags = FeatureFlags()
        result = flags.override_from_header(None)
        assert result is flags  # 원본 그대로 반환

    def test_override_from_header_empty(self):
        flags = FeatureFlags()
        result = flags.override_from_header("")
        assert result is flags

    def test_override_from_header_single(self):
        flags = FeatureFlags(use_rag_pipeline=True, model_version="v2_lora")
        result = flags.override_from_header("USE_RAG_PIPELINE=false")
        assert result.use_rag_pipeline is False
        assert result.model_version == "v2_lora"  # 변경되지 않음

    def test_override_from_header_multiple(self):
        flags = FeatureFlags(use_rag_pipeline=True, model_version="v2_lora")
        result = flags.override_from_header("USE_RAG_PIPELINE=false,MODEL_VERSION=v1_lora")
        assert result.use_rag_pipeline is False
        assert result.model_version == "v1_lora"

    def test_override_from_header_with_spaces(self):
        flags = FeatureFlags()
        result = flags.override_from_header(" USE_RAG_PIPELINE = false , MODEL_VERSION = v1_lora ")
        assert result.use_rag_pipeline is False
        assert result.model_version == "v1_lora"

    def test_override_from_header_invalid_model_version(self):
        flags = FeatureFlags(model_version="v2_lora")
        result = flags.override_from_header("MODEL_VERSION=v3_invalid")
        assert result.model_version == "v2_lora"  # 잘못된 값은 무시

    def test_override_from_header_invalid_format(self):
        flags = FeatureFlags()
        result = flags.override_from_header("INVALID_NO_EQUALS")
        assert result is flags  # 파싱 불가능하면 원본 반환


class TestImmutability:
    """frozen=True dataclass의 불변성을 확인한다."""

    def test_override_immutable(self):
        original = FeatureFlags(use_rag_pipeline=True, model_version="v2_lora")
        overridden = original.override_from_header("USE_RAG_PIPELINE=false")

        # 원본은 변경되지 않는다
        assert original.use_rag_pipeline is True
        assert original.model_version == "v2_lora"

        # 오버라이드된 인스턴스는 새 값을 가진다
        assert overridden.use_rag_pipeline is False
        assert overridden.model_version == "v2_lora"

    def test_frozen_cannot_set_attribute(self):
        flags = FeatureFlags()
        with pytest.raises(AttributeError):
            flags.use_rag_pipeline = False  # type: ignore[misc]


class TestHealthEndpointFeatureFlags:
    """Feature Flags가 /health 응답 구조에 포함될 수 있는지 검증한다."""

    def test_api_health_includes_feature_flags(self):
        """FeatureFlags를 dict로 변환하여 /health 응답 형태를 구성할 수 있는지 확인한다."""
        flags = FeatureFlags(use_rag_pipeline=True, model_version="v2_lora")
        health_response = {
            "status": "healthy",
            "feature_flags": {
                "use_rag_pipeline": flags.use_rag_pipeline,
                "model_version": flags.model_version,
            },
        }
        assert "feature_flags" in health_response
        assert health_response["feature_flags"]["use_rag_pipeline"] is True
        assert health_response["feature_flags"]["model_version"] == "v2_lora"
