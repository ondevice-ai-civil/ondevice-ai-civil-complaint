"""RuntimeConfig 단위 테스트."""

import os
from unittest.mock import patch

import pytest

from src.inference.runtime_config import (
    GenerationDefaults,
    HealthcheckConfig,
    ModelConfig,
    PathConfig,
    RuntimeConfig,
    ServingProfile,
)


class TestServingProfile:
    def test_enum_values(self):
        assert ServingProfile.LOCAL.value == "local"
        assert ServingProfile.SINGLE.value == "single"
        assert ServingProfile.CONTAINER.value == "container"
        assert ServingProfile.AIRGAP.value == "airgap"


class TestRuntimeConfigFromEnv:
    def test_default_is_local(self):
        """SERVING_PROFILE 미설정 시 local 프로필이 기본값이다."""
        with patch.dict(os.environ, {}, clear=True):
            config = RuntimeConfig.from_env()
        assert config.profile == ServingProfile.LOCAL
        assert config.host == "127.0.0.1"
        assert config.reload is True
        assert config.rate_limit_enabled is False

    def test_single_profile(self):
        """SERVING_PROFILE=single 시 프로덕션 기본값이 적용된다."""
        with patch.dict(os.environ, {"SERVING_PROFILE": "single"}, clear=True):
            config = RuntimeConfig.from_env()
        assert config.profile == ServingProfile.SINGLE
        assert config.host == "0.0.0.0"
        assert config.gpu_utilization == 0.85
        assert config.max_model_len == 8192
        assert config.reload is False
        assert config.rate_limit_enabled is True
        assert config.request_timeout_sec == 60

    def test_container_profile(self):
        """SERVING_PROFILE=container 시 컨테이너 기본값이 적용된다."""
        with patch.dict(os.environ, {"SERVING_PROFILE": "container"}, clear=True):
            config = RuntimeConfig.from_env()
        assert config.profile == ServingProfile.CONTAINER
        assert config.host == "0.0.0.0"
        assert config.reload is False
        assert config.rate_limit_enabled is True
        assert config.request_timeout_sec == 60

    def test_airgap_profile(self):
        """SERVING_PROFILE=airgap 시 폐쇄망 기본값이 적용된다."""
        with patch.dict(os.environ, {"SERVING_PROFILE": "airgap"}, clear=True):
            config = RuntimeConfig.from_env()
        assert config.profile == ServingProfile.AIRGAP
        assert config.request_timeout_sec == 90

    def test_cloud_run_defaults_to_container_profile(self):
        """Cloud Run 환경 마커가 있으면 container 프로필을 자동 선택한다."""
        with patch.dict(os.environ, {"K_SERVICE": "govon-api"}, clear=True):
            config = RuntimeConfig.from_env()
        assert config.profile == ServingProfile.CONTAINER
        assert config.host == "0.0.0.0"
        assert config.reload is False

    def test_explicit_profile_overrides_container_detection(self):
        """SERVING_PROFILE 명시는 자동 감지보다 우선한다."""
        env = {"K_SERVICE": "govon-api", "SERVING_PROFILE": "local"}
        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        assert config.profile == ServingProfile.LOCAL
        assert config.host == "127.0.0.1"

    def test_unknown_profile_falls_back_to_local(self):
        """알 수 없는 프로필은 local로 폴백한다."""
        with patch.dict(os.environ, {"SERVING_PROFILE": "unknown"}, clear=True):
            config = RuntimeConfig.from_env()
        assert config.profile == ServingProfile.LOCAL

    def test_env_override_takes_precedence(self):
        """개별 환경변수가 프로필 기본값을 오버라이드한다."""
        env = {
            "SERVING_PROFILE": "local",
            "GPU_UTILIZATION": "0.95",
            "MAX_MODEL_LEN": "16384",
            "PORT": "9000",
        }
        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        assert config.gpu_utilization == 0.95
        assert config.max_model_len == 16384
        assert config.port == 9000

    def test_skip_model_load(self):
        with patch.dict(os.environ, {"SKIP_MODEL_LOAD": "true"}, clear=True):
            config = RuntimeConfig.from_env()
        assert config.skip_model_load is True

    def test_cors_from_env(self):
        env = {"CORS_ORIGINS": "https://a.com,https://b.com"}
        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        assert config.cors_origins == ["https://a.com", "https://b.com"]

    def test_cors_default_for_local(self):
        with patch.dict(os.environ, {"SERVING_PROFILE": "local"}, clear=True):
            config = RuntimeConfig.from_env()
        assert "http://localhost:3000" in config.cors_origins


class TestModelConfig:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            mc = ModelConfig.from_env()
        assert mc.model_path == "umyunsang/GovOn-EXAONE-AWQ-v2"
        assert mc.trust_remote_code is True
        assert mc.dtype == "half"
        assert mc.enforce_eager is True

    def test_env_override(self):
        env = {
            "MODEL_PATH": "/local/model",
            "MODEL_DTYPE": "float16",
            "ENFORCE_EAGER": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            mc = ModelConfig.from_env()
        assert mc.model_path == "/local/model"
        assert mc.dtype == "float16"
        assert mc.enforce_eager is False


class TestGenerationDefaults:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            gd = GenerationDefaults.from_env()
        assert gd.max_tokens == 512
        assert gd.temperature == 0.7
        assert gd.repetition_penalty == 1.1

    def test_env_override(self):
        env = {"GEN_MAX_TOKENS": "1024", "GEN_TEMPERATURE": "0.3"}
        with patch.dict(os.environ, env, clear=True):
            gd = GenerationDefaults.from_env()
        assert gd.max_tokens == 1024
        assert gd.temperature == 0.3


class TestPathConfig:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            pc = PathConfig.from_env()
        assert pc.data_path == ""
        assert pc.index_path == "models/faiss_index/complaints.index"
        assert pc.local_docs_root == ""

    def test_env_override(self):
        env = {"DATA_PATH": "/opt/data.jsonl", "LOCAL_DOCS_ROOT": "/opt/local-docs"}
        with patch.dict(os.environ, env, clear=True):
            pc = PathConfig.from_env()
        assert pc.data_path == "/opt/data.jsonl"
        assert pc.local_docs_root == "/opt/local-docs"


class TestHealthcheckConfig:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            hc = HealthcheckConfig.from_env()
        assert hc.interval_sec == 30
        assert hc.timeout_sec == 10
        assert hc.endpoint == "/health"

    def test_env_override(self):
        env = {"HEALTH_INTERVAL_SEC": "60", "HEALTH_TIMEOUT_SEC": "5"}
        with patch.dict(os.environ, env, clear=True):
            hc = HealthcheckConfig.from_env()
        assert hc.interval_sec == 60
        assert hc.timeout_sec == 5


class TestToUvicornKwargs:
    def test_local_profile(self):
        with patch.dict(os.environ, {"SERVING_PROFILE": "local"}, clear=True):
            config = RuntimeConfig.from_env()
        kwargs = config.to_uvicorn_kwargs()
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8000
        assert kwargs["reload"] is True
        assert kwargs["log_level"] == "debug"

    def test_prod_profile_no_reload(self):
        with patch.dict(os.environ, {"SERVING_PROFILE": "single"}, clear=True):
            config = RuntimeConfig.from_env()
        kwargs = config.to_uvicorn_kwargs()
        assert "reload" not in kwargs

    def test_container_profile_no_reload(self):
        with patch.dict(os.environ, {"SERVING_PROFILE": "container"}, clear=True):
            config = RuntimeConfig.from_env()
        kwargs = config.to_uvicorn_kwargs()
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 8000
        assert "reload" not in kwargs


class TestLogSummary:
    def test_log_summary_does_not_raise(self):
        """log_summary 호출이 예외 없이 완료된다."""
        with patch.dict(os.environ, {}, clear=True):
            config = RuntimeConfig.from_env()
        config.log_summary()
