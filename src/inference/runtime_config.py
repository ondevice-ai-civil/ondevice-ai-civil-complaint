"""GovOn Runtime 서빙 프로필 및 모델 설정 구성.

환경변수 기반으로 로컬 개발, 단일 서버(프로덕션), 폐쇄망(에어갭) 설치용
프로필을 정의하고, generation defaults와 timeout 설정을 표준화한다.

사용법:
    config = RuntimeConfig.from_env()
    config.log_summary()
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# 프로젝트 루트 경로 (src/inference/runtime_config.py → ../../..)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ServingProfile(str, Enum):
    """서빙 프로필. SERVING_PROFILE 환경변수로 선택한다."""

    LOCAL = "local"  # 로컬 개발 환경
    SINGLE = "single"  # 단일 서버 프로덕션
    CONTAINER = "container"  # Docker / Cloud Run / 오프라인 패키지
    AIRGAP = "airgap"  # 폐쇄망 설치


# ---------------------------------------------------------------------------
# 프로필별 기본값 정의
# ---------------------------------------------------------------------------

_PROFILE_DEFAULTS: Dict[ServingProfile, Dict] = {
    ServingProfile.LOCAL: {
        "host": "127.0.0.1",
        "port": 8000,
        "workers": 1,
        "gpu_utilization": 0.7,
        "max_model_len": 4096,
        "log_level": "DEBUG",
        "reload": True,
        "rate_limit_enabled": False,
        "request_timeout_sec": 120,
        "cors_origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
    },
    ServingProfile.SINGLE: {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "gpu_utilization": 0.85,
        "max_model_len": 8192,
        "log_level": "INFO",
        "reload": False,
        "rate_limit_enabled": True,
        "request_timeout_sec": 60,
        "cors_origins": [],
    },
    ServingProfile.CONTAINER: {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "gpu_utilization": 0.85,
        "max_model_len": 8192,
        "log_level": "INFO",
        "reload": False,
        "rate_limit_enabled": True,
        "request_timeout_sec": 60,
        "cors_origins": [],
    },
    ServingProfile.AIRGAP: {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "gpu_utilization": 0.8,
        "max_model_len": 8192,
        "log_level": "INFO",
        "reload": False,
        "rate_limit_enabled": True,
        "request_timeout_sec": 90,
        "cors_origins": [],
    },
}

_CONTAINER_PLATFORM_ENV_MARKERS = (
    "K_SERVICE",
    "K_REVISION",
    "K_CONFIGURATION",
    "KUBERNETES_SERVICE_HOST",
    "SPACE_ID",  # HuggingFace Spaces
)


def _resolve_serving_profile() -> ServingProfile:
    """환경과 명시값을 기준으로 서빙 프로필을 결정한다."""
    profile_name = os.getenv("SERVING_PROFILE")
    if profile_name:
        try:
            return ServingProfile(profile_name.lower())
        except ValueError:
            logger.warning(f"알 수 없는 SERVING_PROFILE '{profile_name}', 기본값 'local' 사용")
            return ServingProfile.LOCAL

    if any(os.getenv(marker) for marker in _CONTAINER_PLATFORM_ENV_MARKERS):
        logger.info("컨테이너 런타임 환경을 감지하여 'container' 프로필을 사용합니다.")
        return ServingProfile.CONTAINER

    return ServingProfile.LOCAL


# ---------------------------------------------------------------------------
# Generation Defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationDefaults:
    """텍스트 생성 기본 파라미터. 엔드포인트 요청에 값이 없을 때 사용된다."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=lambda: ["[|endofturn|]"])

    @classmethod
    def from_env(cls) -> "GenerationDefaults":
        return cls(
            max_tokens=int(os.getenv("GEN_MAX_TOKENS", "512")),
            temperature=float(os.getenv("GEN_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("GEN_TOP_P", "0.9")),
            repetition_penalty=float(os.getenv("GEN_REPETITION_PENALTY", "1.1")),
        )


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """모델 및 어댑터 설정.

    베이스 모델: LGAI-EXAONE/EXAONE-4.0-32B-AWQ (단일 vLLM 인스턴스, ~20GB VRAM)
    - tool calling 네이티브 지원 (BFCL 65.2)
    - vLLM 서빙 옵션: --enable-auto-tool-choice --tool-call-parser hermes

    Multi-LoRA 어댑터:
    - civil-adapter (LoRA #1): draft_civil_response 용도
      학습 데이터: umyunsang/govon-civil-response-data (74K건), QLoRA on AWQ base
    - legal-adapter (LoRA #2): append_evidence 용도
      학습 데이터: neuralfoundry-coder/korean-legal-instruction-sample (232K건), QLoRA on AWQ base
    - 나머지 capability (rag_search, api_lookup, synthesis 등)는 LoRA 없이 base model 사용

    adapter_paths: vLLM --lora-modules 형식으로 전달할 어댑터 경로 목록.
      예: ["civil-adapter=/path/to/civil", "legal-adapter=/path/to/legal"]
    """

    model_path: str = "LGAI-EXAONE/EXAONE-4.0-32B-AWQ"
    trust_remote_code: bool = True
    dtype: str = "half"
    enforce_eager: bool = True
    # Multi-LoRA: vLLM --lora-modules 형식으로 전달할 어댑터 경로 목록
    # 예: ["civil-adapter=/path/to/civil", "legal-adapter=/path/to/legal"]
    adapter_paths: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            model_path=os.getenv("MODEL_PATH", "LGAI-EXAONE/EXAONE-4.0-32B-AWQ"),
            trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "true").lower()
            in ("true", "1", "yes"),
            dtype=os.getenv("MODEL_DTYPE", "half"),
            enforce_eager=os.getenv("ENFORCE_EAGER", "true").lower() in ("true", "1", "yes"),
        )


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------


@dataclass
class PathConfig:
    """데이터·인덱스·로그 경로 설정."""

    data_path: str = ""
    index_path: str = ""
    faiss_index_dir: str = ""
    bm25_index_dir: str = ""
    local_docs_root: str = ""
    agents_dir: str = ""
    log_dir: str = ""
    cache_dir: str = ""

    @classmethod
    def from_env(cls) -> "PathConfig":
        project_root = str(_PROJECT_ROOT)
        return cls(
            data_path=os.getenv("DATA_PATH", ""),
            index_path=os.getenv("INDEX_PATH", "models/faiss_index/complaints.index"),
            faiss_index_dir=os.getenv("FAISS_INDEX_DIR", "models/faiss_index"),
            bm25_index_dir=os.getenv("BM25_INDEX_DIR", "models/bm25_index"),
            local_docs_root=os.getenv("LOCAL_DOCS_ROOT", ""),
            agents_dir=os.getenv("AGENTS_DIR", os.path.join(project_root, "agents")),
            log_dir=os.getenv("LOG_DIR", os.path.join(project_root, "logs")),
            cache_dir=os.getenv("CACHE_DIR", os.path.join(project_root, ".cache")),
        )


# ---------------------------------------------------------------------------
# Healthcheck Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthcheckConfig:
    """헬스체크 설정. shell client readiness probe 용도."""

    endpoint: str = "/health"
    interval_sec: int = 30
    timeout_sec: int = 10
    startup_probe_path: str = "/health"
    readiness_probe_path: str = "/health"

    @classmethod
    def from_env(cls) -> "HealthcheckConfig":
        return cls(
            interval_sec=int(os.getenv("HEALTH_INTERVAL_SEC", "30")),
            timeout_sec=int(os.getenv("HEALTH_TIMEOUT_SEC", "10")),
        )


# ---------------------------------------------------------------------------
# RuntimeConfig (통합 설정)
# ---------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    """GovOn Runtime 통합 설정.

    SERVING_PROFILE 환경변수에 따라 프로필별 기본값을 로드하고,
    개별 환경변수로 오버라이드할 수 있다.
    """

    # 서빙 프로필
    profile: ServingProfile = ServingProfile.LOCAL

    # 서버 설정
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    log_level: str = "DEBUG"
    reload: bool = True

    # GPU / vLLM 설정
    gpu_utilization: float = 0.7
    max_model_len: int = 4096
    skip_model_load: bool = False

    # 보안
    api_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=list)
    rate_limit_enabled: bool = False

    # 타임아웃
    request_timeout_sec: int = 120

    # 하위 설정 객체
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    generation: GenerationDefaults = field(default_factory=GenerationDefaults)
    healthcheck: HealthcheckConfig = field(default_factory=HealthcheckConfig)

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """환경변수에서 전체 런타임 설정을 로드한다.

        1. SERVING_PROFILE에 따른 프로필 기본값 적용
        2. 개별 환경변수로 오버라이드
        """
        profile = _resolve_serving_profile()
        defaults = _PROFILE_DEFAULTS[profile]

        skip_model_load = os.getenv("SKIP_MODEL_LOAD", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # CORS: 환경변수가 있으면 우선, 없으면 프로필 기본값
        cors_env = os.getenv("CORS_ORIGINS", "")
        if cors_env:
            cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        else:
            cors_origins = defaults["cors_origins"]

        return cls(
            profile=profile,
            host=os.getenv("HOST", defaults["host"]),
            port=int(os.getenv("PORT", str(defaults["port"]))),
            workers=int(os.getenv("WORKERS", str(defaults["workers"]))),
            log_level=os.getenv("LOG_LEVEL", defaults["log_level"]),
            reload=os.getenv("RELOAD", str(defaults["reload"])).lower() in ("true", "1", "yes"),
            gpu_utilization=float(os.getenv("GPU_UTILIZATION", str(defaults["gpu_utilization"]))),
            max_model_len=int(os.getenv("MAX_MODEL_LEN", str(defaults["max_model_len"]))),
            skip_model_load=skip_model_load,
            api_key=os.getenv("API_KEY"),
            cors_origins=cors_origins,
            rate_limit_enabled=os.getenv(
                "RATE_LIMIT_ENABLED", str(defaults["rate_limit_enabled"])
            ).lower()
            in ("true", "1", "yes"),
            request_timeout_sec=int(
                os.getenv("REQUEST_TIMEOUT_SEC", str(defaults["request_timeout_sec"]))
            ),
            model=ModelConfig.from_env(),
            paths=PathConfig.from_env(),
            generation=GenerationDefaults.from_env(),
            healthcheck=HealthcheckConfig.from_env(),
        )

    def log_summary(self) -> None:
        """현재 설정 요약을 로그로 출력한다."""
        logger.info("=" * 60)
        logger.info("GovOn Runtime Configuration")
        logger.info("=" * 60)
        logger.info(f"  Profile       : {self.profile.value}")
        logger.info(f"  Host          : {self.host}:{self.port}")
        logger.info(f"  Workers       : {self.workers}")
        logger.info(f"  Log Level     : {self.log_level}")
        logger.info(f"  GPU Util      : {self.gpu_utilization}")
        logger.info(f"  Max Model Len : {self.max_model_len}")
        logger.info(f"  Model Path    : {self.model.model_path}")
        logger.info(f"  Skip Model    : {self.skip_model_load}")
        logger.info(f"  Request Timeout: {self.request_timeout_sec}s")
        logger.info(f"  Rate Limit    : {self.rate_limit_enabled}")
        logger.info(f"  CORS Origins  : {self.cors_origins}")
        logger.info(f"  Healthcheck   : {self.healthcheck.endpoint}")
        logger.info(f"  Data Path     : {self.paths.data_path}")
        logger.info(f"  Index Path    : {self.paths.index_path}")
        logger.info(f"  Local Docs    : {self.paths.local_docs_root or '(disabled)'}")
        logger.info(f"  Log Dir       : {self.paths.log_dir}")
        logger.info("=" * 60)

    def to_uvicorn_kwargs(self) -> Dict:
        """uvicorn.run()에 전달할 키워드 인자를 반환한다."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "log_level": self.log_level.lower(),
            "timeout_keep_alive": self.request_timeout_sec,
        }
        if self.reload:
            kwargs["reload"] = True
        return kwargs
