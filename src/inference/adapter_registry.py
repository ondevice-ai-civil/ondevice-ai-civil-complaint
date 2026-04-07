"""LoRA 어댑터 레지스트리.

YAML 설정(`config/adapters.yaml`)에서 어댑터 메타데이터를 로드하고,
환경변수 ``ADAPTER_PATHS`` 로 경로를 오버라이드할 수 있다.
tool definition 용 enum/description 생성과 vLLM ``LoRARequest`` 팩토리를 제공한다.

사용법::

    registry = AdapterRegistry.get_instance()
    lora_req = registry.get_lora_request("civil")
    enum_list = registry.build_adapter_enum()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:
    from vllm.lora.request import LoRARequest
except ImportError:
    LoRARequest = None  # type: ignore[assignment,misc]

# 프로젝트 루트 경로 (src/inference/adapter_registry.py → ../../..)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "adapters.yaml"


# ---------------------------------------------------------------------------
# 어댑터 메타데이터 데이터클래스
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdapterMeta:
    """단일 LoRA 어댑터의 메타데이터."""

    name: str
    path: str
    description: str
    domain: str
    lora_id: int


# ---------------------------------------------------------------------------
# 레지스트리
# ---------------------------------------------------------------------------

class AdapterRegistry:
    """LoRA 어댑터 레지스트리 (싱글톤).

    ``config/adapters.yaml`` 에서 어댑터 정보를 로드하며,
    ``ADAPTER_PATHS`` 환경변수가 설정되어 있으면 해당 경로로 오버라이드한다.
    """

    _instance: Optional[AdapterRegistry] = None

    def __init__(self, config_path: Path = _DEFAULT_CONFIG) -> None:
        self._adapters: Dict[str, AdapterMeta] = {}
        self._load(config_path)

    # ------------------------------------------------------------------
    # 싱글톤 접근자
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, config_path: Path = _DEFAULT_CONFIG) -> AdapterRegistry:
        """앱 전체에서 공유되는 싱글톤 인스턴스를 반환한다."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """싱글톤 인스턴스를 초기화한다 (테스트 용도)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # 내부 로드
    # ------------------------------------------------------------------

    def _load(self, config_path: Path) -> None:
        """YAML 설정 파일과 환경변수에서 어댑터 목록을 구성한다."""
        raw_adapters: Dict[str, Dict[str, Any]] = {}

        # 1) YAML 로드
        if config_path.exists():
            if yaml is None:
                logger.warning(
                    "PyYAML이 설치되지 않아 어댑터 설정을 로드할 수 없음: {}",
                    config_path,
                )
            else:
                try:
                    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                    raw_adapters = data.get("adapters", {})
                    logger.info("어댑터 설정 로드 완료: {} ({}건)", config_path, len(raw_adapters))
                except Exception as exc:  # noqa: BLE001
                    logger.error("어댑터 설정 로드 실패: {} — {}", config_path, exc)
        else:
            logger.debug("어댑터 설정 파일 없음, 빈 레지스트리로 동작: {}", config_path)

        # 2) ADAPTER_PATHS 환경변수로 path 오버라이드
        env_overrides = self._parse_adapter_paths(os.getenv("ADAPTER_PATHS", ""))
        for name, path_override in env_overrides.items():
            if name in raw_adapters:
                raw_adapters[name]["path"] = path_override
                logger.info("ADAPTER_PATHS 오버라이드: {} → {}", name, path_override)
            else:
                # 환경변수에만 존재하는 어댑터는 최소 메타데이터로 등록
                raw_adapters[name] = {
                    "path": path_override,
                    "description": f"{name} adapter (env)",
                    "domain": name,
                }
                logger.info("ADAPTER_PATHS에서 새 어댑터 등록: {} → {}", name, path_override)

        # 3) sorted 이름 순으로 lora_id 부여 (1부터)
        for idx, name in enumerate(sorted(raw_adapters.keys()), start=1):
            meta = raw_adapters[name]
            self._adapters[name] = AdapterMeta(
                name=name,
                path=meta.get("path", ""),
                description=meta.get("description", ""),
                domain=meta.get("domain", ""),
                lora_id=idx,
            )

        if self._adapters:
            id_map = {a.name: a.lora_id for a in self._adapters.values()}
            logger.info("LoRA ID 매핑: {}", id_map)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def get_lora_request(self, name: str) -> Any:
        """vLLM ``LoRARequest`` 를 동적 생성한다.

        vllm 미설치 환경이거나 존재하지 않는 어댑터 이름이면 ``None`` 을 반환한다.
        """
        if LoRARequest is None:
            return None
        adapter = self._adapters.get(name)
        if adapter is None:
            logger.warning("존재하지 않는 어댑터 요청: {}", name)
            return None
        return LoRARequest(
            lora_name=adapter.name,
            lora_int_id=adapter.lora_id,
            lora_path=adapter.path,
        )

    def build_adapter_enum(self) -> List[str]:
        """tool definition 용 enum 리스트를 반환한다 (어댑터 이름들 + ``"none"``)."""
        return sorted(self._adapters.keys()) + ["none"]

    def build_adapter_description(self) -> str:
        """tool definition 용 description 문자열을 생성한다.

        각 어댑터의 ``name: description`` 형식으로 조합한다.
        """
        lines: List[str] = []
        for name in sorted(self._adapters.keys()):
            adapter = self._adapters[name]
            lines.append(f"{name}: {adapter.description}")
        lines.append('none: 기본 모델 사용 (어댑터 미적용)')
        return "\n".join(lines)

    def list_available(self) -> List[str]:
        """활성 어댑터 이름 목록을 반환한다 (정렬 순)."""
        return sorted(self._adapters.keys())

    def get_meta(self, name: str) -> Optional[AdapterMeta]:
        """어댑터 메타데이터를 반환한다. 없으면 ``None``."""
        return self._adapters.get(name)

    def get_lora_id(self, name: str) -> Optional[int]:
        """어댑터의 LoRA ID를 반환한다. 없으면 ``None``."""
        adapter = self._adapters.get(name)
        return adapter.lora_id if adapter else None

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_adapter_paths(raw: str) -> Dict[str, str]:
        """``ADAPTER_PATHS`` 환경변수를 파싱한다.

        형식: ``"civil=/path/to/civil,legal=/path/to/legal"``
        반환: ``{"civil": "/path/to/civil", "legal": "/path/to/legal"}``
        잘못된 항목은 경고 후 무시한다.
        """
        if not raw or not raw.strip():
            return {}
        result: Dict[str, str] = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                logger.warning("ADAPTER_PATHS 항목 형식 오류 (name=path 필요): {!r}", entry)
                continue
            name, path = entry.split("=", 1)
            name, path = name.strip(), path.strip()
            if not name or not path:
                logger.warning("ADAPTER_PATHS 항목에 빈 이름 또는 경로: {!r}", entry)
                continue
            result[name] = path
        return result

    def __repr__(self) -> str:
        names = ", ".join(self.list_available())
        return f"AdapterRegistry([{names}])"
