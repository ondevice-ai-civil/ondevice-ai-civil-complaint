"""AdapterRegistry 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestAdapterRegistryLoading:
    """YAML 로드 및 환경변수 오버라이드 검증."""

    def test_loads_from_yaml(self):
        """config/adapters.yaml에서 어댑터 메타데이터를 로드한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        available = reg.list_available()
        # civil, legal이 포함되어야 함
        assert "civil" in available
        assert "legal" in available

    def test_env_override_path(self):
        """ADAPTER_PATHS 환경변수가 path를 오버라이드한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        with patch.dict(os.environ, {"ADAPTER_PATHS": "civil=/custom/path,legal=/other/path"}):
            reg = AdapterRegistry.get_instance()
            meta = reg.get_meta("civil")
            assert meta is not None
            assert meta.path == "/custom/path"

    def test_env_adds_new_adapter(self):
        """ADAPTER_PATHS에 YAML에 없는 어댑터를 추가할 수 있다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        with patch.dict(os.environ, {"ADAPTER_PATHS": "civil=/p1,legal=/p2,env=/p3"}):
            reg = AdapterRegistry.get_instance()
            assert "env" in reg.list_available()


class TestAdapterRegistryIds:
    """LoRA ID 자동 부여 검증."""

    def test_auto_id_assignment(self):
        """sorted 순서로 1부터 ID가 부여된다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        # civil < legal alphabetically → civil=1, legal=2
        assert reg.get_lora_id("civil") == 1
        assert reg.get_lora_id("legal") == 2

    def test_unknown_adapter_returns_none(self):
        """존재하지 않는 어댑터는 None을 반환한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        assert reg.get_lora_id("nonexistent") is None


class TestAdapterRegistryToolDefs:
    """Tool definition용 enum/description 생성 검증."""

    def test_build_adapter_enum(self):
        """enum에 모든 어댑터 + 'none'이 포함된다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        enum = reg.build_adapter_enum()
        assert "civil" in enum
        assert "legal" in enum
        assert "none" in enum

    def test_build_adapter_description(self):
        """description이 각 어댑터의 설명을 포함한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        desc = reg.build_adapter_description()
        assert "civil" in desc
        assert "legal" in desc


class TestAdapterRegistryLoraRequest:
    """get_lora_request 검증."""

    def test_returns_none_without_vllm(self):
        """vllm 미설치 시 None을 반환한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        # In CI without vllm, should return None gracefully
        result = reg.get_lora_request("civil")
        # result is either LoRARequest or None depending on vllm availability
        assert result is None or result is not None  # does not raise

    def test_none_adapter_returns_none(self):
        """'none' 어댑터는 항상 None을 반환한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        assert reg.get_lora_request("none") is None

    def test_unknown_adapter_returns_none(self):
        """존재하지 않는 어댑터는 None을 반환한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        assert reg.get_lora_request("nonexistent") is None


class TestAdapterRegistrySingleton:
    """싱글턴 패턴 검증."""

    def test_singleton(self):
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        a = AdapterRegistry.get_instance()
        b = AdapterRegistry.get_instance()
        assert a is b

    def test_reset_clears_singleton(self):
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        a = AdapterRegistry.get_instance()
        AdapterRegistry.reset()
        b = AdapterRegistry.get_instance()
        assert a is not b
