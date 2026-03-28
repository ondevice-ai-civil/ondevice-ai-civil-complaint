"""
vllm_stabilizer 단위 테스트.

transformers / vllm 의존성을 Mock으로 대체하여 GPU 없이 실행 가능.
apply_transformers_patch()가 modern transformers(4.53.0+) 환경에서
no-op으로 동작함을 검증한다.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 무거운 의존성 mock 등록
# ---------------------------------------------------------------------------
sys.modules.setdefault("torch", MagicMock())

_transformers_mock = types.ModuleType("transformers")
_rope_utils = types.ModuleType("transformers.modeling_rope_utils")
_utils = types.ModuleType("transformers.utils")
_generic = types.ModuleType("transformers.utils.generic")
_modeling_utils = types.ModuleType("transformers.modeling_utils")

_transformers_mock.modeling_rope_utils = _rope_utils
_transformers_mock.utils = _utils
_transformers_mock.modeling_utils = _modeling_utils
_utils.generic = _generic

sys.modules["transformers"] = _transformers_mock
sys.modules["transformers.modeling_rope_utils"] = _rope_utils
sys.modules["transformers.utils"] = _utils
sys.modules["transformers.utils.generic"] = _generic
sys.modules["transformers.modeling_utils"] = _modeling_utils

import importlib

if "src.inference.vllm_stabilizer" in sys.modules:
    importlib.reload(sys.modules["src.inference.vllm_stabilizer"])

from src.inference.vllm_stabilizer import apply_transformers_patch

# ---------------------------------------------------------------------------
# apply_transformers_patch 테스트 (No-op 검증)
# ---------------------------------------------------------------------------


class TestApplyTransformersPatch:
    def test_patch_is_callable_and_safe(self):
        """apply_transformers_patch가 호출 가능하며 에러가 발생하지 않는다."""
        # 4.53.0+ 환경에서는 아무것도 주입하지 않아야 함 (no-op)
        apply_transformers_patch()
        assert True

    def test_does_not_inject_into_empty_mock(self):
        """비어있는 mock 모듈에 더 이상 속성을 주입하지 않는다."""
        if hasattr(_rope_utils, "RopeParameters"):
            delattr(_rope_utils, "RopeParameters")

        apply_transformers_patch()

        # 몽키 패치가 제거되었으므로 속성이 생기지 않아야 함
        assert not hasattr(_rope_utils, "RopeParameters")


# ---------------------------------------------------------------------------
# start_vllm_engine 테스트
# ---------------------------------------------------------------------------


class TestStartVllmEngine:
    def test_start_vllm_engine_calls_llm(self):
        """start_vllm_engine이 vllm.LLM을 올바른 인자로 호출한다."""
        mock_llm_class = MagicMock()
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_class

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            from src.inference.vllm_stabilizer import start_vllm_engine

            result = start_vllm_engine("test-model-id")

        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs[1]["model"] == "test-model-id"
        assert call_kwargs[1]["trust_remote_code"] is True
        assert call_kwargs[1]["max_model_len"] == 8192
        assert result == mock_llm_instance

    def test_start_vllm_engine_gpu_utilization(self):
        """GPU memory utilization이 0.8로 설정된다."""
        mock_llm_class = MagicMock()
        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_class

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            from src.inference.vllm_stabilizer import start_vllm_engine

            start_vllm_engine("test-model")

        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["gpu_memory_utilization"] == 0.8
        assert call_kwargs["enforce_eager"] is True
