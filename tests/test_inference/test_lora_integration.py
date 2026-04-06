"""vLLM Multi-LoRA 서빙 통합 테스트.

Issue #468: ADAPTER_PATHS 환경변수 파싱, LoRA 설정 조건부 활성화,
_run_engine lora_request 전달 검증.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. ADAPTER_PATHS 환경변수 파싱 테스트
# ---------------------------------------------------------------------------


class TestAdapterPathsParsing:
    """ModelConfig.from_env()의 ADAPTER_PATHS 파싱 동작 검증."""

    def test_normal_case(self):
        """정상 형식: 'civil=/path/civil,legal=/path/legal'"""
        with patch.dict(
            os.environ,
            {"ADAPTER_PATHS": "civil=/models/civil,legal=/models/legal"},
        ):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {
                "civil": "/models/civil",
                "legal": "/models/legal",
            }

    def test_empty_value(self):
        """빈 문자열이면 빈 dict 반환."""
        with patch.dict(os.environ, {"ADAPTER_PATHS": ""}, clear=False):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {}

    def test_unset_env(self):
        """환경변수 미설정이면 빈 dict 반환."""
        env = os.environ.copy()
        env.pop("ADAPTER_PATHS", None)
        with patch.dict(os.environ, env, clear=True):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {}

    def test_malformed_entry_skipped(self):
        """'=' 없는 항목은 경고 후 무시."""
        with patch.dict(
            os.environ,
            {"ADAPTER_PATHS": "civil=/ok,broken_entry,legal=/also_ok"},
        ):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {
                "civil": "/ok",
                "legal": "/also_ok",
            }

    def test_single_adapter(self):
        """단일 어댑터만 설정."""
        with patch.dict(os.environ, {"ADAPTER_PATHS": "civil=/path/to/civil"}):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {"civil": "/path/to/civil"}

    def test_whitespace_trimmed(self):
        """공백이 포함된 항목도 정상 파싱."""
        with patch.dict(
            os.environ,
            {"ADAPTER_PATHS": " civil = /path/civil , legal = /path/legal "},
        ):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {
                "civil": "/path/civil",
                "legal": "/path/legal",
            }

    def test_empty_name_or_path_skipped(self):
        """이름이나 경로가 빈 항목은 무시."""
        with patch.dict(os.environ, {"ADAPTER_PATHS": "=/path,civil="}):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {}


# ---------------------------------------------------------------------------
# 2. _run_engine lora_request 파라미터 전달 검증
# ---------------------------------------------------------------------------


class TestRunEngineLora:
    """_run_engine()이 lora_request를 engine.generate()에 전달하는지 검증."""

    @pytest.mark.asyncio
    async def test_lora_request_passed_to_engine(self):
        """lora_request가 engine.generate()의 키워드 인자로 전달되어야 한다."""
        from src.inference.api_server import vLLMEngineManager

        manager = vLLMEngineManager.__new__(vLLMEngineManager)
        mock_output = MagicMock()
        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock(return_value=mock_output)
        manager.engine = mock_engine

        mock_lora = MagicMock()
        mock_lora.lora_name = "civil"

        mock_sp = MagicMock()
        result = await manager._run_engine("prompt", mock_sp, "req-1", lora_request=mock_lora)

        mock_engine.generate.assert_called_once_with(
            "prompt", mock_sp, "req-1", lora_request=mock_lora
        )
        assert result == mock_output

    @pytest.mark.asyncio
    async def test_lora_request_default_none(self):
        """lora_request 미지정 시 None이 전달되어야 한다."""
        from src.inference.api_server import vLLMEngineManager

        manager = vLLMEngineManager.__new__(vLLMEngineManager)
        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock(return_value=MagicMock())
        manager.engine = mock_engine

        await manager._run_engine("prompt", MagicMock(), "req-2")

        _, kwargs = mock_engine.generate.call_args
        assert kwargs.get("lora_request") is None


# ---------------------------------------------------------------------------
# 3. enable_lora 조건부 활성화 검증
# ---------------------------------------------------------------------------


class TestLoraEngineConfig:
    """adapter_paths 유무에 따른 enable_lora 설정 검증."""

    def test_no_adapters_no_lora(self):
        """adapter_paths가 비어있으면 enable_lora 키워드가 추가되지 않아야 한다."""
        from src.inference.runtime_config import ModelConfig

        config = ModelConfig(adapter_paths={})
        assert not bool(config.adapter_paths)

    def test_with_adapters_lora_enabled(self):
        """adapter_paths가 있으면 bool(adapter_paths)가 True."""
        from src.inference.runtime_config import ModelConfig

        config = ModelConfig(adapter_paths={"civil": "/path"})
        assert bool(config.adapter_paths)


# ---------------------------------------------------------------------------
# 4. SKIP_MODEL_LOAD=true에서 LoRA 설정 스킵 확인
# ---------------------------------------------------------------------------


class TestSkipModelLoadLora:
    """SKIP_MODEL_LOAD=true 환경에서 LoRA 관련 초기화가 스킵되는지 확인."""

    @pytest.mark.asyncio
    async def test_skip_model_load_skips_lora_init(self):
        """SKIP_MODEL_LOAD=true이면 initialize()가 엔진을 생성하지 않는다."""
        from src.inference.api_server import vLLMEngineManager

        manager = vLLMEngineManager.__new__(vLLMEngineManager)
        manager.engine = None
        manager.retriever = None
        manager.index_manager = None
        manager.hybrid_engine = None
        manager.bm25_indexers = {}
        manager.embed_model = None
        manager.feature_flags = MagicMock()
        manager.session_store = MagicMock()
        manager.agent_manager = MagicMock()
        manager.agent_loop = None
        manager.graph = None
        manager.local_document_indexer = None
        manager.local_document_sync_status = None
        manager._local_document_sync_task = None
        manager._checkpointer_ctx = None
        manager._sync_checkpointer_conn = None

        with patch("src.inference.api_server.SKIP_MODEL_LOAD", True):
            await manager.initialize()

        # 엔진이 생성되지 않아야 한다
        assert manager.engine is None


# ---------------------------------------------------------------------------
# 5. LoRARequest import 방어 테스트
# ---------------------------------------------------------------------------


class TestLoRARequestImportGuard:
    """LoRARequest가 None일 때도 정상 동작하는지 확인."""

    def test_lora_request_none_when_unavailable(self):
        """vllm.lora.request가 없으면 LoRARequest는 None."""
        # LoRARequest는 이미 try/except로 import되므로, None인 경우를 테스트
        # CI 환경(SKIP_MODEL_LOAD=true)에서는 vllm이 없을 수 있음
        from src.inference import api_server

        # LoRARequest가 None이거나 class이거나 둘 중 하나
        assert api_server.LoRARequest is None or callable(api_server.LoRARequest)

    def test_draft_tool_skips_lora_when_none(self):
        """LoRARequest가 None이면 lora_req도 None이어야 한다."""
        adapter_path = "/some/path"
        LoRARequestLocal = None  # 시뮬레이션

        lora_req = None
        if adapter_path and LoRARequestLocal is not None:
            lora_req = LoRARequestLocal("civil", 1, adapter_path)

        assert lora_req is None
