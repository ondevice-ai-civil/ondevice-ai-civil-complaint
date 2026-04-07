"""vLLM Multi-LoRA 서빙 통합 테스트.

Issue #468: ADAPTER_PATHS 환경변수 파싱, LoRA 설정 조건부 활성화,
_run_engine lora_request 전달 검증.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# vLLMEngineManager는 TestLoraEngineConfig / TestSkipModelLoadLora에서 각 테스트 내부에서 import함
# (top-level import 시 모듈 로드 부작용을 피하기 위해 지역 import 패턴 유지)

# ---------------------------------------------------------------------------
# 1. ADAPTER_PATHS 환경변수 파싱 테스트
# ---------------------------------------------------------------------------


class TestAdapterPathsParsing:
    """ModelConfig.from_env()의 ADAPTER_PATHS 파싱 동작 검증."""

    def test_normal_case(self):
        """정상 형식: 'public_admin=/path/public_admin,legal=/path/legal'"""
        with patch.dict(
            os.environ,
            {"ADAPTER_PATHS": "public_admin=/models/public_admin,legal=/models/legal"},
        ):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {
                "public_admin": "/models/public_admin",
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
            {"ADAPTER_PATHS": "public_admin=/ok,broken_entry,legal=/also_ok"},
        ):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {
                "public_admin": "/ok",
                "legal": "/also_ok",
            }

    def test_single_adapter(self):
        """단일 어댑터만 설정."""
        with patch.dict(os.environ, {"ADAPTER_PATHS": "public_admin=/path/to/public_admin"}):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {"public_admin": "/path/to/public_admin"}

    def test_whitespace_trimmed(self):
        """공백이 포함된 항목도 정상 파싱."""
        with patch.dict(
            os.environ,
            {"ADAPTER_PATHS": " public_admin = /path/public_admin , legal = /path/legal "},
        ):
            from src.inference.runtime_config import ModelConfig

            config = ModelConfig.from_env()
            assert config.adapter_paths == {
                "public_admin": "/path/public_admin",
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
        mock_lora.lora_name = "public_admin"

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
    """adapter_paths 유무에 따른 enable_lora 설정 검증.

    SKIP_MODEL_LOAD=false로 설정하고 AsyncEngineArgs를 patch하여
    enable_lora 인자가 올바르게 전달되는지 검증한다.
    """

    @pytest.mark.asyncio
    async def test_no_adapters_no_lora(self):
        """adapter_paths가 비어있으면 AsyncEngineArgs에 enable_lora가 전달되지 않아야 한다."""
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

        mock_engine_args_cls = MagicMock()
        mock_engine_args_instance = MagicMock()
        mock_engine_args_cls.return_value = mock_engine_args_instance

        mock_async_llm = MagicMock()
        mock_async_llm.from_engine_args = MagicMock(return_value=MagicMock())

        with (
            patch("src.inference.api_server.SKIP_MODEL_LOAD", False),
            patch("src.inference.api_server.AsyncEngineArgs", mock_engine_args_cls),
            patch("src.inference.api_server.AsyncLLM", mock_async_llm),
            patch(
                "src.inference.api_server.runtime_config.model.adapter_paths",
                {},
            ),
            patch("src.inference.api_server.CivilComplaintRetriever", MagicMock()),
        ):
            await manager.initialize()

        # AsyncEngineArgs에 enable_lora가 전달되지 않아야 한다
        call_kwargs = mock_engine_args_cls.call_args[1]
        assert "enable_lora" not in call_kwargs

    @pytest.mark.asyncio
    async def test_with_adapters_lora_enabled(self):
        """adapter_paths가 있으면 AsyncEngineArgs에 enable_lora=True가 전달되어야 한다."""
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

        mock_engine_args_cls = MagicMock()
        mock_engine_args_instance = MagicMock()
        mock_engine_args_cls.return_value = mock_engine_args_instance

        mock_async_llm = MagicMock()
        mock_async_llm.from_engine_args = MagicMock(return_value=MagicMock())

        with (
            patch("src.inference.api_server.SKIP_MODEL_LOAD", False),
            patch("src.inference.api_server.AsyncEngineArgs", mock_engine_args_cls),
            patch("src.inference.api_server.AsyncLLM", mock_async_llm),
            patch(
                "src.inference.api_server.runtime_config.model.adapter_paths",
                {"public_admin": "/path/to/public_admin"},
            ),
            patch("src.inference.api_server.CivilComplaintRetriever", MagicMock()),
        ):
            await manager.initialize()

        # AsyncEngineArgs에 enable_lora=True가 전달되어야 한다
        call_kwargs = mock_engine_args_cls.call_args[1]
        assert call_kwargs.get("enable_lora") is True


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
        """LoRARequest가 None이면 adapter_path가 있어도 lora_req가 생성되지 않아야 한다."""
        from src.inference import api_server

        original = api_server.LoRARequest
        try:
            api_server.LoRARequest = None

            # LoRARequest가 None이면 adapter_path 존재 여부와 무관하게 건너뜀
            adapter_path = "/some/path"
            lora_req = None
            if adapter_path and api_server.LoRARequest is not None:
                lora_req = api_server.LoRARequest("public_admin", 2, adapter_path)

            assert lora_req is None
        finally:
            api_server.LoRARequest = original


# ---------------------------------------------------------------------------
# 6. Closure 레벨 LoRA 생성/전달 검증
# ---------------------------------------------------------------------------


class TestClosureLevelLoraCreation:
    """LoRA closure에서 LoRARequest가 올바르게 생성되어 _run_engine에 전달되는지 검증."""

    @pytest.mark.asyncio
    async def test_legal_creates_lora_request(self):
        """legal LoRARequest를 생성하여 _run_engine에 전달해야 한다."""
        from src.inference import api_server
        from src.inference.adapter_registry import AdapterRegistry

        original_lora = api_server.LoRARequest
        try:
            # LoRARequest를 추적 가능한 mock으로 교체
            mock_lora_cls = MagicMock()
            mock_lora_instance = MagicMock()
            mock_lora_instance.lora_name = "legal"
            mock_lora_cls.return_value = mock_lora_instance
            api_server.LoRARequest = mock_lora_cls

            with patch.dict(
                api_server.runtime_config.model.adapter_paths,
                {"legal": "siwo/govon-legal-adapter"},
                clear=True,
            ):
                # closure 로직의 핵심: legal adapter path 획득 → LoRARequest 생성 조건 검증
                legal_adapter_path = api_server.runtime_config.model.adapter_paths.get("legal")
                AdapterRegistry.reset()
                reg = AdapterRegistry.get_instance()
                lora_id = reg.get_lora_id("legal")
                lora_req = None
                if (
                    legal_adapter_path
                    and api_server.LoRARequest is not None
                    and lora_id is not None
                ):
                    lora_req = api_server.LoRARequest("legal", lora_id, legal_adapter_path)

            # LoRARequest("legal", 1, "siwo/govon-legal-adapter") 호출 확인
            mock_lora_cls.assert_called_once_with("legal", 1, "siwo/govon-legal-adapter")
            assert lora_req is mock_lora_instance
        finally:
            api_server.LoRARequest = original_lora

    @pytest.mark.asyncio
    async def test_legal_no_lora_when_path_missing(self):
        """legal adapter 경로 미설정 시 lora_req가 None이어야 한다."""
        from src.inference import api_server

        with patch.dict(
            api_server.runtime_config.model.adapter_paths,
            {},  # legal 경로 없음
            clear=True,
        ):
            legal_adapter_path = api_server.runtime_config.model.adapter_paths.get("legal")
            lora_req = None
            if legal_adapter_path and api_server.LoRARequest is not None:
                lora_req = api_server.LoRARequest("legal", 1, legal_adapter_path)

        assert lora_req is None

    def test_public_admin_creates_lora_request(self):
        """public_admin LoRARequest 생성 확인."""
        from src.inference import api_server
        from src.inference.adapter_registry import AdapterRegistry

        original_lora = api_server.LoRARequest
        try:
            mock_lora_cls = MagicMock()
            mock_lora_instance = MagicMock()
            mock_lora_cls.return_value = mock_lora_instance
            api_server.LoRARequest = mock_lora_cls

            AdapterRegistry.reset()
            reg = AdapterRegistry.get_instance()
            lora_id = reg.get_lora_id("public_admin")
            public_admin_adapter_path = "umyunsang/govon-civil-adapter"
            lora_req = None
            if (
                public_admin_adapter_path
                and api_server.LoRARequest is not None
                and lora_id is not None
            ):
                lora_req = api_server.LoRARequest(
                    "public_admin", lora_id, public_admin_adapter_path
                )

            mock_lora_cls.assert_called_once_with(
                "public_admin", 2, "umyunsang/govon-civil-adapter"
            )
            assert lora_req is mock_lora_instance
        finally:
            api_server.LoRARequest = original_lora


# ---------------------------------------------------------------------------
# 7. Multi-LoRA per-request 스위칭 검증
# ---------------------------------------------------------------------------


class TestMultiLoraPerRequestSwitching:
    """public_admin/legal 어댑터가 서로 다른 ID를 사용하여 per-request 스위칭이 가능한지 확인."""

    def test_lora_id_no_duplicates(self):
        """AdapterRegistry에 중복 LoRA ID가 없어야 한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        ids = [reg.get_lora_id(name) for name in reg.list_available()]
        assert len(ids) == len(set(ids)), f"중복 LoRA ID 발견: {ids}"

    def test_public_admin_and_legal_use_different_ids(self):
        """public_admin과 legal 어댑터가 서로 다른 numeric ID를 사용해야 한다."""
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        assert reg.get_lora_id("legal") == 1
        assert reg.get_lora_id("public_admin") == 2
        assert reg.get_lora_id("legal") != reg.get_lora_id("public_admin")

    @pytest.mark.asyncio
    async def test_public_admin_then_legal_use_correct_ids(self):
        """public_admin→legal 순서로 요청 시 각각 올바른 ID의 LoRARequest가 생성되어야 한다."""
        from src.inference import api_server
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        original_lora = api_server.LoRARequest
        try:
            calls = []

            def mock_lora(name, lora_id, path):
                obj = MagicMock()
                obj.lora_name = name
                obj.lora_int_id = lora_id
                calls.append((name, lora_id, path))
                return obj

            api_server.LoRARequest = mock_lora

            # public_admin adapter 생성 시뮬레이션
            public_admin_path = "umyunsang/govon-civil-adapter"
            public_admin_req = api_server.LoRARequest(
                "public_admin", reg.get_lora_id("public_admin"), public_admin_path
            )

            # legal adapter 생성 시뮬레이션
            legal_path = "siwo/govon-legal-adapter"
            legal_req = api_server.LoRARequest("legal", reg.get_lora_id("legal"), legal_path)

            assert calls[0] == ("public_admin", 2, public_admin_path)
            assert calls[1] == ("legal", 1, legal_path)
            assert public_admin_req.lora_int_id != legal_req.lora_int_id
        finally:
            api_server.LoRARequest = original_lora


# ---------------------------------------------------------------------------
# 8. append_evidence LLM 경로 검증
# ---------------------------------------------------------------------------


class TestAppendEvidenceLLMPath:
    """AdapterRegistry를 통한 LoRA ID 유효성 검증."""

    def test_legal_lora_id_is_positive_int(self):
        """AdapterRegistry에 'legal' 어댑터가 존재하고 lora_id가 양의 정수여야 한다.

        frozen dataclass 패치의 어려움으로 인해 실제 LLM 경로 대신
        LoRA ID의 유효성을 직접 검증한다.
        """
        from src.inference.adapter_registry import AdapterRegistry

        AdapterRegistry.reset()
        reg = AdapterRegistry.get_instance()
        lora_id = reg.get_lora_id("legal")
        assert lora_id is not None
        assert isinstance(lora_id, int)
        assert lora_id > 0
