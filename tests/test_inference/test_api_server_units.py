"""
api_server.py 유틸리티 함수 단위 테스트.

vLLMEngineManager의 내부 메서드와 모듈 레벨 유틸리티 함수를 검증한다.
GPU/모델 의존성 없이 실행 가능.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 무거운 의존성 mock 등록
# ---------------------------------------------------------------------------
_vllm_mock = MagicMock()
_vllm_mock.AsyncLLM = MagicMock()
_vllm_mock.SamplingParams = MagicMock()
sys.modules.setdefault("vllm", _vllm_mock)
sys.modules.setdefault("vllm.engine", _vllm_mock)
sys.modules.setdefault("vllm.engine.arg_utils", _vllm_mock)
sys.modules.setdefault("vllm.engine.async_llm_engine", _vllm_mock)
sys.modules.setdefault("vllm.sampling_params", _vllm_mock)

_st_mock = MagicMock()
sys.modules.setdefault("sentence_transformers", _st_mock)

sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("transformers.modeling_rope_utils", MagicMock())
sys.modules.setdefault("transformers.utils", MagicMock())
sys.modules.setdefault("transformers.utils.generic", MagicMock())

if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

# ---------------------------------------------------------------------------
# api_server import
# ---------------------------------------------------------------------------

from src.inference.api_server import (
    _rate_limit,
    get_feature_flags,
    manager,
    verify_api_key,
    vLLMEngineManager,
)

# ---------------------------------------------------------------------------
# _escape_special_tokens 테스트
# ---------------------------------------------------------------------------


class TestEscapeSpecialTokens:
    def setup_method(self):
        self.mgr = vLLMEngineManager()

    def test_escapes_user_token(self):
        """[|user|] 토큰을 이스케이프한다."""
        result = self.mgr._escape_special_tokens("hello [|user|] world")
        assert "[|user|]" not in result
        assert "\\[|user|\\]" in result

    def test_escapes_assistant_token(self):
        """[|assistant|] 토큰을 이스케이프한다."""
        result = self.mgr._escape_special_tokens("[|assistant|]")
        assert "\\[|assistant|\\]" in result

    def test_escapes_system_token(self):
        """[|system|] 토큰을 이스케이프한다."""
        result = self.mgr._escape_special_tokens("[|system|]")
        assert "\\[|system|\\]" in result

    def test_escapes_thought_tags(self):
        """<thought> 태그를 이스케이프한다."""
        result = self.mgr._escape_special_tokens("<thought>내부 추론</thought>")
        assert "<thought>" not in result
        assert "\\<thought\\>" in result

    def test_no_special_tokens(self):
        """특수 토큰이 없으면 원본을 반환한다."""
        text = "일반 텍스트입니다."
        result = self.mgr._escape_special_tokens(text)
        assert result == text

    def test_empty_string(self):
        """빈 문자열 처리."""
        assert self.mgr._escape_special_tokens("") == ""

    def test_multiple_tokens(self):
        """여러 특수 토큰을 모두 이스케이프한다."""
        text = "[|system|]시스템[|endofturn|][|user|]사용자[|endofturn|]"
        result = self.mgr._escape_special_tokens(text)
        assert "[|system|]" not in result
        assert "[|user|]" not in result
        assert "[|endofturn|]" not in result


# ---------------------------------------------------------------------------
# _strip_thought_blocks 테스트
# ---------------------------------------------------------------------------


class TestStripThoughtBlocks:
    def test_removes_thought_block(self):
        """<thought>...</thought> 블록을 제거한다."""
        text = "<thought>내부 추론 과정</thought>최종 답변입니다."
        result = vLLMEngineManager._strip_thought_blocks(text)
        assert result == "최종 답변입니다."

    def test_removes_multiline_thought_block(self):
        """여러 줄 thought 블록을 제거한다."""
        text = "<thought>\n분석 중...\n결론 도출\n</thought>\n답변: 복구 예정"
        result = vLLMEngineManager._strip_thought_blocks(text)
        assert "분석 중" not in result
        assert "답변: 복구 예정" in result

    def test_no_thought_block(self):
        """thought 블록이 없으면 원본을 반환한다."""
        text = "일반 답변입니다."
        result = vLLMEngineManager._strip_thought_blocks(text)
        assert result == text

    def test_empty_string(self):
        """빈 문자열 처리."""
        assert vLLMEngineManager._strip_thought_blocks("") == ""

    def test_multiple_thought_blocks(self):
        """여러 thought 블록을 모두 제거한다."""
        text = "<thought>1차 분석</thought>결과1 <thought>2차 분석</thought>결과2"
        result = vLLMEngineManager._strip_thought_blocks(text)
        assert "1차 분석" not in result
        assert "2차 분석" not in result
        assert "결과1" in result
        assert "결과2" in result


# ---------------------------------------------------------------------------
# _extract_query 테스트
# ---------------------------------------------------------------------------


class TestExtractQuery:
    def setup_method(self):
        self.mgr = vLLMEngineManager()

    def test_extract_with_complaint_label(self):
        """민원 내용: 라벨이 있으면 해당 내용을 추출한다."""
        prompt = "[|user|]민원 내용: 도로가 파손되었습니다.[|endofturn|]"
        result = self.mgr._extract_query(prompt)
        assert result == "도로가 파손되었습니다."

    def test_extract_without_complaint_label(self):
        """민원 내용: 라벨이 없으면 user 블록 전체를 반환한다."""
        prompt = "[|user|]도로 파손 신고[|endofturn|]"
        result = self.mgr._extract_query(prompt)
        assert result == "도로 파손 신고"

    def test_extract_no_user_tag(self):
        """[|user|] 태그가 없으면 원본 프롬프트를 반환한다."""
        prompt = "일반 텍스트"
        result = self.mgr._extract_query(prompt)
        assert result == prompt


# ---------------------------------------------------------------------------
# _rate_limit 테스트
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_returns_decorator(self):
        """rate_limit은 데코레이터를 반환한다."""
        decorator = _rate_limit("60/minute")
        assert callable(decorator)

    def test_noop_decorator_preserves_function(self):
        """slowapi 미설치 환경에서 noop 데코레이터가 함수를 보존한다."""
        # _RATE_LIMIT_AVAILABLE이 False일 때의 동작 테스트
        with patch("src.inference.api_server._RATE_LIMIT_AVAILABLE", False):
            decorator = _rate_limit("10/minute")

            def dummy():
                return "ok"

            result = decorator(dummy)
            assert result is dummy


# ---------------------------------------------------------------------------
# verify_api_key 테스트
# ---------------------------------------------------------------------------


class TestVerifyApiKey:
    @pytest.mark.asyncio
    async def test_skips_when_no_api_key_set(self):
        """API_KEY 미설정 + ALLOW_NO_AUTH 시 인증을 건너뛴다."""
        with (
            patch("src.inference.api_server._API_KEY", None),
            patch("src.inference.api_server._ALLOW_NO_AUTH", True),
        ):
            result = await verify_api_key(api_key="anything")
            assert result is None

    @pytest.mark.asyncio
    async def test_valid_api_key(self):
        """유효한 API 키는 통과한다."""
        with patch("src.inference.api_server._API_KEY", "secret"):
            result = await verify_api_key(api_key="secret")
            assert result is None

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises(self):
        """유효하지 않은 API 키는 401을 반환한다."""
        from fastapi import HTTPException

        with patch("src.inference.api_server._API_KEY", "secret"):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key="wrong-key")
            assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# get_feature_flags 테스트
# ---------------------------------------------------------------------------


class TestGetFeatureFlags:
    def test_returns_default_flags(self):
        """헤더가 없으면 기본 플래그를 반환한다."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = None

        flags = get_feature_flags(mock_request)
        assert flags.model_version == manager.feature_flags.model_version

    def test_overrides_from_header(self):
        """X-Feature-Flag 헤더로 플래그를 오버라이드한다."""
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "MODEL_VERSION=v1_lora"

        flags = get_feature_flags(mock_request)
        assert flags.model_version == "v1_lora"


class TestImportWithoutSqlalchemy:
    def test_api_server_imports_without_sqlalchemy_when_local_docs_disabled(self):
        repo_root = Path(__file__).resolve().parents[2]
        script = textwrap.dedent("""
            import builtins
            import sys
            from unittest.mock import MagicMock

            original_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "sqlalchemy" or name.startswith("sqlalchemy."):
                    raise ModuleNotFoundError("No module named 'sqlalchemy'")
                return original_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import

            _vllm_mock = MagicMock()
            _vllm_mock.AsyncLLM = MagicMock()
            _vllm_mock.SamplingParams = MagicMock()
            sys.modules.setdefault("vllm", _vllm_mock)
            sys.modules.setdefault("vllm.engine", _vllm_mock)
            sys.modules.setdefault("vllm.engine.arg_utils", _vllm_mock)
            sys.modules.setdefault("vllm.engine.async_llm_engine", _vllm_mock)
            sys.modules.setdefault("vllm.sampling_params", _vllm_mock)
            sys.modules.setdefault("sentence_transformers", MagicMock())
            sys.modules.setdefault("transformers", MagicMock())
            sys.modules.setdefault("transformers.modeling_rope_utils", MagicMock())
            sys.modules.setdefault("transformers.utils", MagicMock())
            sys.modules.setdefault("transformers.utils.generic", MagicMock())
            sys.modules.setdefault("faiss", MagicMock())
            sys.modules.setdefault("torch", MagicMock())

            import src.inference.api_server  # noqa: F401
            """)
        env = {
            **os.environ,
            "SKIP_MODEL_LOAD": "true",
            "LOCAL_DOCS_ROOT": "",
            "PYTHONPATH": str(repo_root),
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
