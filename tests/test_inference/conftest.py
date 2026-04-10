"""
테스트 공통 fixture.
"""

import sys
import types
from unittest.mock import MagicMock

# database.py 모듈 레벨의 PostgreSQL engine 생성을 우회
_mock_database = types.ModuleType("src.inference.db.database")
_mock_database.engine = MagicMock()
_mock_database.SessionLocal = MagicMock()
_mock_database.get_db = MagicMock()
sys.modules["src.inference.db.database"] = _mock_database

# ---------------------------------------------------------------------------
# vllm / sentence_transformers — E2E 테스트에서 공통 사용
# ---------------------------------------------------------------------------
_vllm_mock = MagicMock()
_vllm_mock.AsyncLLM = MagicMock()
_vllm_mock.SamplingParams = MagicMock()
sys.modules.setdefault("vllm", _vllm_mock)
sys.modules.setdefault("vllm.engine", _vllm_mock)
sys.modules.setdefault("vllm.engine.arg_utils", _vllm_mock)
sys.modules.setdefault("vllm.engine.async_llm_engine", _vllm_mock)
sys.modules.setdefault("vllm.sampling_params", _vllm_mock)
sys.modules.setdefault("sentence_transformers", MagicMock())

# faiss 모듈이 설치되지 않은 환경에서도 DB 테스트가 동작하도록 mock 등록
# 이미 실제 faiss가 로드된 경우에는 mock하지 않는다
# setdefault를 사용하여 실제 faiss를 덮어쓰지 않음 (직접 대입 대신)
_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, MagicMock)
if not _faiss_is_real:
    sys.modules.setdefault("faiss", MagicMock())

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Renderer fixture removed — src.cli.renderer was deleted in the npm TUI migration.
