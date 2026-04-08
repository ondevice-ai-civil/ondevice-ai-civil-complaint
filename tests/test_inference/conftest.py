"""
테스트 공통 fixture.
"""

import sys
from unittest.mock import MagicMock

# faiss 모듈이 설치되지 않은 환경에서도 테스트가 동작하도록 mock 등록
# setdefault를 사용하여 실제 faiss를 덮어쓰지 않음
_faiss_module = sys.modules.get("faiss")
_faiss_is_real = _faiss_module is not None and not isinstance(_faiss_module, MagicMock)
if not _faiss_is_real:
    sys.modules.setdefault("faiss", MagicMock())

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_renderer_narrow_warning_state():
    """Ensure renderer module state does not leak between tests."""
    from src.cli import renderer

    renderer._reset_narrow_warning()
    yield
    renderer._reset_narrow_warning()
