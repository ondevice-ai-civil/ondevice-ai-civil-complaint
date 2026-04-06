"""API 보안 인증 통합 테스트.

Issue #492: test_api_integration.py 완성 및 tests/test_inference/ 이동.
API Key 인증 성공/실패 케이스를 mock 기반으로 검증한다.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 무거운 의존성 mock 등록 (vLLM, sentence-transformers 등)
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
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("transformers.modeling_rope_utils", MagicMock())
sys.modules.setdefault("transformers.utils", MagicMock())
sys.modules.setdefault("transformers.utils.generic", MagicMock())
sys.modules.setdefault("torch", sys.modules.get("torch", MagicMock()))

with patch("src.inference.vllm_stabilizer.apply_transformers_patch"):
    import src.inference.api_server as _api_server_module
    from src.inference.api_server import app, verify_api_key

from fastapi import HTTPException
from fastapi.testclient import TestClient

_API_KEY_ATTR = "_API_KEY"  # api_server 내부 API key 변수 이름


@pytest.fixture()
def api_key():
    return "test-secret-key-for-integration"


@pytest.fixture()
def client(api_key, monkeypatch):
    monkeypatch.setenv("API_KEY", api_key)
    monkeypatch.setattr(_api_server_module, _API_KEY_ATTR, api_key)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 인증 실패 케이스 (기존 테스트)
# ---------------------------------------------------------------------------


def test_security_authentication_missing_key(client):
    """API Key 헤더 없이 요청하면 401이 반환돼야 한다."""
    response = client.post(
        "/v2/agent/run",
        json={"query": "테스트 민원입니다."},
    )
    assert response.status_code == 401


def test_security_authentication_wrong_key(client):
    """잘못된 API Key를 전달하면 401이 반환돼야 한다."""
    response = client.post(
        "/v2/agent/run",
        headers={"X-API-Key": "wrong-key"},
        json={"query": "테스트 민원입니다."},
    )
    assert response.status_code == 401


def test_security_authentication_empty_key(client):
    """빈 API Key를 전달하면 401이 반환돼야 한다."""
    response = client.post(
        "/v2/agent/run",
        headers={"X-API-Key": ""},
        json={"query": "테스트 민원입니다."},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 인증 성공 케이스
# ---------------------------------------------------------------------------


def test_security_authentication_success(client, api_key):
    """올바른 API Key를 전달하면 인증이 성공해야 한다.

    graph가 초기화되지 않은 상태(503)이더라도 401은 아니어야 한다.
    """
    response = client.post(
        "/v2/agent/run",
        headers={"X-API-Key": api_key},
        json={"query": "테스트 민원입니다."},
    )
    # 인증 자체는 성공 → 401이 아님
    assert response.status_code != 401, (
        f"인증 성공 케이스에서 인증 오류({response.status_code})가 반환되었습니다."
    )
    # graph 미초기화 시 503 또는 정상 응답 기대
    assert response.status_code in (200, 503)


def test_health_endpoint_no_auth_required(client):
    """헬스체크 엔드포인트는 인증 없이 접근 가능해야 한다."""
    response = client.get("/health")
    assert response.status_code == 200


def test_security_authentication_success_health_with_key(client, api_key):
    """헬스체크는 API Key가 있어도 200이어야 한다."""
    response = client.get("/health", headers={"X-API-Key": api_key})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# verify_api_key 함수 직접 테스트
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_api_key_with_correct_key(api_key, monkeypatch):
    """올바른 키를 전달하면 예외가 발생하지 않아야 한다."""
    monkeypatch.setattr(_api_server_module, _API_KEY_ATTR, api_key)
    result = await verify_api_key(api_key)
    assert result is None  # no exception raised


@pytest.mark.asyncio
async def test_verify_api_key_with_wrong_key(api_key, monkeypatch):
    """잘못된 키를 전달하면 HTTPException(401)이 발생해야 한다."""
    monkeypatch.setattr(_api_server_module, _API_KEY_ATTR, api_key)
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key("totally-wrong-key")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_api_key_no_key_configured(monkeypatch):
    """API_KEY가 설정되지 않으면 모든 키가 허용돼야 한다 (개발 모드)."""
    monkeypatch.setattr(_api_server_module, _API_KEY_ATTR, None)
    # 예외 없이 통과해야 한다
    result = await verify_api_key("any-key")
    assert result is None

