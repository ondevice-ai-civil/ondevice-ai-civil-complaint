"""API 보안 인증 통합 테스트 — issue #492.

인증이 필요한 v2/v3 엔드포인트와 인증이 불필요한 /health를 검증한다.

테스트 환경:
- conftest.py가 vllm/sentence_transformers/database/faiss를 미리 mock 처리하므로
  이 파일에서 중복 mock 불필요.
- _API_KEY / _ALLOW_NO_AUTH는 모듈 레벨 변수이므로 patch()로 주입한다.
- manager.graph / graph_v3 를 None으로 유지하여 인증 레이어만 격리 검증한다.
  (인증 실패 → 401, 인증 통과 → 503 또는 그 이상의 처리)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.inference.api_server import app

_VALID_KEY = "govon-test-secret"
_AGENT_PAYLOAD = {"query": "테스트 민원입니다."}

# v2 인증 대상 엔드포인트 목록
_V2_ENDPOINTS = [
    ("POST", "/v2/agent/run"),
    ("POST", "/v2/agent/stream"),
    ("POST", "/v2/agent/approve"),
    ("POST", "/v2/agent/cancel"),
]

# v3 인증 대상 엔드포인트 목록
_V3_ENDPOINTS = [
    ("POST", "/v3/agent/run"),
    ("POST", "/v3/agent/stream"),
]


@pytest.fixture
def client():
    """FastAPI TestClient (auth 상태는 각 테스트에서 직접 patch)."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def auth_client():
    """_API_KEY가 설정된 TestClient — 인증 활성화 상태."""
    with patch("src.inference.api_server._API_KEY", _VALID_KEY):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Group 1: /health — 인증 없이 접근 가능
# ---------------------------------------------------------------------------


class TestHealthNoAuth:
    """GET /health 엔드포인트는 API 키 없이도 접근 가능해야 한다."""

    def test_health_returns_200_without_key(self, client):
        """/health는 X-API-Key 없이 200을 반환해야 한다."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_200_with_api_key_configured(self, auth_client):
        """/health는 API_KEY가 설정된 환경에서도 키 없이 200을 반환해야 한다."""
        resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_health_response_contains_required_fields(self, client):
        """/health 응답에 status, agents_loaded, feature_flags 필드가 있어야 한다."""
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "agents_loaded" in data
        assert "feature_flags" in data

    def test_health_returns_profile_and_session_store(self, client):
        """/health 응답에 profile, session_store 필드가 포함되어야 한다."""
        resp = client.get("/health")
        data = resp.json()
        assert "profile" in data
        assert "session_store" in data


# ---------------------------------------------------------------------------
# Group 2: v2 엔드포인트 인증 검증
# ---------------------------------------------------------------------------


class TestV2Auth:
    """POST /v2/agent/* 엔드포인트의 API 키 인증 동작을 검증한다."""

    @pytest.mark.parametrize("method,path", _V2_ENDPOINTS)
    def test_valid_key_is_not_rejected(self, auth_client, method, path):
        """유효한 X-API-Key로 요청하면 401이 아닌 응답을 받아야 한다."""
        resp = auth_client.request(
            method, path, json=_AGENT_PAYLOAD, headers={"X-API-Key": _VALID_KEY}
        )
        assert resp.status_code != 401, f"{path}: 유효한 키로 401 발생 — {resp.json()}"

    @pytest.mark.parametrize("method,path", _V2_ENDPOINTS)
    def test_missing_key_returns_401(self, auth_client, method, path):
        """X-API-Key 헤더가 없으면 401을 반환해야 한다."""
        resp = auth_client.request(method, path, json=_AGENT_PAYLOAD)
        assert resp.status_code == 401, f"{path}: 키 누락 시 401 기대 — {resp.status_code}"

    @pytest.mark.parametrize("method,path", _V2_ENDPOINTS)
    def test_wrong_key_returns_401(self, auth_client, method, path):
        """잘못된 X-API-Key로 요청하면 401을 반환해야 한다."""
        resp = auth_client.request(
            method, path, json=_AGENT_PAYLOAD, headers={"X-API-Key": "invalid-key"}
        )
        assert resp.status_code == 401, f"{path}: 잘못된 키로 401 기대 — {resp.status_code}"

    @pytest.mark.parametrize("method,path", _V2_ENDPOINTS)
    def test_empty_key_returns_401(self, auth_client, method, path):
        """빈 X-API-Key 헤더로 요청하면 401을 반환해야 한다."""
        resp = auth_client.request(method, path, json=_AGENT_PAYLOAD, headers={"X-API-Key": ""})
        assert resp.status_code == 401, f"{path}: 빈 키로 401 기대 — {resp.status_code}"


# ---------------------------------------------------------------------------
# Group 3: v3 엔드포인트 인증 검증
# ---------------------------------------------------------------------------


class TestV3Auth:
    """POST /v3/agent/* 엔드포인트의 API 키 인증 동작을 검증한다."""

    @pytest.mark.parametrize("method,path", _V3_ENDPOINTS)
    def test_valid_key_is_not_rejected(self, auth_client, method, path):
        """유효한 X-API-Key로 요청하면 401이 아닌 응답을 받아야 한다."""
        resp = auth_client.request(
            method, path, json=_AGENT_PAYLOAD, headers={"X-API-Key": _VALID_KEY}
        )
        assert resp.status_code != 401, f"{path}: 유효한 키로 401 발생 — {resp.json()}"

    @pytest.mark.parametrize("method,path", _V3_ENDPOINTS)
    def test_missing_key_returns_401(self, auth_client, method, path):
        """X-API-Key 헤더가 없으면 401을 반환해야 한다."""
        resp = auth_client.request(method, path, json=_AGENT_PAYLOAD)
        assert resp.status_code == 401, f"{path}: 키 누락 시 401 기대 — {resp.status_code}"

    @pytest.mark.parametrize("method,path", _V3_ENDPOINTS)
    def test_wrong_key_returns_401(self, auth_client, method, path):
        """잘못된 X-API-Key로 요청하면 401을 반환해야 한다."""
        resp = auth_client.request(
            method, path, json=_AGENT_PAYLOAD, headers={"X-API-Key": "wrong-key"}
        )
        assert resp.status_code == 401, f"{path}: 잘못된 키로 401 기대 — {resp.status_code}"


# ---------------------------------------------------------------------------
# Group 4: ALLOW_NO_AUTH 동작 검증
# ---------------------------------------------------------------------------


class TestAllowNoAuth:
    """ALLOW_NO_AUTH 환경 변수에 따른 인증 우회 동작을 검증한다."""

    def test_no_api_key_configured_without_allow_no_auth_returns_401(self, client):
        """API_KEY 미설정 + ALLOW_NO_AUTH=false → 401을 반환해야 한다."""
        with patch("src.inference.api_server._API_KEY", None):
            with patch("src.inference.api_server._ALLOW_NO_AUTH", False):
                resp = client.post("/v2/agent/run", json=_AGENT_PAYLOAD)
        assert resp.status_code == 401

    def test_allow_no_auth_true_bypasses_auth(self, client):
        """API_KEY 미설정 + ALLOW_NO_AUTH=true → 401이 아닌 응답을 받아야 한다."""
        with patch("src.inference.api_server._API_KEY", None):
            with patch("src.inference.api_server._ALLOW_NO_AUTH", True):
                resp = client.post("/v2/agent/run", json=_AGENT_PAYLOAD)
        assert resp.status_code != 401

    def test_allow_no_auth_applies_to_v3(self, client):
        """ALLOW_NO_AUTH=true는 v3 엔드포인트에도 동일하게 적용된다."""
        with patch("src.inference.api_server._API_KEY", None):
            with patch("src.inference.api_server._ALLOW_NO_AUTH", True):
                resp = client.post("/v3/agent/run", json=_AGENT_PAYLOAD)
        assert resp.status_code != 401

    @pytest.mark.parametrize("path", ["/v2/agent/stream", "/v3/agent/stream"])
    def test_allow_no_auth_applies_to_stream_endpoints(self, client, path):
        """ALLOW_NO_AUTH=true는 stream 엔드포인트에도 동일하게 적용된다."""
        with patch("src.inference.api_server._API_KEY", None):
            with patch("src.inference.api_server._ALLOW_NO_AUTH", True):
                resp = client.post(path, json=_AGENT_PAYLOAD)
        assert resp.status_code != 401
