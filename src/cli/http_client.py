"""GovOn 로컬 daemon API HTTP 클라이언트.

Issue #144: CLI-daemon/LangGraph runtime 연동 및 session resume.
Issue #140: CLI 승인 UI 및 최소 명령 체계 (백엔드 부분).

로컬 daemon(uvicorn)의 REST API를 래핑하는 클라이언트.
run / approve / cancel 등 핵심 엔드포인트에 접근한다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from loguru import logger


class GovOnClient:
    """GovOn 로컬 daemon HTTP 클라이언트.

    Parameters
    ----------
    base_url : str
        daemon base URL (예: "http://127.0.0.1:8000").
    """

    _RUN_TIMEOUT = 120.0
    _DEFAULT_TIMEOUT = 30.0

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """GET /health — daemon 상태를 확인한다.

        Returns
        -------
        dict
            서버가 반환하는 health 응답.

        Raises
        ------
        ConnectionError
            daemon에 연결할 수 없을 때.
        """
        return self._get("/health", timeout=self._DEFAULT_TIMEOUT)

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v2/agent/run — 에이전트 실행 요청.

        Parameters
        ----------
        query : str
            사용자 입력 쿼리.
        session_id : str | None
            기존 세션을 이어받을 경우 session ID.

        Returns
        -------
        dict
            서버 응답 (thread_id, status 등 포함).
        """
        body: Dict[str, Any] = {"query": query}
        if session_id is not None:
            body["session_id"] = session_id

        logger.debug(f"[http_client] run: session_id={session_id} query_len={len(query)}")
        return self._post("/v2/agent/run", body=body, timeout=self._RUN_TIMEOUT)

    def approve(self, thread_id: str, approved: bool) -> Dict[str, Any]:
        """POST /v2/agent/approve — 승인 또는 거절.

        Parameters
        ----------
        thread_id : str
            승인/거절할 graph thread ID.
        approved : bool
            True이면 승인, False이면 거절.

        Returns
        -------
        dict
            서버 응답.
        """
        body = {"thread_id": thread_id, "approved": approved}
        logger.debug(f"[http_client] approve: thread_id={thread_id} approved={approved}")
        return self._post("/v2/agent/approve", body=body, timeout=self._DEFAULT_TIMEOUT)

    def cancel(self, thread_id: str) -> Dict[str, Any]:
        """POST /v2/agent/cancel — 실행 중인 세션 취소.

        Parameters
        ----------
        thread_id : str
            취소할 graph thread ID.

        Returns
        -------
        dict
            서버 응답.
        """
        body = {"thread_id": thread_id}
        logger.debug(f"[http_client] cancel: thread_id={thread_id}")
        return self._post("/v2/agent/cancel", body=body, timeout=self._DEFAULT_TIMEOUT)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _get(self, path: str, *, timeout: float) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"daemon이 실행 중이 아닙니다. ({self._base_url})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"[http_client] HTTP {exc.response.status_code}: {url}")
            raise

    def _post(
        self,
        path: str,
        *,
        body: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=body)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"daemon이 실행 중이 아닙니다. ({self._base_url})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"[http_client] HTTP {exc.response.status_code}: {url}")
            raise
