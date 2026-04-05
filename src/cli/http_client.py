"""GovOn 로컬 daemon API HTTP 클라이언트.

Issue #144: CLI-daemon/LangGraph runtime 연동 및 session resume.
Issue #140: CLI 승인 UI 및 최소 명령 체계 (백엔드 부분).

로컬 daemon(uvicorn)의 REST API를 래핑하는 클라이언트.
run / approve / cancel 등 핵심 엔드포인트에 접근한다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, Iterator, Optional

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
        logger.debug(f"[http_client] approve: thread_id={thread_id} approved={approved}")
        return self._post_params(
            "/v2/agent/approve",
            params={"thread_id": thread_id, "approved": str(approved).lower()},
            timeout=self._DEFAULT_TIMEOUT,
        )

    def stream(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """POST /v2/agent/stream — SSE 스트리밍으로 노드별 이벤트를 수신한다.

        Parameters
        ----------
        query : str
            사용자 입력 쿼리.
        session_id : str | None
            기존 세션을 이어받을 경우 session ID.

        Yields
        ------
        dict
            파싱된 SSE 이벤트 dict. 최소 ``node``와 ``status`` 키를 포함한다.

        Raises
        ------
        ConnectionError
            daemon에 연결할 수 없을 때.
        httpx.HTTPStatusError
            HTTP 오류 응답 시.
        """
        body: Dict[str, Any] = {"query": query}
        if session_id is not None:
            body["session_id"] = session_id

        url = f"{self._base_url}/v2/agent/stream"
        logger.debug(f"[http_client] stream: session_id={session_id} query_len={len(query)}")

        try:
            timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                            if not data_str:
                                continue
                            try:
                                event = json.loads(data_str)
                                yield event
                            except json.JSONDecodeError:
                                logger.warning(f"[http_client] SSE JSON 파싱 실패: {data_str!r}")
                                continue
        except httpx.ConnectError as exc:
            raise ConnectionError(f"daemon이 실행 중이 아닙니다. ({self._base_url})") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"[http_client] HTTP {exc.response.status_code}: {url}")
            raise

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
        logger.debug(f"[http_client] cancel: thread_id={thread_id}")
        return self._post_params(
            "/v2/agent/cancel",
            params={"thread_id": thread_id},
            timeout=self._DEFAULT_TIMEOUT,
        )

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
            raise ConnectionError(f"daemon이 실행 중이 아닙니다. ({self._base_url})") from exc
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
            raise ConnectionError(f"daemon이 실행 중이 아닙니다. ({self._base_url})") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"[http_client] HTTP {exc.response.status_code}: {url}")
            raise

    def _post_params(
        self,
        path: str,
        *,
        params: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """쿼리 파라미터를 사용하는 POST 요청 헬퍼.

        `/v2/agent/approve`, `/v2/agent/cancel` 등 FastAPI 엔드포인트가
        쿼리 파라미터를 기대할 때 사용한다.
        """
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(f"daemon이 실행 중이 아닙니다. ({self._base_url})") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"[http_client] HTTP {exc.response.status_code}: {url}")
            raise
