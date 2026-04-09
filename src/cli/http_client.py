"""GovOn local daemon API HTTP client.

Issue #144: CLI-daemon/LangGraph runtime integration and session resume.
Issue #140: CLI approval UI and minimal command structure (backend part).

Client wrapping the REST API of the local daemon (uvicorn).
Accesses core endpoints: run / approve / cancel.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Generator, Iterator, Optional

import httpx
from loguru import logger


class GovOnClient:
    """GovOn local daemon HTTP client.

    Parameters
    ----------
    base_url : str
        daemon base URL (e.g. "http://127.0.0.1:8000").
    """

    _RUN_TIMEOUT = 120.0
    _DEFAULT_TIMEOUT = 30.0
    _COLD_START_TIMEOUT = int(os.getenv("GOVON_COLD_START_TIMEOUT", "600"))
    _COLD_START_INTERVAL = int(os.getenv("GOVON_COLD_START_INTERVAL", "5"))

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """GET /health — check daemon status.

        Returns
        -------
        dict
            Health response returned by the server.

        Raises
        ------
        ConnectionError
            When daemon is unreachable.
        """
        return self._get("/health", timeout=self._DEFAULT_TIMEOUT)

    def wait_for_ready(self) -> bool:
        """Wait until the server is ready (cold start / sleeping handling).

        When a remote server such as HF Space is in sleeping/building/starting
        state, displays status messages instead of errors and waits until ready.

        Returns
        -------
        bool
            True if the server becomes ready, False on timeout.
        """
        url = f"{self._base_url}/health"
        deadline = time.monotonic() + self._COLD_START_TIMEOUT
        last_status = ""
        attempt = 0

        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        if attempt > 0:
                            self._cold_start_print(
                                "\r✦ 서버 준비 완료.                              "
                            )
                            print()
                        return True
                    if resp.status_code == 503:
                        new_status = "⊛ 서버 시작 중… (503 응답 대기)"
                    else:
                        new_status = f"⊛ 서버 응답 대기 중… (HTTP {resp.status_code})"
            except httpx.ConnectError:
                new_status = "⊛ 서버에 연결 중… (sleeping 상태에서 깨어나는 중)"
            except httpx.TimeoutException:
                new_status = "⊛ 서버 응답 대기 중… (빌드 또는 모델 로딩 중)"
            except httpx.RequestError:
                new_status = "⊛ 서버 연결 시도 중…"

            elapsed = attempt * self._COLD_START_INTERVAL
            if new_status != last_status or attempt % 6 == 0:
                self._cold_start_print(f"\r\u23f3 {new_status} ({elapsed}s)")
                last_status = new_status

            attempt += 1
            time.sleep(self._COLD_START_INTERVAL)

        self._cold_start_print("\r✘ 서버 연결 시간 초과.                            ")
        print()
        return False

    @staticmethod
    def _cold_start_print(msg: str) -> None:
        """Overwrite cold start status message on the same line."""
        sys.stdout.write(msg)
        sys.stdout.flush()

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v2/agent/run — agent execution request.

        Parameters
        ----------
        query : str
            User input query.
        session_id : str | None
            Session ID to resume an existing session.

        Returns
        -------
        dict
            Server response (includes thread_id, status, etc.).
        """
        body: Dict[str, Any] = {"query": query}
        if session_id is not None:
            body["session_id"] = session_id

        logger.debug(f"[http_client] run: session_id={session_id} query_len={len(query)}")
        return self._post("/v2/agent/run", body=body, timeout=self._RUN_TIMEOUT)

    def approve(self, thread_id: str, approved: bool) -> Dict[str, Any]:
        """POST /v2/agent/approve — approve or reject.

        Parameters
        ----------
        thread_id : str
            Graph thread ID to approve or reject.
        approved : bool
            True to approve, False to reject.

        Returns
        -------
        dict
            Server response.
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
        """POST /v2/agent/stream — receive per-node events via SSE streaming.

        Parameters
        ----------
        query : str
            User input query.
        session_id : str | None
            Session ID to resume an existing session.

        Yields
        ------
        dict
            Parsed SSE event dict. Contains at least ``node`` and ``status`` keys.

        Raises
        ------
        ConnectionError
            When daemon is unreachable.
        httpx.HTTPStatusError
            On HTTP error response.
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
                                logger.warning(f"[http_client] SSE JSON parse failed: {data_str!r}")
                                continue
        except httpx.ConnectError as exc:
            raise ConnectionError(f"daemon is not running. ({self._base_url})") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"[http_client] HTTP {exc.response.status_code}: {url}")
            raise

    def stream_v3(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_iterations: int = 10,
    ) -> Generator[Dict[str, Any], None, None]:
        """POST /v3/agent/stream — v3 ReAct fine-grained SSE streaming.

        Parameters
        ----------
        query : str
            User input query.
        session_id : str | None
            Session ID to resume an existing session.
        max_iterations : int
            Maximum ReAct loop iterations.

        Yields
        ------
        dict
            Parsed SSE event dict. Use the ``type`` key to distinguish event types.
        """
        body: Dict[str, Any] = {"query": query, "max_iterations": max_iterations}
        if session_id is not None:
            body["session_id"] = session_id

        url = f"{self._base_url}/v3/agent/stream"
        logger.debug(f"[http_client] stream_v3: session_id={session_id} query_len={len(query)}")

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
                                logger.warning(
                                    f"[http_client] v3 SSE JSON parse failed: len={len(data_str)}"
                                )
                                continue
        except httpx.ConnectError as exc:
            raise ConnectionError(f"daemon is not running. ({self._base_url})") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"[http_client] HTTP {exc.response.status_code}: {url}")
            raise

    def run_v3(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """POST /v3/agent/run — v3 ReAct blocking execution.

        Parameters
        ----------
        query : str
            User input query.
        session_id : str | None
            Session ID to resume an existing session.
        max_iterations : int
            Maximum ReAct loop iterations.

        Returns
        -------
        dict
            Server response (includes metadata).
        """
        body: Dict[str, Any] = {"query": query, "max_iterations": max_iterations}
        if session_id is not None:
            body["session_id"] = session_id

        logger.debug(f"[http_client] run_v3: session_id={session_id} query_len={len(query)}")
        return self._post("/v3/agent/run", body=body, timeout=self._RUN_TIMEOUT)

    def cancel(self, thread_id: str) -> Dict[str, Any]:
        """POST /v2/agent/cancel — cancel a running session.

        Parameters
        ----------
        thread_id : str
            Graph thread ID to cancel.

        Returns
        -------
        dict
            Server response.
        """
        logger.debug(f"[http_client] cancel: thread_id={thread_id}")
        return self._post_params(
            "/v2/agent/cancel",
            params={"thread_id": thread_id},
            timeout=self._DEFAULT_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, *, timeout: float) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(f"daemon is not running. ({self._base_url})") from exc
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
            raise ConnectionError(f"daemon is not running. ({self._base_url})") from exc
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
        """POST request helper using query parameters.

        Used when FastAPI endpoints like ``/v2/agent/approve`` and
        ``/v2/agent/cancel`` expect query parameters.
        """
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(f"daemon is not running. ({self._base_url})") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"[http_client] HTTP {exc.response.status_code}: {url}")
            raise
