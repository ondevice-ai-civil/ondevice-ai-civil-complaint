"""HTTP/SSE 클라이언트 레이어.

httpx 우선, urllib fallback. 요청/응답 자동 로깅.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .config import API_KEY, BASE_URL, TIMEOUT

try:
    import httpx

    _HTTP_BACKEND = "httpx"

    def _build_headers() -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    async def http_get(path: str, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=_build_headers())
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"_raw": resp.text[:200]}

    async def http_post(path: str, body: dict, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body, headers=_build_headers())
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"_raw": resp.text[:200]}

    async def http_post_sse(
        path: str, body: dict, timeout: float = TIMEOUT
    ) -> tuple[int, list[dict]]:
        """SSE 스트리밍 POST. 청크를 수집하여 파싱된 이벤트 목록을 반환한다."""
        url = BASE_URL + path
        h = _build_headers()
        h["Accept"] = "text/event-stream"
        events: list[dict] = []
        status_code = 0
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=body, headers=h) as resp:
                status_code = resp.status_code
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        events.append({"_raw": payload})
        return status_code, events

    async def http_get_raw(url: str, timeout: float = 10) -> tuple[int, str]:
        """Raw GET for external connectivity checks."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code, resp.text[:200]

except ImportError:
    import urllib.error
    import urllib.request

    _HTTP_BACKEND = "urllib"

    def _build_headers() -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    async def http_get(path: str, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        req = urllib.request.Request(url, headers=_build_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {}

    async def http_post(path: str, body: dict, timeout: float = TIMEOUT) -> tuple[int, dict]:
        url = BASE_URL + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=_build_headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, {}

    async def http_post_sse(
        path: str, body: dict, timeout: float = TIMEOUT
    ) -> tuple[int, list[dict]]:
        url = BASE_URL + path
        data = json.dumps(body).encode()
        h = _build_headers()
        h["Accept"] = "text/event-stream"
        req = urllib.request.Request(url, data=data, headers=h, method="POST")
        events: list[dict] = []
        status_code = 0
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status_code = r.status
                for raw_line in r:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        events.append({"_raw": payload})
        except urllib.error.HTTPError as e:
            status_code = e.code
        return status_code, events

    async def http_get_raw(url: str, timeout: float = 10) -> tuple[int, str]:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode()[:200]
        except urllib.error.HTTPError as e:
            return e.code, ""
        except Exception:
            return 0, ""


def get_http_backend() -> str:
    return _HTTP_BACKEND
