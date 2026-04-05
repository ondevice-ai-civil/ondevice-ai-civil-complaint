"""GovOn daemon lifecycle 관리.

Issue #144: CLI-daemon/LangGraph runtime 연동 및 session resume.

uvicorn으로 백그라운드에서 GovOn API 서버를 기동하고,
PID 파일로 프로세스 상태를 추적한다.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger


class DaemonManager:
    """GovOn API 서버 daemon lifecycle 관리자.

    PID 파일과 /health 엔드포인트를 결합하여 daemon 상태를 확인하고,
    필요 시 uvicorn으로 백그라운드 기동한다.

    환경변수 ``GOVON_PORT``로 포트를 오버라이드할 수 있다 (기본: 8000).
    """

    GOVON_HOME = Path.home() / ".govon"
    _HEALTH_CHECK_TIMEOUT = 30  # 최대 대기 초
    _HEALTH_CHECK_INTERVAL = 1  # 재시도 간격 (초)

    def __init__(self) -> None:
        self.GOVON_HOME.mkdir(parents=True, exist_ok=True)
        self.port: int = int(os.environ.get("GOVON_PORT", "8000"))
        self.pid_path: Path = self.GOVON_HOME / "daemon.pid"
        self.log_path: Path = self.GOVON_HOME / "daemon.log"

    def get_base_url(self) -> str:
        """daemon base URL을 반환한다."""
        return f"http://127.0.0.1:{self.port}"

    def is_running(self) -> bool:
        """daemon이 실행 중인지 확인한다.

        PID 파일이 존재하고 해당 프로세스가 살아 있으며,
        /health 엔드포인트가 응답할 때 True를 반환한다.
        """
        pid = self._read_pid()
        if pid is None:
            return False

        # PID 프로세스 생존 확인
        if not self._pid_alive(pid):
            logger.debug(f"[daemon] PID {pid} 프로세스가 없음. PID 파일 제거.")
            self._remove_pid()
            return False

        # /health HTTP 확인
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.get_base_url()}/health")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return False

    def start(self) -> bool:
        """uvicorn을 백그라운드로 기동하고 PID를 기록한다.

        Returns
        -------
        bool
            기동 성공 여부 (health check 통과 시 True).
        """
        # 레이스 컨디션 방지: 기동 전 한 번 더 health check
        if self.is_running():
            logger.info("[daemon] 이미 실행 중입니다.")
            return True

        log_file = open(self.log_path, "a")  # noqa: WPS515

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.inference.api_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]

        logger.info(f"[daemon] 기동 명령: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        log_file.close()

        self._write_pid(proc.pid)
        logger.info(f"[daemon] 프로세스 기동 완료. PID={proc.pid}")

        # health check 대기
        return self._wait_until_healthy()

    def stop(self) -> None:
        """daemon을 정상 종료한다 (SIGTERM → timeout 후 SIGKILL)."""
        pid = self._read_pid()
        if pid is None:
            logger.info("[daemon] PID 파일이 없습니다. 실행 중이 아닌 것으로 간주합니다.")
            return

        if not self._pid_alive(pid):
            logger.info(f"[daemon] PID {pid} 프로세스가 없습니다.")
            self._remove_pid()
            return

        logger.info(f"[daemon] SIGTERM 전송: PID={pid}")
        os.kill(pid, signal.SIGTERM)

        # 최대 10초 대기
        for _ in range(10):
            time.sleep(1)
            if not self._pid_alive(pid):
                logger.info(f"[daemon] PID {pid} 정상 종료됨.")
                self._remove_pid()
                return

        logger.warning(f"[daemon] SIGKILL 전송: PID={pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._remove_pid()

    def ensure_running(self) -> str:
        """daemon이 실행 중임을 보장하고 base URL을 반환한다.

        실행 중이 아니면 start()를 호출한다.

        Returns
        -------
        str
            daemon base URL (예: "http://127.0.0.1:8000").

        Raises
        ------
        RuntimeError
            daemon 기동에 실패한 경우.
        """
        if not self.is_running():
            success = self.start()
            if not success:
                raise RuntimeError(
                    "GovOn daemon 기동에 실패했습니다. " f"로그를 확인하세요: {self.log_path}"
                )
        return self.get_base_url()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _read_pid(self) -> Optional[int]:
        """PID 파일에서 PID를 읽는다. 파일이 없으면 None."""
        if not self.pid_path.exists():
            return None
        try:
            first_line = self.pid_path.read_text().strip().splitlines()[0]
            return int(first_line.split()[0])
        except (ValueError, OSError, IndexError):
            return None

    def _write_pid(self, pid: int) -> None:
        """PID와 기동 시각(epoch timestamp)을 파일에 기록한다."""
        self.pid_path.write_text(f"{pid} {int(time.time())}")

    def _remove_pid(self) -> None:
        """PID 파일을 제거한다."""
        try:
            self.pid_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """프로세스가 살아 있는지 확인한다."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # 프로세스가 존재하지만 권한이 없는 경우 → 살아 있음으로 간주
            return True

    def _wait_until_healthy(self) -> bool:
        """health check가 통과할 때까지 최대 30초 대기한다."""
        deadline = time.monotonic() + self._HEALTH_CHECK_TIMEOUT
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=3.0) as client:
                    resp = client.get(f"{self.get_base_url()}/health")
                    if resp.status_code == 200:
                        logger.info("[daemon] health check 통과.")
                        return True
            except (httpx.ConnectError, httpx.TimeoutException, Exception):
                pass
            time.sleep(self._HEALTH_CHECK_INTERVAL)

        logger.error("[daemon] health check timeout (30초).")
        return False
