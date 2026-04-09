"""GovOn daemon lifecycle management.

Issue #144: CLI-daemon/LangGraph runtime integration and session resume.

Starts the GovOn API server in the background via uvicorn and tracks
the process state via a PID file.

.. note::
   This module is for **local daemon use only**.
   When connecting to a remote server, set the ``GOVON_RUNTIME_URL``
   environment variable and ``shell.py``'s ``main()`` will skip this
   module entirely and connect directly to the specified URL.
   This approach is recommended for Docker, cloud deployments, and CI.
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
    """GovOn API server daemon lifecycle manager.

    Combines PID file tracking with /health endpoint polling to verify
    daemon state, and starts it via uvicorn as a background process when
    needed.

    Override the default port (8000) via the ``GOVON_PORT`` environment
    variable.
    """

    GOVON_HOME = Path.home() / ".govon"
    _HEALTH_CHECK_TIMEOUT = 120  # seconds
    _HEALTH_CHECK_INTERVAL = 1  # seconds between retries

    def __init__(self) -> None:
        self.GOVON_HOME.mkdir(parents=True, exist_ok=True)
        self.port: int = int(os.environ.get("GOVON_PORT", "8000"))
        self.pid_path: Path = self.GOVON_HOME / "daemon.pid"
        self.log_path: Path = self.GOVON_HOME / "daemon.log"

    def get_base_url(self) -> str:
        """Return the daemon base URL."""
        return f"http://127.0.0.1:{self.port}"

    def is_running(self) -> bool:
        """Check whether the daemon is running.

        Returns True when the PID file exists, the process is alive, and
        the /health endpoint responds with HTTP 200.
        """
        pid = self._read_pid()
        if pid is None:
            return False

        # Verify the process is still alive
        if not self._pid_alive(pid):
            logger.debug(f"[daemon] PID {pid} not found; removing PID file.")
            self._remove_pid()
            return False

        # Verify /health HTTP endpoint
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.get_base_url()}/health")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError, OSError):
            return False

    def start(self) -> bool:
        """Start uvicorn in the background and record its PID.

        Returns
        -------
        bool
            True if the daemon started successfully (health check passed).
        """
        # Guard against race conditions: re-check before starting
        if self.is_running():
            logger.info("[daemon] Already running.")
            return True

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

        if self._port_in_use():
            logger.error(f"[daemon] Port {self.port} is already in use.")
            return False

        logger.info(f"[daemon] Starting: {' '.join(cmd)}")

        with open(self.log_path, "a") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )

        self._write_pid(proc.pid)
        logger.info(f"[daemon] Process started. PID={proc.pid}")

        # Wait for the health check to pass
        healthy = self._wait_until_healthy()
        if not healthy:
            logger.error("[daemon] Health check failed; cleaning up.")
            self.stop()
            return False
        return True

    def stop(self) -> None:
        """Gracefully stop the daemon (SIGTERM, then SIGKILL after timeout)."""
        pid = self._read_pid()
        if pid is None:
            logger.info("[daemon] No PID file found; assuming not running.")
            return

        if not self._pid_alive(pid):
            logger.info(f"[daemon] PID {pid} not found.")
            self._remove_pid()
            return

        logger.info(f"[daemon] Sending SIGTERM: PID={pid}")
        os.kill(pid, signal.SIGTERM)

        # Wait up to 10 seconds for graceful shutdown
        for _ in range(10):
            time.sleep(1)
            if not self._pid_alive(pid):
                logger.info(f"[daemon] PID {pid} terminated gracefully.")
                self._remove_pid()
                return

        logger.warning(f"[daemon] Sending SIGKILL: PID={pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._remove_pid()

    def ensure_running(self) -> str:
        """Ensure the daemon is running and return its base URL.

        Calls start() if the daemon is not already running.

        Returns
        -------
        str
            Daemon base URL, e.g. "http://127.0.0.1:8000".

        Raises
        ------
        RuntimeError
            If the daemon fails to start.
        """
        if not self.is_running():
            success = self.start()
            if not success:
                raise RuntimeError(f"GovOn daemon failed to start. Check logs: {self.log_path}")
        return self.get_base_url()

    def _port_in_use(self) -> bool:
        """Return True if the target port is already bound."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_pid(self) -> Optional[int]:
        """Read the PID from the PID file; return None if absent or invalid."""
        if not self.pid_path.exists():
            return None
        try:
            first_line = self.pid_path.read_text().strip().splitlines()[0]
            return int(first_line.split()[0])
        except (ValueError, OSError, IndexError):
            return None

    def _write_pid(self, pid: int) -> None:
        """Write PID and start timestamp (epoch) to the PID file."""
        self.pid_path.write_text(f"{pid} {int(time.time())}")

    def _remove_pid(self) -> None:
        """Remove the PID file, ignoring missing-file errors."""
        try:
            self.pid_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Return True if the process with *pid* is alive."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we lack permissions — treat as alive
            return True

    def _wait_until_healthy(self) -> bool:
        """Poll /health until it returns 200 or the timeout (120 s) expires."""
        deadline = time.monotonic() + self._HEALTH_CHECK_TIMEOUT
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=3.0) as client:
                    resp = client.get(f"{self.get_base_url()}/health")
                    if resp.status_code == 200:
                        logger.info("[daemon] Health check passed.")
                        return True
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError, OSError):
                pass
            time.sleep(self._HEALTH_CHECK_INTERVAL)

        logger.error("[daemon] Health check timed out (120 s).")
        return False
