"""govon server 서브커맨드 단위 테스트."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cli.server import (
    DEFAULT_IMAGE,
    DEFAULT_TAG,
    _detect_compose_command,
    _detect_docker,
    handle_server,
)


class TestDetectDocker(unittest.TestCase):
    """Docker 감지 로직 테스트."""

    @patch("src.cli.server.shutil.which", return_value="/usr/bin/docker")
    def test_docker_found(self, mock_which: MagicMock) -> None:
        result = _detect_docker()
        self.assertEqual(result, "/usr/bin/docker")

    @patch("src.cli.server.shutil.which", return_value=None)
    def test_docker_not_found(self, mock_which: MagicMock) -> None:
        result = _detect_docker()
        self.assertIsNone(result)


class TestDetectComposeCommand(unittest.TestCase):
    """docker compose v1/v2 감지 테스트."""

    @patch("src.cli.server.subprocess.run")
    @patch("src.cli.server._detect_docker", return_value="/usr/bin/docker")
    def test_compose_v2(self, mock_docker: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        result = _detect_compose_command()
        self.assertEqual(result, ["/usr/bin/docker", "compose"])

    @patch("src.cli.server.shutil.which")
    @patch("src.cli.server.subprocess.run")
    @patch("src.cli.server._detect_docker", return_value="/usr/bin/docker")
    def test_compose_v1_fallback(
        self, mock_docker: MagicMock, mock_run: MagicMock, mock_which: MagicMock
    ) -> None:
        # v2 실패
        mock_run.return_value = MagicMock(returncode=1)
        # v1 발견
        mock_which.return_value = "/usr/local/bin/docker-compose"
        result = _detect_compose_command()
        self.assertEqual(result, ["/usr/local/bin/docker-compose"])

    @patch("src.cli.server.shutil.which", return_value=None)
    @patch("src.cli.server.subprocess.run")
    @patch("src.cli.server._detect_docker", return_value=None)
    def test_no_compose(
        self, mock_docker: MagicMock, mock_run: MagicMock, mock_which: MagicMock
    ) -> None:
        result = _detect_compose_command()
        self.assertIsNone(result)


class TestHandleServer(unittest.TestCase):
    """handle_server 디스패처 테스트."""

    def test_no_args_shows_help(self) -> None:
        """인자 없이 호출하면 도움말 출력 후 0을 반환해야 한다."""
        code = handle_server([])
        self.assertEqual(code, 0)

    def test_help_flag(self) -> None:
        code = handle_server(["--help"])
        self.assertEqual(code, 0)

    def test_unknown_command(self) -> None:
        code = handle_server(["unknown_cmd"])
        self.assertEqual(code, 1)

    @patch("src.cli.server._detect_docker", return_value=None)
    def test_pull_no_docker(self, mock_docker: MagicMock) -> None:
        """Docker가 없으면 pull이 1을 반환해야 한다."""
        code = handle_server(["pull"])
        self.assertEqual(code, 1)

    @patch("src.cli.server._run_cmd", return_value=None)
    @patch("src.cli.server._detect_docker", return_value="/usr/bin/docker")
    def test_pull_default_tag(self, mock_docker: MagicMock, mock_run: MagicMock) -> None:
        code = handle_server(["pull"])
        self.assertEqual(code, 0)
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        self.assertIn(f"{DEFAULT_IMAGE}:{DEFAULT_TAG}", cmd_args)

    @patch("src.cli.server._run_cmd", return_value=None)
    @patch("src.cli.server._detect_docker", return_value="/usr/bin/docker")
    def test_pull_custom_tag(self, mock_docker: MagicMock, mock_run: MagicMock) -> None:
        code = handle_server(["pull", "v1.0.6"])
        self.assertEqual(code, 0)
        cmd_args = mock_run.call_args[0][0]
        self.assertIn(f"{DEFAULT_IMAGE}:v1.0.6", cmd_args)

    @patch("src.cli.server._detect_docker", return_value=None)
    def test_start_no_docker(self, mock_docker: MagicMock) -> None:
        code = handle_server(["start"])
        self.assertEqual(code, 1)

    @patch("src.cli.server._detect_docker", return_value=None)
    def test_stop_no_docker(self, mock_docker: MagicMock) -> None:
        code = handle_server(["stop"])
        self.assertEqual(code, 1)

    @patch("src.cli.server._detect_docker", return_value=None)
    def test_status_no_docker(self, mock_docker: MagicMock) -> None:
        code = handle_server(["status"])
        self.assertEqual(code, 1)

    @patch("src.cli.server._detect_docker", return_value=None)
    def test_logs_no_docker(self, mock_docker: MagicMock) -> None:
        code = handle_server(["logs"])
        self.assertEqual(code, 1)


class TestShellServerDispatch(unittest.TestCase):
    """shell.py main()에서 server 서브커맨드 분기 테스트."""

    @patch("src.cli.server.handle_server", return_value=0)
    def test_server_subcommand_dispatch(self, mock_handle: MagicMock) -> None:
        """sys.argv에 'server start'가 있으면 handle_server가 호출되어야 한다."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["govon", "server", "start"]
            with self.assertRaises(SystemExit) as cm:
                from src.cli.shell import main

                main()
            self.assertEqual(cm.exception.code, 0)
            mock_handle.assert_called_once_with(["start"])
        finally:
            sys.argv = original_argv

    @patch("src.cli.server.handle_server", return_value=0)
    def test_server_pull_with_tag(self, mock_handle: MagicMock) -> None:
        """'server pull v1.0.6'이 올바르게 전달되어야 한다."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["govon", "server", "pull", "v1.0.6"]
            with self.assertRaises(SystemExit) as cm:
                from src.cli.shell import main

                main()
            self.assertEqual(cm.exception.code, 0)
            mock_handle.assert_called_once_with(["pull", "v1.0.6"])
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
