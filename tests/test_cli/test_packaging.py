"""패키징 및 엔트리포인트 검증 테스트.

Issue #405: 설치 번들 정비 및 daemon bootstrap 스크립트
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestEntryPointImportable(unittest.TestCase):
    """govon 엔트리포인트 임포트 가능 여부를 검증한다."""

    def test_govon_entry_point_importable(self) -> None:
        """from src.cli.shell import main이 정상적으로 임포트되어야 한다."""
        from src.cli.shell import main  # noqa: F401

        self.assertTrue(callable(main), "main은 callable이어야 합니다.")


class TestRuntimeUrlEnvSkipsDaemon(unittest.TestCase):
    """GOVON_RUNTIME_URL 환경변수 설정 시 daemon 기동을 건너뛰는지 검증한다."""

    def test_runtime_url_env_skips_daemon(self) -> None:
        """GOVON_RUNTIME_URL이 설정된 경우 DaemonManager.ensure_running()이 호출되지 않아야 한다."""
        # 기존 sys.argv 보존
        original_argv = sys.argv[:]

        try:
            sys.argv = ["govon", "--status"]

            with patch.dict(os.environ, {"GOVON_RUNTIME_URL": "http://remote.example.com:8000"}):
                with patch("src.cli.shell.DaemonManager") as mock_dm_cls:
                    mock_dm = MagicMock()
                    mock_dm_cls.return_value = mock_dm

                    with self.assertRaises(SystemExit) as cm:
                        from src.cli.shell import main

                        main()

                    # --status 플래그가 있으므로 sys.exit(0)으로 종료되어야 한다
                    self.assertEqual(cm.exception.code, 0)

                    # GOVON_RUNTIME_URL이 설정되어 있으면 DaemonManager는 인스턴스화되지 않는다
                    mock_dm_cls.assert_not_called()
        finally:
            sys.argv = original_argv

    def test_runtime_url_base_url_used(self) -> None:
        """GOVON_RUNTIME_URL이 GovOnClient의 base_url로 사용되어야 한다."""
        original_argv = sys.argv[:]
        remote_url = "http://remote.example.com:9000"

        try:
            sys.argv = ["govon", "test query"]

            captured_base_url: list[str] = []

            def fake_client_init(self_inner: object, base_url: str) -> None:  # type: ignore[override]
                captured_base_url.append(base_url)

            with patch.dict(os.environ, {"GOVON_RUNTIME_URL": remote_url}):
                with patch("src.cli.shell.GovOnClient") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client.stream.side_effect = ConnectionError("테스트용 연결 오류")
                    mock_client.run.return_value = {"text": "ok", "status": "completed"}
                    mock_client_cls.return_value = mock_client

                    from src.cli.shell import main

                    try:
                        main()
                    except SystemExit:
                        pass
                    except Exception:
                        pass

                    # GovOnClient가 remote_url.rstrip("/")로 초기화되어야 한다
                    mock_client_cls.assert_called_once()
                    actual_url = mock_client_cls.call_args[0][0]
                    self.assertEqual(actual_url, remote_url.rstrip("/"))
        finally:
            sys.argv = original_argv


class TestCliExtrasImportable(unittest.TestCase):
    """cli extra에 포함된 패키지들이 임포트 가능한지 검증한다.

    패키지가 설치되어 있지 않으면 테스트를 건너뜁니다.
    'pip install govon[cli]' 또는 'pip install -e .[cli]' 후에 모두 통과해야 합니다.
    """

    def test_httpx_importable(self) -> None:
        """httpx가 임포트 가능해야 한다."""
        try:
            import httpx  # noqa: F401
        except ImportError:
            self.skipTest("httpx가 설치되어 있지 않습니다. 'pip install govon[cli]'를 실행하세요.")
        self.assertTrue(hasattr(httpx, "Client"), "httpx.Client가 존재해야 합니다.")

    def test_rich_importable(self) -> None:
        """rich가 임포트 가능해야 한다."""
        try:
            import rich  # noqa: F401
        except ImportError:
            self.skipTest("rich가 설치되어 있지 않습니다. 'pip install govon[cli]'를 실행하세요.")
        self.assertIsNotNone(rich)

    def test_prompt_toolkit_importable(self) -> None:
        """prompt_toolkit이 임포트 가능해야 한다."""
        try:
            import prompt_toolkit  # noqa: F401
        except ImportError:
            self.skipTest(
                "prompt_toolkit이 설치되어 있지 않습니다. 'pip install govon[cli]'를 실행하세요."
            )
        self.assertIsNotNone(prompt_toolkit)


if __name__ == "__main__":
    unittest.main()
