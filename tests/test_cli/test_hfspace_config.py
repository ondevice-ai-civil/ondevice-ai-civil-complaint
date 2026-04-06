"""HF Spaces 배포 설정 검증 테스트."""

import os
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestHFSpaceConfig(unittest.TestCase):
    def test_dockerfile_hfspace_exists(self):
        assert (PROJECT_ROOT / "Dockerfile.hfspace").exists()

    def test_dockerfile_exposes_7860(self):
        with open(PROJECT_ROOT / "Dockerfile.hfspace") as f:
            content = f.read()
        assert "EXPOSE 7860" in content
        assert "PORT=7860" in content

    def test_dockerfile_has_nonroot_user(self):
        with open(PROJECT_ROOT / "Dockerfile.hfspace") as f:
            content = f.read()
        assert "USER user" in content or "USER 1000" in content

    def test_space_id_in_container_markers(self):
        from src.inference.runtime_config import _CONTAINER_PLATFORM_ENV_MARKERS

        assert "SPACE_ID" in _CONTAINER_PLATFORM_ENV_MARKERS

    def test_env_example_exists(self):
        assert (PROJECT_ROOT / ".env.hfspace.example").exists()

    def test_deploy_script_exists(self):
        deploy_script = PROJECT_ROOT / "scripts" / "deploy-hfspace.sh"
        assert deploy_script.exists()
        # 실행 권한 확인
        assert os.access(deploy_script, os.X_OK)
