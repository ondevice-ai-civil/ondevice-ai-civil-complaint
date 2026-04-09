"""govon server 서브커맨드 -- Docker 기반 백엔드 관리.

Usage:
    govon server pull [TAG]   — docker pull ghcr.io/govon-org/govon:<TAG>
    govon server start        — docker compose up -d
    govon server stop         — docker compose down
    govon server status       — 컨테이너 상태 + /health 체크
    govon server logs         — docker compose logs -f
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

try:
    from rich.console import Console
    from rich.table import Table

    _console = Console()
    _RICH = True
except ImportError:  # pragma: no cover
    _console = None  # type: ignore[assignment]
    _RICH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = "ghcr.io/govon-org/govon"
DEFAULT_TAG = "latest"
HEALTH_ENDPOINT = "http://localhost:{port}/health"
DEFAULT_PORT = 8000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_success(msg: str) -> None:
    if _RICH:
        _console.print(f"[bold green]✓[/bold green] {msg}")
    else:
        print(f"[OK] {msg}")


def _print_error(msg: str) -> None:
    if _RICH:
        _console.print(f"[bold red]✗[/bold red] {msg}")
    else:
        print(f"[ERROR] {msg}", file=sys.stderr)


def _print_info(msg: str) -> None:
    if _RICH:
        _console.print(f"[dim]→[/dim] {msg}")
    else:
        print(f"→ {msg}")


def _print_warn(msg: str) -> None:
    if _RICH:
        _console.print(f"[yellow]![/yellow] {msg}")
    else:
        print(f"[WARN] {msg}")


def _detect_docker() -> str | None:
    """Docker 실행 파일 경로를 반환한다. 없으면 None."""
    return shutil.which("docker")


def _detect_compose_command() -> list[str] | None:
    """docker compose (v2) 또는 docker-compose (v1) 명령을 감지한다.

    Returns:
        실행 가능한 compose 명령 리스트, 없으면 None.
    """
    # v2: docker compose
    docker = _detect_docker()
    if docker:
        result = subprocess.run(
            [docker, "compose", "version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return [docker, "compose"]

    # v1: docker-compose
    docker_compose_v1 = shutil.which("docker-compose")
    if docker_compose_v1:
        return [docker_compose_v1]

    return None


def _find_compose_file() -> Path | None:
    """docker-compose.yml 파일을 현재 디렉토리 또는 프로젝트 루트에서 찾는다."""
    cwd = Path.cwd()
    for candidate in [cwd / "docker-compose.yml", cwd / "docker-compose.yaml"]:
        if candidate.is_file():
            return candidate
    return None


def _get_host_port() -> int:
    """환경변수 또는 기본값에서 호스트 포트를 결정한다."""
    return int(os.environ.get("HOST_PORT", str(DEFAULT_PORT)))


def _run_cmd(cmd: list[str], *, stream: bool = False) -> subprocess.CompletedProcess | None:
    """subprocess로 명령을 실행한다. stream=True이면 실시간 출력."""
    _print_info(f"실행: {' '.join(cmd)}")
    try:
        if stream:
            proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
            return None
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        _print_error(f"명령을 찾을 수 없습니다: {cmd[0]}")
        return None
    except KeyboardInterrupt:
        return None


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_pull(tag: str = DEFAULT_TAG) -> int:
    """Docker 이미지를 pull한다."""
    docker = _detect_docker()
    if not docker:
        _print_error("Docker가 설치되어 있지 않습니다. https://docs.docker.com/get-docker/ 에서 설치하세요.")
        return 1

    image = f"{DEFAULT_IMAGE}:{tag}"
    _print_info(f"이미지 다운로드 중: {image}")
    result = _run_cmd([docker, "pull", image], stream=True)
    if result is None:
        _print_success(f"이미지 pull 완료: {image}")
    return 0


def _cmd_start() -> int:
    """docker compose up -d로 백엔드를 시작한다."""
    docker = _detect_docker()
    if not docker:
        _print_error("Docker가 설치되어 있지 않습니다. https://docs.docker.com/get-docker/ 에서 설치하세요.")
        return 1

    compose_cmd = _detect_compose_command()
    if not compose_cmd:
        _print_error("docker compose를 사용할 수 없습니다. Docker Desktop을 업데이트하세요.")
        return 1

    compose_file = _find_compose_file()
    if not compose_file:
        _print_error("docker-compose.yml 파일을 찾을 수 없습니다. 프로젝트 루트에서 실행하세요.")
        return 1

    cmd = [*compose_cmd, "-f", str(compose_file), "up", "-d"]

    # .env 파일이 있으면 자동으로 전달
    env_file = compose_file.parent / ".env"
    if env_file.is_file():
        cmd = [*compose_cmd, "-f", str(compose_file), "--env-file", str(env_file), "up", "-d"]
        _print_info(f".env 파일 감지: {env_file}")

    result = _run_cmd(cmd, stream=True)
    port = _get_host_port()
    _print_success(f"백엔드 시작됨 — http://localhost:{port}")
    return 0


def _cmd_stop() -> int:
    """docker compose down으로 백엔드를 중지한다."""
    docker = _detect_docker()
    if not docker:
        _print_error("Docker가 설치되어 있지 않습니다.")
        return 1

    compose_cmd = _detect_compose_command()
    if not compose_cmd:
        _print_error("docker compose를 사용할 수 없습니다.")
        return 1

    compose_file = _find_compose_file()
    if not compose_file:
        _print_error("docker-compose.yml 파일을 찾을 수 없습니다.")
        return 1

    cmd = [*compose_cmd, "-f", str(compose_file), "down"]
    _run_cmd(cmd, stream=True)
    _print_success("백엔드가 중지되었습니다.")
    return 0


def _cmd_status() -> int:
    """컨테이너 상태와 /health 엔드포인트를 확인한다."""
    docker = _detect_docker()
    if not docker:
        _print_error("Docker가 설치되어 있지 않습니다.")
        return 1

    # 컨테이너 상태 표시
    container_name = os.environ.get("GOVON_CONTAINER_NAME", "govon-backend")
    result = subprocess.run(
        [docker, "ps", "--filter", f"name={container_name}", "--format",
         "{{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"],
        capture_output=True,
        text=True,
    )

    if _RICH:
        table = Table(title="GovOn 컨테이너 상태")
        table.add_column("ID", style="cyan")
        table.add_column("이미지", style="magenta")
        table.add_column("상태", style="green")
        table.add_column("포트", style="yellow")

        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 4:
                    table.add_row(parts[0][:12], parts[1], parts[2], parts[3])
                elif len(parts) >= 3:
                    table.add_row(parts[0][:12], parts[1], parts[2], "-")
            _console.print(table)
        else:
            _print_warn(f"실행 중인 '{container_name}' 컨테이너가 없습니다.")
    else:
        if result.stdout.strip():
            print(f"컨테이너 상태:\n{result.stdout}")
        else:
            _print_warn(f"실행 중인 '{container_name}' 컨테이너가 없습니다.")

    # /health 엔드포인트 체크
    port = _get_host_port()
    health_url = HEALTH_ENDPOINT.format(port=port)
    _print_info(f"헬스체크: {health_url}")

    try:
        resp = httpx.get(health_url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            status_val = data.get("status", "unknown")
            if status_val == "healthy":
                _print_success(f"백엔드 정상 (status={status_val})")
            else:
                _print_warn(f"백엔드 응답: status={status_val}")
        else:
            _print_warn(f"헬스체크 응답: HTTP {resp.status_code}")
    except httpx.ConnectError:
        _print_error(f"백엔드에 연결할 수 없습니다 ({health_url})")
    except httpx.TimeoutException:
        _print_error(f"헬스체크 타임아웃 ({health_url})")
    except Exception as exc:
        _print_error(f"헬스체크 실패: {exc}")

    return 0


def _cmd_logs() -> int:
    """docker compose logs -f로 로그를 스트리밍한다."""
    docker = _detect_docker()
    if not docker:
        _print_error("Docker가 설치되어 있지 않습니다.")
        return 1

    compose_cmd = _detect_compose_command()
    if not compose_cmd:
        _print_error("docker compose를 사용할 수 없습니다.")
        return 1

    compose_file = _find_compose_file()
    if not compose_file:
        _print_error("docker-compose.yml 파일을 찾을 수 없습니다.")
        return 1

    cmd = [*compose_cmd, "-f", str(compose_file), "logs", "-f", "--tail", "100"]
    _print_info("로그 스트리밍 시작 (Ctrl+C로 중단)")
    _run_cmd(cmd, stream=True)
    return 0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {
    "pull": "Docker 이미지 다운로드",
    "start": "백엔드 시작 (docker compose up -d)",
    "stop": "백엔드 중지 (docker compose down)",
    "status": "컨테이너 상태 및 헬스체크",
    "logs": "백엔드 로그 스트리밍",
}


def _print_server_help() -> None:
    """server 서브커맨드 도움말을 출력한다."""
    if _RICH:
        table = Table(title="govon server 명령", show_header=True)
        table.add_column("명령", style="cyan", no_wrap=True)
        table.add_column("설명")
        for cmd, desc in _SUBCOMMANDS.items():
            table.add_row(f"govon server {cmd}", desc)
        table.add_row("govon server pull [TAG]", "특정 태그의 이미지 다운로드")
        _console.print(table)
    else:
        print("govon server 명령:")
        print("─" * 50)
        for cmd, desc in _SUBCOMMANDS.items():
            print(f"  govon server {cmd:<10} {desc}")
        print(f"  govon server pull [TAG]  특정 태그의 이미지 다운로드")
        print("─" * 50)


def handle_server(argv: list[str]) -> int:
    """server 서브커맨드를 처리한다.

    Args:
        argv: 'server' 이후의 인자 리스트. 예: ['start'], ['pull', 'v1.0.6']

    Returns:
        종료 코드 (0=성공).
    """
    if not argv:
        _print_server_help()
        return 0

    action = argv[0].lower()

    if action == "pull":
        tag = argv[1] if len(argv) > 1 else DEFAULT_TAG
        return _cmd_pull(tag)
    elif action == "start":
        return _cmd_start()
    elif action == "stop":
        return _cmd_stop()
    elif action == "status":
        return _cmd_status()
    elif action == "logs":
        return _cmd_logs()
    elif action in ("--help", "-h", "help"):
        _print_server_help()
        return 0
    else:
        _print_error(f"알 수 없는 server 명령: {action}")
        _print_server_help()
        return 1
