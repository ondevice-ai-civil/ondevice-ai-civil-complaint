"""Slash command parser and handler for GovOn CLI."""

COMMANDS: dict[str, str] = {
    "/help": "사용 가능한 명령과 도움말을 표시합니다.",
    "/clear": "터미널 화면을 초기화합니다.",
    "/exit": "셸을 종료합니다.",
}

_HELP_TEXT = """GovOn CLI 사용법
────────────────────────────────────────
  govon                       인터랙티브 REPL 모드
  govon "질문"                단발 실행 모드
  govon --session <id>        기존 세션 재개
  govon --session <id> "질문" 기존 세션에서 단발 실행
  govon --status              daemon 상태 확인
  govon --stop                daemon 중지

슬래시 명령
────────────────────────────────────────"""

for _cmd, _desc in COMMANDS.items():
    _HELP_TEXT += f"\n  {_cmd:<10} {_desc}"

_HELP_TEXT += "\n────────────────────────────────────────\n업무 요청은 자연어로 직접 입력하세요."


def is_command(text: str) -> bool:
    """Return True if text is a slash command."""
    return text.strip().startswith("/")


def handle_command(text: str) -> str | None:
    """Execute a slash command and return a result string, or None.

    Raises SystemExit for /exit.
    """
    cmd = text.strip().split()[0].lower()

    if cmd == "/help":
        return _HELP_TEXT

    if cmd == "/clear":
        print("\033[2J\033[H", end="", flush=True)
        return None

    if cmd == "/exit":
        raise SystemExit(0)

    return f"알 수 없는 명령입니다: {cmd}\n/help를 입력하세요."
