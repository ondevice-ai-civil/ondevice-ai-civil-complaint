"""Claude Code-style spinner display for GovOn CLI.

Displays an animated spinner with fun status verbs and elapsed time
while waiting for agent responses. Output goes to stderr so piping
stdout remains clean.

Display format:
    {spinner_char} {verb}... ({elapsed} | {tokens} tokens)
"""

from __future__ import annotations

import random
import sys
import time
from threading import Event, Lock, Thread

SPINNER_CHARS: tuple[str, ...] = ("\u00b7", "\u273b", "\u273d", "\u2736", "\u2733", "\u2722")

STATUS_VERBS: tuple[str, ...] = (
    # 사자성어
    "온고지신 (溫故知新)",
    "유비무환 (有備無患)",
    "일석이조 (一石二鳥)",
    "대기만성 (大器晩成)",
    "자강불식 (自強不息)",
    "형설지공 (螢雪之功)",
    "우공이산 (愚公移山)",
    "마부위침 (磨斧爲針)",
    "견인불발 (堅忍不拔)",
    "금상첨화 (錦上添花)",
    "선견지명 (先見之明)",
    "타산지석 (他山之石)",
    "화룡점정 (畫龍點睛)",
    "사필귀정 (事必歸正)",
    "전화위복 (轉禍爲福)",
    "고진감래 (苦盡甘來)",
    "불철주야 (不撤晝夜)",
    "절차탁마 (切磋琢磨)",
    "심사숙고 (深思熟考)",
    "천재일우 (千載一遇)",
    "이심전심 (以心傳心)",
    "백문불여일견 (百聞不如一見)",
    "각골난망 (刻骨難忘)",
    "결초보은 (結草報恩)",
    "개과천선 (改過遷善)",
    # 속담
    "천 리 길도 한 걸음부터",
    "구슬이 서 말이라도 꿰어야 보배",
    "돌다리도 두들겨 보고 건너라",
    "뜻이 있는 곳에 길이 있다",
    "공든 탑이 무너지랴",
    "티끌 모아 태산",
    "하늘은 스스로 돕는 자를 돕는다",
    "실패는 성공의 어머니",
    "세 살 버릇 여든까지 간다",
    "빈 수레가 요란하다",
    "등잔 밑이 어둡다",
    "호랑이도 제 말 하면 온다",
    "원숭이도 나무에서 떨어진다",
    "소 잃고 외양간 고친다",
    "가는 말이 고와야 오는 말이 곱다",
    # 재미있는 상식
    "꿀벌은 한 숟갈의 꿀을 위해 평생 난다",
    "문어의 심장은 세 개다",
    "바나나는 식물학적으로 장과(berry)이다",
    "지구에서 가장 오래된 나무는 5000살이 넘는다",
    "인간의 뇌는 60%가 지방이다",
    "해파리는 뇌가 없다",
    "낙타의 혹에는 물이 아니라 지방이 있다",
)


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into human-readable string like '1m 23s'."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes = total // 60
    secs = total % 60
    return f"{minutes}m {secs:02d}s"


def _format_tokens(count: int) -> str:
    """Format token count with k-suffix for readability."""
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


class SpinnerDisplay:
    """Animated spinner that shows status verbs and elapsed time on stderr.

    Uses a background thread for smooth animation independent of main logic.
    Compatible with Rich's Live display but implemented standalone for
    minimal dependency and stderr-only output.
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._tokens: int = 0
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._verb: str = ""
        self._char_index: int = 0
        self._running: bool = False

    def start(self) -> None:
        """Begin the spinner animation on stderr."""
        if self._running:
            return
        self._start_time = time.monotonic()
        self._tokens = 0
        self._verb = random.choice(STATUS_VERBS)  # noqa: S311
        self._char_index = 0
        self._stop_event.clear()
        self._running = True
        self._thread = Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the spinner and clear the line on stderr."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Clear the spinner line
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def update_tokens(self, count: int) -> None:
        """Update the displayed token count."""
        with self._lock:
            self._tokens = count

    def _animate(self) -> None:
        """Background animation loop cycling spinner chars and verbs."""
        frame = 0
        while not self._stop_event.is_set():
            self._char_index = frame % len(SPINNER_CHARS)
            # Change verb every ~90 frames (~3 seconds at 30fps)
            if frame > 0 and frame % 90 == 0:
                self._verb = random.choice(STATUS_VERBS)  # noqa: S311

            char = SPINNER_CHARS[self._char_index]
            elapsed = time.monotonic() - self._start_time
            elapsed_str = _format_elapsed(elapsed)

            with self._lock:
                tokens = self._tokens

            if tokens > 0:
                token_str = _format_tokens(tokens)
                line = (
                    f"\r{char} {self._verb}\u2026 ({elapsed_str} \u00b7 \u2193 {token_str} tokens)"
                )
            else:
                line = f"\r{char} {self._verb}\u2026 ({elapsed_str})"

            sys.stderr.write(f"\033[K{line}")
            sys.stderr.flush()

            frame += 1
            self._stop_event.wait(timeout=0.033)  # ~30fps

    def __enter__(self) -> SpinnerDisplay:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self.stop()
