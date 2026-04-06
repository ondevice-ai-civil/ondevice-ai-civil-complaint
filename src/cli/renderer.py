"""Result rendering for GovOn CLI.

Uses `rich` when available; falls back to plain print() otherwise.
"""

from __future__ import annotations

import contextlib
import sys
from contextlib import contextmanager
from typing import Generator, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.status import Status
    from rich.text import Text

    _console = Console()
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _console = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Node status message mapping
# ---------------------------------------------------------------------------

NODE_STATUS_MESSAGES: dict[str, str] = {
    "session_load": "세션 로드 중…",
    "planner": "계획 수립 중…",
    "approval_wait": "승인 대기 중…",
    "tool_execute": "도구 실행 중…",
    "synthesis": "답변 생성 중…",
    "persist": "저장 중…",
}


def get_node_message(node_name: str) -> str:
    """Return a human-readable status message for a given node name."""
    return NODE_STATUS_MESSAGES.get(node_name, f"{node_name} 처리 중…")


# ---------------------------------------------------------------------------
# Spinner context manager
# ---------------------------------------------------------------------------


class StreamingStatusDisplay:
    """Context manager that shows a spinner and updates the message per node.

    Wraps rich.status.Status when rich is available; falls back to plain print().
    """

    def __init__(self, initial_message: str = "처리 중…") -> None:
        self._initial_message = initial_message
        self._status: Optional["Status"] = None  # type: ignore[name-defined]

    def __enter__(self) -> "StreamingStatusDisplay":
        if _RICH_AVAILABLE:
            self._status = _console.status(self._initial_message, spinner="dots")
            self._status.__enter__()
        else:
            print(f"→ {self._initial_message}", flush=True)
        return self

    def update(self, message: str) -> None:
        """Update the displayed status message."""
        if _RICH_AVAILABLE and self._status is not None:
            self._status.update(message)
        else:
            print(f"→ {message}", flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if _RICH_AVAILABLE and self._status is not None:
            self._status.__exit__(exc_type, exc_val, exc_tb)
            self._status = None


def render_evidence_section(evidence_items: list) -> str:
    """EvidenceItem dict 리스트를 출처 섹션 텍스트로 변환한다.

    source_type별로 그룹화하여 표시한다:
      [로컬 문서] — rag 출처 (file_path, page, score 포함)
      [외부 API]  — api 출처 (URL 포함)
      [LLM 생성]  — llm_generated 출처

    Parameters
    ----------
    evidence_items : list
        EvidenceItem.to_dict() 형태의 dict 리스트.

    Returns
    -------
    str
        출처 섹션 텍스트. items가 없으면 빈 문자열.
    """
    if not evidence_items:
        return ""

    # source_type별 그룹화
    rag_items = [i for i in evidence_items if i.get("source_type") == "rag"]
    api_items = [i for i in evidence_items if i.get("source_type") == "api"]
    llm_items = [i for i in evidence_items if i.get("source_type") == "llm_generated"]

    lines: list[str] = ["── 참조 근거 ──"]
    idx = 1

    if rag_items:
        lines.append("[로컬 문서]")
        for item in rag_items:
            title = item.get("title") or item.get("link_or_path", "")
            page = item.get("page")
            score = item.get("score", 0.0)
            page_str = f" (p.{page})" if page is not None else ""
            score_str = f" [{score:.2f}]" if score else ""
            lines.append(f"  {idx}. {title}{page_str}{score_str}")
            idx += 1

    if api_items:
        lines.append("[외부 API]")
        for item in api_items:
            title = item.get("title", "")
            link = item.get("link_or_path", "")
            link_str = f" — {link}" if link else ""
            lines.append(f"  {idx}. {title}{link_str}")
            idx += 1

    if llm_items:
        lines.append("[LLM 생성]")
        for item in llm_items:
            title = item.get("title", "")
            excerpt = item.get("excerpt", "")[:80]
            lines.append(f"  {idx}. {title}: {excerpt}" if title else f"  {idx}. {excerpt}")
            idx += 1

    return "\n".join(lines) if len(lines) > 1 else ""


def render_result(result: dict) -> None:
    """Render the final agent response to the terminal.

    Expected keys (at least one required):
      - result["text"] or result["response"]: main answer text
      - result["evidence_items"]: EvidenceItem dict 리스트 (structured, 우선)
      - result["citations"] or result["sources"]: list of source strings (fallback)
    """
    text_body: str = result.get("text") or result.get("response") or ""
    evidence_items: list = result.get("evidence_items") or []
    citations: list = result.get("citations") or result.get("sources") or []

    if _RICH_AVAILABLE:
        content = Text(text_body)
        if evidence_items:
            evidence_text = render_evidence_section(evidence_items)
            if evidence_text:
                content.append(f"\n\n{evidence_text}\n", style="dim")
        elif citations:
            content.append("\n\n출처\n", style="bold")
            for idx, src in enumerate(citations, 1):
                content.append(f"  {idx}. {src}\n", style="dim")
        _console.print(Panel(content, title="[bold green]GovOn[/bold green]", border_style="green"))
    else:
        print("\n── GovOn ──────────────────────────────────")
        print(text_body)
        if evidence_items:
            evidence_text = render_evidence_section(evidence_items)
            if evidence_text:
                print(f"\n{evidence_text}")
        elif citations:
            print("\n출처")
            for idx, src in enumerate(citations, 1):
                print(f"  {idx}. {src}")
        print("───────────────────────────────────────────\n")


def render_status(message: str) -> None:
    """Render a transient status / progress message."""
    if _RICH_AVAILABLE:
        _console.print(f"[dim]→ {message}[/dim]")
    else:
        print(f"→ {message}")


def render_error(message: str) -> None:
    """Render an error message in red."""
    if _RICH_AVAILABLE:
        _console.print(f"[bold red]오류:[/bold red] {message}")
    else:
        print(f"오류: {message}")


def render_session_info(session_id: str) -> None:
    """Render session resume hint at shell exit."""
    hint = f"[session: {session_id}]  govon --session {session_id} 로 재개 가능"
    if _RICH_AVAILABLE:
        _console.print(f"[dim]{hint}[/dim]")
    else:
        print(hint)
