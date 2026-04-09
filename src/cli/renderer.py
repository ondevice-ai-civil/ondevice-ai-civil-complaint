"""Result rendering for GovOn CLI.

Uses `rich` when available; falls back to plain print() otherwise.
"""

from __future__ import annotations

import sys
from threading import Lock
from typing import Any

from src.cli.terminal import (
    get_narrow_terminal_warning,
    get_terminal_columns,
    is_layout_supported,
)
from src.cli.theme import get_theme

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.status import Status
    from rich.table import Table
    from rich.text import Text

    _console = Console()
    _stderr_console = Console(stderr=True)
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _console = None  # type: ignore[assignment]
    _stderr_console = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False

_HAS_WARNED_NARROW_TERMINAL = False
_NARROW_WARNING_LOCK = Lock()

# ---------------------------------------------------------------------------
# Node status message mapping
# ---------------------------------------------------------------------------

NODE_STATUS_MESSAGES: dict[str, str] = {
    "session_load": "세션 로드 중…",
    "agent": "에이전트 추론 중…",
    "approval_wait": "승인 대기 중…",
    "tools": "도구 실행 중…",
    "persist": "저장 중…",
}

MARKDOWN_CODE_THEME = "monokai"
STRUCTURED_TOOL_ORDER = ("stats_lookup", "keyword_analyzer", "demographics_lookup")
STRUCTURED_TOOL_TITLES = {
    "stats_lookup": "민원 통계",
    "keyword_analyzer": "키워드 분석",
    "demographics_lookup": "인구통계",
}
STRUCTURED_API_TITLES = {
    "doc_count": "채널별 접수 건수",
    "trend": "추이",
    "statistics": "기간별 통계",
    "org_ranking": "기관 순위",
    "region_ranking": "지역 순위",
    "core_keyword": "핵심 키워드",
    "related_word": "연관어",
    "gender": "성별 분포",
    "age": "연령 분포",
    "population": "인구 대비 비율",
}
TABLE_COLUMN_PRIORITY = (
    "keyword",
    "topic",
    "label",
    "term",
    "hits",
    "value",
    "ratio",
    "prebRatio",
    "prevRatio",
    "population",
    "pttn",
    "dfpt",
    "saeol",
)
TABLE_COLUMN_LABELS = {
    "keyword": "키워드",
    "topic": "항목",
    "label": "항목",
    "term": "항목",
    "hits": "건수",
    "value": "값",
    "ratio": "비율",
    "prebRatio": "전일 대비",
    "prevRatio": "전기 대비",
    "population": "인구",
    "pttn": "국민신문고",
    "dfpt": "민원24",
    "saeol": "새올",
    "source_type": "출처",
    "title": "제목",
    "page": "페이지",
    "score": "점수",
    "link_or_path": "경로/링크",
}
TABLE_HIDDEN_KEYS = {"_source_api"}
EVIDENCE_SOURCE_LABELS = {
    "rag": "로컬 문서",
    "api": "외부 API",
    "llm_generated": "LLM 생성",
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
        self._status: Status | None = None  # type: ignore[name-defined]
        self._use_rich = False

    def __enter__(self) -> "StreamingStatusDisplay":
        self._use_rich, _ = _resolve_render_mode()
        if self._use_rich:
            self._status = _stderr_console.status(self._initial_message, spinner="dots")
            self._status.__enter__()
        else:
            print(f"→ {self._initial_message}", file=sys.stderr, flush=True)
        return self

    def update(self, message: str) -> None:
        """Update the displayed status message."""
        if self._use_rich and self._status is not None:
            self._status.update(message)
        else:
            print(f"→ {message}", file=sys.stderr, flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._use_rich and self._status is not None:
            self._status.__exit__(exc_type, exc_val, exc_tb)
            self._status = None


def _warn_narrow_terminal_once(columns: int) -> None:
    """Emit the narrow-terminal fallback warning once per narrow-state entry."""
    global _HAS_WARNED_NARROW_TERMINAL

    with _NARROW_WARNING_LOCK:
        if _HAS_WARNED_NARROW_TERMINAL:
            return
        _HAS_WARNED_NARROW_TERMINAL = True

    print(get_narrow_terminal_warning(columns), file=sys.stderr, flush=True)


def _reset_narrow_warning() -> None:
    """Reset narrow-terminal warning state for tests and wide-terminal recovery."""
    global _HAS_WARNED_NARROW_TERMINAL

    with _NARROW_WARNING_LOCK:
        _HAS_WARNED_NARROW_TERMINAL = False


def _resolve_render_mode() -> tuple[bool, int]:
    """Return (use_rich, terminal_columns) for the current render call."""
    columns = get_terminal_columns()
    if not is_layout_supported(columns):
        _warn_narrow_terminal_once(columns)
        return False, columns
    _reset_narrow_warning()
    return _RICH_AVAILABLE, columns


def _plain_rule(columns: int) -> str:
    """Return a separator that fits within the current terminal."""
    return "─" * max(columns - 2, 12)


def _format_table_value(key: str, value: Any) -> str:
    """Format a structured value for rich/plain table rendering."""
    if value in ("", None):
        return "-"

    if key == "source_type":
        return EVIDENCE_SOURCE_LABELS.get(str(value), str(value))
    if key == "page":
        return f"p.{value}"
    if key == "score":
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)
    if key in {"hits", "population", "pttn", "dfpt", "saeol"}:
        try:
            return f"{int(float(value)):,}"
        except (TypeError, ValueError):
            return str(value)
    if key == "value":
        try:
            value_f = float(value)
            return f"{value_f:,.1f}" if value_f % 1 else f"{value_f:,.0f}"
        except (TypeError, ValueError):
            return str(value)
    if key in {"ratio", "prebRatio", "prevRatio"}:
        text = str(value)
        return text if text.endswith("%") else f"{text}%"
    return str(value)


def _select_table_columns(rows: list[dict], columns: int) -> list[str]:
    """Select visible table columns based on row shape and terminal width."""
    visible_keys: list[str] = []
    seen: set[str] = set()

    for key in TABLE_COLUMN_PRIORITY:
        if any(row.get(key) not in ("", None) for row in rows):
            visible_keys.append(key)
            seen.add(key)

    for row in rows:
        for key in row:
            if key in TABLE_HIDDEN_KEYS or key in seen:
                continue
            if row.get(key) not in ("", None):
                visible_keys.append(key)
                seen.add(key)

    max_columns = 5 if columns >= 120 else 4 if columns >= 80 else 2
    return visible_keys[:max_columns]


def _build_rich_table(rows: list[dict], columns: int, *, column_keys: list[str] | None = None):
    """Build a Rich table from structured rows."""
    selected_keys = column_keys or _select_table_columns(rows, columns)
    if not selected_keys:
        return None

    table = Table(expand=True)
    for key in selected_keys:
        table.add_column(
            TABLE_COLUMN_LABELS.get(key, key),
            overflow="fold",
            no_wrap=key in {"source_type", "page", "score"},
        )

    for row in rows:
        table.add_row(*(_format_table_value(key, row.get(key)) for key in selected_keys))

    return table


def _render_plain_table(
    title: str,
    rows: list[dict],
    columns: int,
    *,
    column_keys: list[str] | None = None,
) -> str:
    """Render structured rows as a tab-delimited plain-text table."""
    selected_keys = column_keys or _select_table_columns(rows, columns)
    if not selected_keys:
        return ""

    lines = [title, "\t".join(TABLE_COLUMN_LABELS.get(key, key) for key in selected_keys)]
    for row in rows:
        lines.append("\t".join(_format_table_value(key, row.get(key)) for key in selected_keys))
    return "\n".join(lines)


def _iter_structured_result_sections(tool_results: dict[str, Any]) -> list[tuple[str, list[dict]]]:
    """Extract table-ready structured result sections from tool results."""
    sections: list[tuple[str, list[dict]]] = []

    for tool_name in STRUCTURED_TOOL_ORDER:
        payload = tool_results.get(tool_name)
        if not isinstance(payload, dict):
            continue
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            continue

        grouped_rows: dict[str, list[dict]] = {}
        for row in results:
            if not isinstance(row, dict):
                continue
            grouped_rows.setdefault(str(row.get("_source_api") or "results"), []).append(row)

        for source_api, rows in grouped_rows.items():
            source_title = STRUCTURED_API_TITLES.get(source_api)
            tool_title = STRUCTURED_TOOL_TITLES.get(tool_name, tool_name)
            title = f"{tool_title} · {source_title}" if source_title else tool_title
            sections.append((title, rows))

    return sections


def _build_evidence_table_rows(evidence_items: list[dict]) -> list[dict]:
    """Normalize evidence items into a table-oriented row schema."""
    rows: list[dict] = []
    for item in evidence_items:
        rows.append(
            {
                "source_type": item.get("source_type"),
                "title": item.get("title") or item.get("excerpt", ""),
                "page": item.get("page"),
                "score": item.get("score"),
                "link_or_path": item.get("link_or_path"),
            }
        )
    return rows


def _select_evidence_columns(columns: int) -> list[str]:
    """Return evidence table columns based on terminal width."""
    if columns >= 120:
        return ["source_type", "title", "page", "score", "link_or_path"]
    if columns >= 80:
        return ["source_type", "title", "score"]
    return ["source_type", "title"]


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


def _build_citations_text(citations: list[str]) -> Text:
    """Return a styled fallback citations block for rich rendering."""
    content = Text("\n출처\n", style="bold")
    for idx, src in enumerate(citations, 1):
        content.append(f"  {idx}. {src}\n", style="dim")
    return content


def render_result(result: dict) -> None:
    """Render the final agent response inline (no Panel box).

    Claude Code style: full-width separator, free-flowing markdown, separator.

    Expected keys (at least one required):
      - result["text"] or result["response"]: main answer text
      - result["evidence_items"]: EvidenceItem dict list (structured, preferred)
      - result["citations"] or result["sources"]: list of source strings (fallback)
      - result["tool_results"]: stats/keyword/demographics structured result dict
    """
    text_body: str = result.get("text") or result.get("response") or ""
    evidence_items: list = result.get("evidence_items") or []
    citations: list = result.get("citations") or result.get("sources") or []
    tool_results: dict[str, Any] = result.get("tool_results") or {}

    use_rich, columns = _resolve_render_mode()

    if use_rich:
        theme = get_theme()
        _console.print()

        # Inline markdown — no Panel box
        if text_body:
            _console.print(Markdown(text_body, code_theme=MARKDOWN_CODE_THEME))

        # Structured tool result tables
        for title, rows in _iter_structured_result_sections(tool_results):
            table = _build_rich_table(rows, columns)
            if table is None:
                continue
            _console.print()
            _console.print(Text(title, style=theme.brand_accent))
            _console.print(table)

        # Evidence / citations
        if evidence_items:
            evidence_rows = _build_evidence_table_rows(evidence_items)
            evidence_table = _build_rich_table(
                evidence_rows,
                columns,
                column_keys=_select_evidence_columns(columns),
            )
            if evidence_table is not None:
                _console.print()
                _console.print(Text("참조 근거", style="bold"))
                _console.print(evidence_table)
        elif citations:
            _console.print(_build_citations_text(citations))

        _console.print()
    else:
        rule = _plain_rule(columns)
        print(f"\n{rule}")
        print(text_body)
        for title, rows in _iter_structured_result_sections(tool_results):
            table_text = _render_plain_table(title, rows, columns)
            if table_text:
                print(f"\n{table_text}")
        if evidence_items:
            evidence_table = _render_plain_table(
                "참조 근거",
                _build_evidence_table_rows(evidence_items),
                columns,
                column_keys=_select_evidence_columns(columns),
            )
            if evidence_table:
                print(f"\n{evidence_table}")
        elif citations:
            print("\n출처")
            for idx, src in enumerate(citations, 1):
                print(f"  {idx}. {src}")
        print(f"{rule}\n")


def render_status(message: str) -> None:
    """Render a transient status / progress message to stderr."""
    use_rich, _ = _resolve_render_mode()
    if use_rich:
        theme = get_theme()
        _stderr_console.print(f"[{theme.text_secondary}]→ {message}[/{theme.text_secondary}]")
    else:
        print(f"→ {message}", file=sys.stderr)


def render_error(message: str) -> None:
    """Render an error message in red to stderr."""
    use_rich, _ = _resolve_render_mode()
    if use_rich:
        theme = get_theme()
        _stderr_console.print(f"[{theme.status_error}]오류:[/{theme.status_error}] {message}")
    else:
        print(f"오류: {message}", file=sys.stderr)


def render_thinking(content: str) -> None:
    """LLM thinking 과정을 dim 스타일로 표시."""
    use_rich, _ = _resolve_render_mode()
    if use_rich:
        theme = get_theme()
        _console.print(f"[{theme.text_secondary}]{content}[/{theme.text_secondary}]", end="")
    else:
        print(content, end="", flush=True)


def render_tool_progress(tool_name: str, status: str, latency_ms: float = 0) -> None:
    """도구 실행 3단계 진행 표시 (start / end)."""
    use_rich, _ = _resolve_render_mode()
    theme = get_theme()

    if status == "start":
        msg = f"┌ ⚙ {tool_name}"
        style = theme.tool_start
    else:
        if latency_ms:
            msg = f"└ ✦ {tool_name} 완료 ({latency_ms:.0f}ms)"
        else:
            msg = f"└ ✦ {tool_name} 완료"
        style = theme.tool_end

    if use_rich:
        _console.print(f"[{style}]{msg}[/{style}]")
    else:
        print(msg, flush=True)


def render_metadata(metadata: dict) -> None:
    """실행 메타데이터 (iterations, tool calls, latency) 표시."""
    iterations = metadata.get("total_iterations", 0)
    tool_calls = metadata.get("total_tool_calls", 0)
    latency = metadata.get("total_latency_ms", 0)

    summary = f"iterations={iterations}  tools={tool_calls}  latency={latency:.0f}ms"

    use_rich, _ = _resolve_render_mode()
    if use_rich:
        theme = get_theme()
        _console.print(f"[{theme.text_secondary}]⎯ {summary}[/{theme.text_secondary}]")
    else:
        print(f"⎯ {summary}")


def render_session_info(session_id: str) -> None:
    """Render session resume hint at shell exit."""
    hint = f"[session: {session_id}]  govon --session {session_id} 로 재개 가능"
    use_rich, _ = _resolve_render_mode()
    if use_rich:
        theme = get_theme()
        _console.print(f"[{theme.text_secondary}]{hint}[/{theme.text_secondary}]")
    else:
        print(hint)
