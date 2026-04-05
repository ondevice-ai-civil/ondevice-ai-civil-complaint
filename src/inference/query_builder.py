"""Context-aware query builder for search-oriented tools.

Issue #159: follow-up 요청에서 원문 질문, 기존 초안, 최근 tool 요약을 반영해
RAG/API 조회용 query variant를 만든다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

if TYPE_CHECKING:
    from .session_context import SessionContext


SEARCH_TOOL_HINTS: dict[str, str] = {
    "rag_search": "관련 법령 지침 매뉴얼 공지 내부 문서",
    "api_lookup": "유사 민원 사례 통계 최근 이슈",
}

_FOLLOW_UP_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"근거",
        r"출처",
        r"링크",
        r"이유",
        r"보강",
        r"추가",
        r"다시",
        r"수정",
        r"정중",
        r"공손",
        r"이 답변",
        r"위 답변",
        r"기존 답변",
    )
)

_MAX_QUERY_LEN = 480
_MAX_USER_LEN = 180
_MAX_ASSISTANT_LEN = 220
_MAX_TOOL_SUMMARY_LEN = 120

# 지시대명사/참조 표현: 이전 turn을 가리키는 표현이 있으면 follow-up으로 간주
_ANAPHORA_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"그거",
        r"이거",
        r"저거",
        r"이것",
        r"그것",
        r"저것",
        r"위\s*(답변|내용|글|설명|항목)",
        r"아래\s*(답변|내용|글|설명|항목)",
        r"이\s*(답변|내용|글|설명)",
        r"그\s*(답변|내용|글|설명)",
        r"기존\s*(답변|내용|글|초안)",
        r"방금",
        r"앞서",
        r"위에서",
    )
)

# 자기완결 판정: 쿼리에 독립 명사가 이 수 이상이면 자기완결적으로 간주
_SELF_CONTAINED_NOUN_MIN = 2


def build_runtime_query_context(session: "SessionContext", current_query: str) -> Dict[str, Any]:
    """세션에서 query builder 입력용 구조화 컨텍스트를 추출한다."""
    previous_user, previous_assistant = extract_previous_turns(session, current_query)
    return {
        "session_id": session.session_id,
        "query": normalize_text(current_query),
        "session_context": session.build_context_summary(),
        "previous_user_query": clip_text(previous_user, _MAX_USER_LEN),
        "previous_assistant_response": clip_text(previous_assistant, _MAX_ASSISTANT_LEN),
        "recent_tool_summary": clip_text(build_recent_tool_summary(session), _MAX_TOOL_SUMMARY_LEN),
    }


def extract_previous_turns(
    session: "SessionContext",
    current_query: str,
) -> tuple[str, str]:
    """현재 요청 직전의 user / assistant turn을 추출한다."""
    turns = list(session.recent_history)
    normalized_current = normalize_text(current_query)
    if (
        turns
        and turns[-1].role == "user"
        and normalize_text(turns[-1].content) == normalized_current
    ):
        turns = turns[:-1]

    previous_user = next(
        (normalize_text(turn.content) for turn in reversed(turns) if turn.role == "user"),
        "",
    )
    previous_assistant = next(
        (normalize_text(turn.content) for turn in reversed(turns) if turn.role == "assistant"),
        "",
    )
    return previous_user, previous_assistant


def build_recent_tool_summary(session: "SessionContext") -> str:
    """최근 tool 실행을 짧은 요약으로 변환한다."""
    parts: list[str] = []
    for record in session.recent_tool_runs[-3:]:
        tool_parts = [record.tool]
        if isinstance(record.metadata, dict):
            query = normalize_text(record.metadata.get("query", ""))
            text_preview = normalize_text(record.metadata.get("text_preview", ""))
            if query:
                tool_parts.append(clip_text(query, 60))
            elif text_preview:
                tool_parts.append(clip_text(text_preview, 60))
            count = record.metadata.get("count")
            if count is not None:
                tool_parts.append(f"count {count}")
        parts.append(" ".join(part for part in tool_parts if part))
    return " | ".join(parts)


def build_query_variants(
    query: str,
    *,
    tool_names: Sequence[str],
    context: Mapping[str, Any],
) -> Dict[str, str]:
    """RAG/API lookup용 tool-specific query variant를 생성한다."""
    normalized_query = normalize_text(query)
    previous_user = clip_text(context.get("previous_user_query", ""), _MAX_USER_LEN)
    previous_assistant = clip_text(
        context.get("previous_assistant_response", ""), _MAX_ASSISTANT_LEN
    )
    recent_tool_summary = clip_text(context.get("recent_tool_summary", ""), _MAX_TOOL_SUMMARY_LEN)
    follow_up = should_use_follow_up_context(
        normalized_query,
        tool_names=tool_names,
        previous_user=previous_user,
        previous_assistant=previous_assistant,
    )

    base_segments = [normalized_query]
    if follow_up:
        base_segments = []
        if previous_user:
            base_segments.append(previous_user)
        if previous_assistant:
            base_segments.append(previous_assistant)
        base_segments.append(normalized_query)
        if recent_tool_summary:
            base_segments.append(recent_tool_summary)

    variants: Dict[str, str] = {}
    for tool_name in tool_names:
        hint = SEARCH_TOOL_HINTS.get(tool_name)
        if not hint:
            continue
        variants[tool_name] = compose_query([*base_segments, hint], limit=_MAX_QUERY_LEN)
    return variants


def resolve_tool_query(tool_name: str, context: Mapping[str, Any]) -> str:
    """tool별 query variant가 있으면 사용하고, 없으면 원문 요청을 사용한다."""
    query_variants = context.get("query_variants", {})
    if isinstance(query_variants, Mapping):
        variant = normalize_text(query_variants.get(tool_name, ""))
        if variant:
            return variant
    return normalize_text(context.get("query", ""))


def is_self_contained_query(query: str) -> bool:
    """쿼리가 이전 맥락 없이 독립적으로 이해 가능한지 판정한다.

    지시대명사나 참조 표현이 없고, 독립 명사(공백 구분 토큰)가 충분히
    포함된 쿼리는 자기완결적으로 간주한다.
    """
    if any(pattern.search(query) for pattern in _ANAPHORA_PATTERNS):
        return False
    tokens = [t for t in query.split() if len(t) >= 2]
    return len(tokens) >= _SELF_CONTAINED_NOUN_MIN


def should_use_follow_up_context(
    query: str,
    *,
    tool_names: Sequence[str],
    previous_user: str,
    previous_assistant: str,
) -> bool:
    """이전 user/assistant turn을 query에 섞어야 하는 follow-up인지 판단한다.

    ``append_evidence`` 플랜이더라도 쿼리가 자기완결적이면 이전 맥락을
    주입하지 않는다. 실제 후속 질문(지시대명사·참조 표현 포함 또는
    _FOLLOW_UP_PATTERNS 매칭)인 경우에만 True를 반환한다.
    """
    if not (previous_user or previous_assistant):
        return False

    # 자기완결적 쿼리면 append_evidence 플랜이어도 이전 맥락 주입 안 함
    if is_self_contained_query(query):
        return False

    if "append_evidence" in tool_names:
        return True
    return any(pattern.search(query) for pattern in _FOLLOW_UP_PATTERNS)


def compose_query(parts: Sequence[Any], *, limit: int) -> str:
    """중복과 과도한 길이를 줄여 검색용 query를 합성한다."""
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = normalize_text(part)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    query = " ".join(deduped).strip()
    return clip_text(query, limit)


def clip_text(value: Any, limit: int) -> str:
    """긴 문장을 고정 길이로 잘라낸다."""
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def normalize_text(value: Any) -> str:
    """공백을 정규화한 단일 라인 텍스트로 변환한다."""
    return re.sub(r"\s+", " ", str(value or "")).strip()
