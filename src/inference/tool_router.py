"""GovOn MVP용 tool type 정의 모듈."""

from __future__ import annotations

from enum import Enum


class ToolType(str, Enum):
    """MVP 내부 capability 카탈로그."""

    RAG_SEARCH = "rag_search"
    API_LOOKUP = "api_lookup"
    ISSUE_DETECTOR = "issue_detector"
    STATS_LOOKUP = "stats_lookup"
    KEYWORD_ANALYZER = "keyword_analyzer"
    DEMOGRAPHICS_LOOKUP = "demographics_lookup"


ToolName = ToolType | str


def tool_name(tool: ToolName) -> str:
    return tool.value if isinstance(tool, ToolType) else str(tool)
