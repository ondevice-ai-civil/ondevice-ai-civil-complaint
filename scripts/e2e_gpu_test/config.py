"""E2E 테스트 설정 및 상수."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict

BASE_URL = os.environ.get("GOVON_RUNTIME_URL", "http://localhost:7860").rstrip("/")
API_KEY = os.environ.get("API_KEY")
TIMEOUT = int(os.environ.get("E2E_TIMEOUT", "300"))
BASE_MODEL = os.environ.get("BASE_MODEL", "LGAI-EXAONE/EXAONE-4.0-32B-AWQ")

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = os.environ.get("E2E_RESULTS_PATH", f"e2e_results_{TIMESTAMP}.json")
LOG_PATH = os.environ.get("E2E_LOG_PATH", f"e2e_log_{TIMESTAMP}.jsonl")

VALID_TOOLS = frozenset(
    {
        "api_lookup",
        "draft_response",
        "issue_detector",
        "stats_lookup",
        "keyword_analyzer",
        "demographics_lookup",
    }
)

# 6-node 그래프 표준 흐름 (승인 경로)
EXPECTED_APPROVED_FLOW = [
    "session_load",
    "planner",
    "approval_wait",
    "tool_execute",
    "synthesis",
    "persist",
]

# 거절 경로
EXPECTED_REJECTED_FLOW = [
    "session_load",
    "planner",
    "approval_wait",
    "persist",
]


@dataclass(frozen=True)
class NodeSLA:
    """노드별 SLA 임계값 (초)."""

    name: str
    max_p95_sec: float


NODE_SLA_THRESHOLDS: list[NodeSLA] = [
    NodeSLA("planner", 10.0),
    NodeSLA("tool_execute", 60.0),
    NodeSLA("synthesis", 30.0),
    NodeSLA("session_load", 5.0),
    NodeSLA("persist", 5.0),
]


LEGAL_PATTERNS = [
    r"제\s*\d+\s*조",
    r"제\s*\d+\s*항",
    r"법률",
    r"시행령",
    r"조례",
    r"판례",
    r"대법원",
    r"법",
    r"령",
    r"규정",
]

CIVIL_LAW_PATTERNS = [r"민법", r"제\s*\d+\s*조", r"임대차", r"계약", r"손해배상", r"채권", r"채무"]
CRIMINAL_LAW_PATTERNS = [r"형법", r"형사", r"처벌", r"벌금", r"징역", r"보호법", r"제\s*\d+\s*조"]
IP_PATTERNS = [r"상표법", r"특허법", r"저작권", r"지식재산", r"제\s*\d+\s*조", r"침해"]
PRECEDENT_PATTERNS = [r"대법원", r"판례", r"판결", r"선고", r"\d{4}\s*[다나]\s*\d+"]
