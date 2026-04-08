"""파이프라인 흐름 추적 및 검증.

SSE 이벤트 스트림에서 노드 전이를 추출하고,
기대 흐름과 비교하여 파이프라인 무결성을 검증한다.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import EXPECTED_APPROVED_FLOW, EXPECTED_REJECTED_FLOW


@dataclass
class NodeTransition:
    """노드 전이 기록."""

    node: str
    status: str  # "started" | "completed" | "error"
    timestamp: float
    latency_ms: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


class PipelineFlowTracker:
    """SSE 이벤트에서 노드 전이를 추출하고 추적한다."""

    def __init__(self) -> None:
        self._transitions: List[NodeTransition] = []
        self._start_time: float = time.monotonic()
        self._node_starts: Dict[str, float] = {}

    def reset(self) -> None:
        self._transitions = []
        self._start_time = time.monotonic()
        self._node_starts = {}

    def track_event(self, event: Dict[str, Any]) -> None:
        """SSE 이벤트를 파싱하여 노드 전이를 기록한다."""
        node = event.get("node", "")
        status = event.get("status", "")
        now = time.monotonic()

        if not node:
            return

        if status in ("started", "running"):
            self._node_starts[node] = now
            self._transitions.append(
                NodeTransition(
                    node=node,
                    status="started",
                    timestamp=now - self._start_time,
                )
            )
        elif status in ("completed", "done", "awaiting_approval"):
            start = self._node_starts.pop(node, now)
            latency_ms = (now - start) * 1000
            self._transitions.append(
                NodeTransition(
                    node=node,
                    status="completed",
                    timestamp=now - self._start_time,
                    latency_ms=latency_ms,
                    detail=event,
                )
            )
        elif status == "error":
            self._transitions.append(
                NodeTransition(
                    node=node,
                    status="error",
                    timestamp=now - self._start_time,
                    detail=event,
                )
            )

    @property
    def node_sequence(self) -> List[str]:
        """관측된 노드 순서 (중복 제거, 순서 유지)."""
        seen = set()
        result = []
        for t in self._transitions:
            if t.node not in seen:
                seen.add(t.node)
                result.append(t.node)
        return result

    @property
    def transitions(self) -> List[NodeTransition]:
        return list(self._transitions)

    def to_text(self) -> str:
        """노드 흐름을 텍스트로 표현한다.

        예: "session_load(2ms) -> planner(1.2s) -> approval_wait -> ..."
        """
        parts = []
        for t in self._transitions:
            if t.status == "completed" and t.latency_ms > 0:
                if t.latency_ms >= 1000:
                    parts.append(f"{t.node}({t.latency_ms / 1000:.1f}s)")
                else:
                    parts.append(f"{t.node}({t.latency_ms:.0f}ms)")
            elif t.status == "started":
                continue  # completed에서 처리
            else:
                parts.append(t.node)
        return " -> ".join(parts) if parts else "(no transitions)"


class FlowValidator:
    """기대 흐름과 실제 흐름을 비교한다."""

    @staticmethod
    def validate_approved_flow(actual_nodes: List[str]) -> tuple[bool, List[str]]:
        """승인 경로 검증. (valid, issues) 반환."""
        return FlowValidator._validate(actual_nodes, EXPECTED_APPROVED_FLOW)

    @staticmethod
    def validate_rejected_flow(actual_nodes: List[str]) -> tuple[bool, List[str]]:
        """거절 경로 검증."""
        return FlowValidator._validate(actual_nodes, EXPECTED_REJECTED_FLOW)

    @staticmethod
    def _validate(actual: List[str], expected: List[str]) -> tuple[bool, List[str]]:
        issues: List[str] = []

        # 기대 노드가 실제에 순서대로 포함되어 있는지 확인
        expected_idx = 0
        for node in actual:
            if expected_idx < len(expected) and node == expected[expected_idx]:
                expected_idx += 1

        if expected_idx < len(expected):
            missing = expected[expected_idx:]
            issues.append(f"누락된 노드: {missing}")

        # 비정상 노드 감지
        unexpected = [n for n in actual if n not in expected and n != "__interrupt__"]
        if unexpected:
            issues.append(f"예상 외 노드: {unexpected}")

        return len(issues) == 0, issues


class LatencyAggregator:
    """노드별 레이턴시 통계를 수집한다."""

    def __init__(self) -> None:
        self._latencies: Dict[str, List[float]] = {}

    def record(self, node: str, latency_ms: float) -> None:
        self._latencies.setdefault(node, []).append(latency_ms)

    def record_from_tracker(self, tracker: PipelineFlowTracker) -> None:
        """FlowTracker의 전이에서 레이턴시를 수집한다."""
        for t in tracker.transitions:
            if t.status == "completed" and t.latency_ms > 0:
                self.record(t.node, t.latency_ms)

    def stats(self, node: str) -> Dict[str, float]:
        """노드의 min/max/avg/p95/count를 반환한다."""
        values = self._latencies.get(node, [])
        if not values:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "p95": 0}

        sorted_v = sorted(values)
        p95_idx = max(0, int(len(sorted_v) * 0.95) - 1)
        return {
            "count": len(values),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "avg": round(statistics.mean(values), 2),
            "p95": round(sorted_v[p95_idx], 2),
        }

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        return {node: self.stats(node) for node in sorted(self._latencies.keys())}

    def summary_text(self) -> str:
        """레이턴시 요약을 텍스트로 반환."""
        lines = []
        for node, s in self.all_stats().items():
            lines.append(
                f"  {node}: avg={s['avg']:.0f}ms p95={s['p95']:.0f}ms "
                f"min={s['min']:.0f}ms max={s['max']:.0f}ms (n={s['count']})"
            )
        return "\n".join(lines) if lines else "  (no latency data)"
