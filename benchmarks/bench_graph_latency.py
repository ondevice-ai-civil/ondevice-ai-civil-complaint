"""LangGraph 노드별 레이턴시 벤치마크.

Issue #163: graph 전체 및 노드별 레이턴시 p50/p95/p99 측정.

사용법::

    SKIP_MODEL_LOAD=true python -m benchmarks.bench_graph_latency --repeat 5

mock adapter를 사용하여 실제 LLM/API 호출 없이 graph 실행 레이턴시를 측정한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

os.environ.setdefault("SKIP_MODEL_LOAD", "true")

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.inference.graph.builder import build_govon_graph  # noqa: E402
from src.inference.graph.executor_adapter import ExecutorAdapter  # noqa: E402
from src.inference.graph.planner_adapter import RegexPlannerAdapter  # noqa: E402
from src.inference.session_context import SessionStore  # noqa: E402

# ---------------------------------------------------------------------------
# Mock adapters with realistic latencies
# ---------------------------------------------------------------------------

SIMULATED_LATENCIES = {
    "rag_search": 0.2,  # 200ms
    "api_lookup": 0.5,  # 500ms
    "draft_civil_response": 1.0,  # 1000ms
    "append_evidence": 0.3,  # 300ms
}


class BenchExecutorAdapter(ExecutorAdapter):
    """벤치마크용 executor — 현실적인 지연 시뮬레이션."""

    async def execute(
        self,
        tool_name: str,
        query: str,
        context: dict,
    ) -> dict:
        latency = SIMULATED_LATENCIES.get(tool_name, 0.1)
        await asyncio.sleep(latency)
        return {
            "success": True,
            "text": f"[bench] {tool_name} result",
            "latency_ms": latency * 1000,
        }

    def list_tools(self) -> list[str]:
        return list(SIMULATED_LATENCIES.keys())


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_single(run_id: int, db_dir: str) -> dict:
    """단일 graph 실행 후 레이턴시를 반환한다."""
    # run마다 별도 DB를 사용하여 측정 편향 방지
    db_path = os.path.join(db_dir, f"bench_sessions_{run_id}.sqlite3")
    session_store = SessionStore(db_path=db_path)
    planner = RegexPlannerAdapter()
    executor = BenchExecutorAdapter()
    graph = build_govon_graph(
        planner_adapter=planner,
        executor_adapter=executor,
        session_store=session_store,
        checkpointer=MemorySaver(),
    )

    thread_id = f"bench-{run_id}"
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "session_id": f"bench-session-{run_id}",
        "request_id": f"bench-request-{run_id}",
        "messages": [HumanMessage(content="도로 파손 관련 민원 답변 초안 작성해주세요")],
    }

    t0 = time.monotonic()

    # Phase 1: interrupt까지 실행
    graph.invoke(initial, config=config)

    # Phase 2: 승인 후 완료
    result = graph.invoke(
        Command(resume={"approved": True}),
        config=config,
    )

    total_ms = round((time.monotonic() - t0) * 1000, 2)

    # node_latencies는 _merge_dicts reducer로 누적되어 최종 state에 포함된다.
    node_latencies: dict = result.get("node_latencies") or {}

    return {
        "run_id": run_id,
        "total_ms": total_ms,
        "has_final_text": bool(result.get("final_text")),
        "node_latencies": node_latencies,
    }


def _percentile(data: list[float], p: float) -> float:
    """백분위수를 계산한다."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph latency benchmark")
    parser.add_argument("--repeat", type=int, default=5, help="반복 실행 횟수")
    parser.add_argument("--output", type=str, default=None, help="JSON 출력 파일 경로")
    args = parser.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory(prefix="govon_bench_") as db_dir:
        results = []
        for i in range(args.repeat):
            r = _run_single(i, db_dir)
            results.append(r)
            print(f"  Run {i}: {r['total_ms']:.1f}ms", file=sys.stderr)

    # TemporaryDirectory 종료 후에도 results 리스트는 여전히 접근 가능
    totals = [r["total_ms"] for r in results]

    # 노드별 레이턴시 통계 집계
    all_node_keys: set[str] = set()
    for r in results:
        all_node_keys.update(r.get("node_latencies", {}).keys())

    node_stats: dict = {}
    for node_key in sorted(all_node_keys):
        values = [
            r["node_latencies"][node_key]
            for r in results
            if node_key in r.get("node_latencies", {})
        ]
        if values:
            node_stats[node_key] = {
                "p50": round(_percentile(values, 50), 2),
                "p95": round(_percentile(values, 95), 2),
                "mean": round(statistics.mean(values), 2),
            }

    stats = {
        "repeat": args.repeat,
        "total_ms": {
            "p50": round(_percentile(totals, 50), 2),
            "p95": round(_percentile(totals, 95), 2),
            "p99": round(_percentile(totals, 99), 2),
            "mean": round(statistics.mean(totals), 2),
            "stdev": round(statistics.stdev(totals), 2) if len(totals) > 1 else 0,
        },
        "node_latencies_ms": node_stats,
        "runs": results,
    }

    output = json.dumps(stats, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output)

    print(
        f"\n  Summary (n={args.repeat}): "
        f"p50={stats['total_ms']['p50']}ms "
        f"p95={stats['total_ms']['p95']}ms "
        f"p99={stats['total_ms']['p99']}ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
