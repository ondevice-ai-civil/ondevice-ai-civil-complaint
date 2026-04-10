/**
 * MockGovOnClient — offline mock implementing IClient.
 *
 * Routes queries to different TUI scenarios so every UI branch can be
 * exercised without a running backend.
 *
 * Activate via:  GOVON_MOCK=true node bin/govon.js
 *
 * ┌──────────────────────────────────────────────────────────────────┐
 * │  Test Prompts                                                    │
 * ├─────────────────┬────────────────────────────────────────────────┤
 * │  민원 통계       │  v3 normal: thinking → tools → answer stream  │
 * │  승인 테스트     │  v2 approval flow: ApprovalPrompt y/n         │
 * │  에러 테스트     │  v3 server error event                        │
 * │  도구 실패       │  v3 tool_end with success=false               │
 * │  간단 질문       │  v3 direct answer, no tool calls              │
 * │  긴 답변        │  v3 long markdown streaming stress test        │
 * │  (anything else) │  v3 normal (same as 민원 통계)                │
 * └─────────────────┴────────────────────────────────────────────────┘
 */

import type { IClient } from './client.interface.js';
import type {
  V3SSEEvent,
  V2SSEEvent,
  AgentRunResponse,
  ApprovalResponse,
} from './types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Abort-aware sleep. Rejects immediately with AbortError when the signal fires,
 * so Esc cancellation propagates without waiting for the full delay.
 */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }

    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);

    const onAbort = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      reject(new DOMException('Aborted', 'AbortError'));
    };

    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

function uuid(): string {
  return crypto.randomUUID();
}

/** Stream text word-by-word as response_delta events. */
async function* streamWords(
  text: string,
  delayMs: number,
  signal?: AbortSignal,
): AsyncGenerator<V3SSEEvent> {
  const words = text.split(' ');
  for (const word of words) {
    if (signal?.aborted) return;
    await sleep(delayMs, signal);
    yield { type: 'response_delta', content: word + ' ' };
  }
}

// ---------------------------------------------------------------------------
// Scenario: query → scenario key
// ---------------------------------------------------------------------------

type ScenarioKey = 'stats' | 'approval' | 'error' | 'tool_fail' | 'simple' | 'long';

function resolveScenario(query: string): ScenarioKey {
  const q = query.trim();
  if (q.includes('승인')) return 'approval';
  if (q.includes('에러') || q.includes('오류')) return 'error';
  if (q.includes('실패') || q.includes('도구 실패')) return 'tool_fail';
  if (q.includes('간단')) return 'simple';
  if (q.includes('긴 답변') || q.includes('길게')) return 'long';
  return 'stats';
}

// ---------------------------------------------------------------------------
// Mock answer data per scenario
// ---------------------------------------------------------------------------

const ANSWERS: Record<ScenarioKey, string> = {
  stats:
    '2024년 1분기 민원 통계를 분석한 결과, 총 **12,847건**의 민원이 접수되었습니다.\n\n' +
    '## 주요 카테고리\n\n' +
    '| 카테고리 | 건수 | 비율 |\n' +
    '|---------|------|------|\n' +
    '| 도로·교통 | 3,241 | 25.2% |\n' +
    '| 환경·위생 | 2,891 | 22.5% |\n' +
    '| 건축·주택 | 2,104 | 16.4% |\n' +
    '| 복지·보건 | 1,982 | 15.4% |\n' +
    '| 기타 | 2,629 | 20.5% |\n\n' +
    '전 분기 대비 **8.3%** 증가했으며, 도로·교통 분야의 증가율이 가장 높았습니다.',

  approval:
    '승인 후 실행한 결과, 해당 민원에 대한 상세 통계를 조회했습니다.\n\n' +
    '**승인된 작업:** 민원 데이터베이스 직접 조회\n' +
    '**결과:** 정상 처리 완료',

  error: '', // not used — error event is emitted instead

  tool_fail:
    '도구 실행 중 일부 오류가 발생했으나, 캐시된 데이터로 응답합니다.\n\n' +
    '> ⚠ `demographics_lookup` 도구가 타임아웃으로 실패했습니다.\n\n' +
    '사용 가능한 데이터 기준으로, 2024년 상반기 민원 접수 건수는 약 **24,500건**입니다.',

  simple:
    'GovOn은 정부 민원 데이터를 분석하는 AI 에이전트입니다.\n\n' +
    '주요 기능:\n' +
    '- 민원 통계 조회 및 분석\n' +
    '- 키워드 트렌드 분석\n' +
    '- 인구통계 기반 민원 패턴 파악\n\n' +
    '질문을 입력하면 적절한 도구를 선택하여 답변합니다.',

  long:
    '# 2024년 전국 민원 종합 분석 보고서\n\n' +
    '## 1. 개요\n\n' +
    '본 보고서는 2024년 1월부터 12월까지 접수된 전국 민원 데이터를 종합 분석한 결과입니다. ' +
    '총 **156,284건**의 민원을 대상으로 카테고리별, 지역별, 시기별 분석을 수행했습니다.\n\n' +
    '## 2. 카테고리별 분석\n\n' +
    '### 2.1 도로·교통 (38,421건, 24.6%)\n\n' +
    '도로·교통 분야는 전체 민원의 약 1/4을 차지하며, 특히 **보도블록 파손**(8,234건)과 ' +
    '**불법 주정차**(6,891건)가 상위를 기록했습니다.\n\n' +
    '```\n' +
    '보도블록 파손    ████████████████████  8,234건\n' +
    '불법 주정차      ████████████████     6,891건\n' +
    '신호등 고장      ████████████         5,102건\n' +
    '도로 포트홀      ██████████           4,567건\n' +
    '기타            ████████████████████████████  13,627건\n' +
    '```\n\n' +
    '### 2.2 환경·위생 (35,192건, 22.5%)\n\n' +
    '환경·위생 민원은 계절적 변동이 뚜렷합니다:\n\n' +
    '| 분기 | 건수 | 전분기 대비 |\n' +
    '|------|------|----------|\n' +
    '| Q1 | 6,234 | - |\n' +
    '| Q2 | 9,891 | +58.7% |\n' +
    '| Q3 | 12,456 | +25.9% |\n' +
    '| Q4 | 6,611 | -46.9% |\n\n' +
    '여름철(Q2-Q3)에 **악취**, **해충**, **쓰레기 불법투기** 민원이 급증하는 패턴입니다.\n\n' +
    '### 2.3 건축·주택 (25,604건, 16.4%)\n\n' +
    '건축·주택 민원의 주요 유형:\n' +
    '1. 층간소음 분쟁 — 9,234건 (36.1%)\n' +
    '2. 불법 건축물 신고 — 5,891건 (23.0%)\n' +
    '3. 주택 안전점검 요청 — 4,102건 (16.0%)\n' +
    '4. 리모델링 허가 문의 — 3,567건 (13.9%)\n' +
    '5. 기타 — 2,810건 (11.0%)\n\n' +
    '## 3. 지역별 분석\n\n' +
    '수도권(서울·경기·인천)이 전체의 **47.3%** 를 차지하며, ' +
    '인구 대비 민원 발생률은 세종시가 가장 높았습니다.\n\n' +
    '## 4. 결론 및 제언\n\n' +
    '- 도로·교통 인프라 예방 정비 확대 필요\n' +
    '- 여름철 환경·위생 민원 대응 체계 사전 구축\n' +
    '- 층간소음 조정 서비스 접근성 개선\n' +
    '- 데이터 기반 민원 예측 시스템 도입 검토\n\n' +
    '---\n' +
    '*이 보고서는 GovOn AI 에이전트가 자동 생성한 분석입니다.*',
};

// ---------------------------------------------------------------------------
// V3 scenario generators
// ---------------------------------------------------------------------------

async function* scenarioStats(
  query: string,
  sessionId: string,
  threadId: string,
  signal?: AbortSignal,
): AsyncGenerator<V3SSEEvent> {
  // Iteration 1: thinking → 2 tool calls
  yield { type: 'thinking_start', iteration: 1 };
  for (const line of [
    'Analyzing the user query to determine the appropriate tools... ',
    'The query is about civil complaint statistics. I should use the stats_lookup tool. ',
    'Let me also check for keyword analysis to provide additional context. ',
  ]) {
    if (signal?.aborted) return;
    await sleep(80, signal);
    yield { type: 'thinking_delta', content: line };
  }
  yield {
    type: 'thinking_end',
    iteration: 1,
    tool_calls: [
      { name: 'stats_lookup', args: { period: '2024-Q1' } },
      { name: 'keyword_analyzer', args: { query } },
    ],
  };

  // Tool execution
  yield { type: 'tool_start', tool: 'stats_lookup' };
  if (signal?.aborted) return;
  await sleep(300, signal);
  yield { type: 'tool_end', tool: 'stats_lookup', success: true };

  yield { type: 'tool_start', tool: 'keyword_analyzer' };
  if (signal?.aborted) return;
  await sleep(200, signal);
  yield { type: 'tool_end', tool: 'keyword_analyzer', success: true };

  // Iteration 2: compose final answer
  yield { type: 'thinking_start', iteration: 2 };
  await sleep(50, signal);
  yield { type: 'thinking_delta', content: 'Tools returned successfully. Composing the final answer now.' };
  yield { type: 'thinking_end', iteration: 2, tool_calls: [] };

  yield* streamWords(ANSWERS.stats, 30, signal);

  yield {
    type: 'run_complete',
    thread_id: threadId,
    session_id: sessionId,
    text: ANSWERS.stats,
    evidence_items: [
      { source_type: 'api', content: '2024-Q1 civil complaint statistics (mock)', score: 0.95 },
      { source_type: 'rag', content: 'Keyword frequency analysis for query (mock)', score: 0.87 },
    ],
    metadata: { total_iterations: 2, total_tool_calls: 2, total_latency_ms: 1200 },
  };
}

async function* scenarioError(
  _sessionId: string,
  signal?: AbortSignal,
): AsyncGenerator<V3SSEEvent> {
  // Start thinking, then crash mid-stream
  yield { type: 'thinking_start', iteration: 1 };
  await sleep(100, signal);
  yield { type: 'thinking_delta', content: 'Processing the query... ' };
  if (signal?.aborted) return;
  await sleep(200, signal);
  yield { type: 'thinking_delta', content: 'Attempting to connect to data source... ' };
  await sleep(300, signal);
  yield { type: 'error', error: '[MockError] Backend service unavailable: connection to data source timed out after 30s' };
}

async function* scenarioToolFail(
  query: string,
  sessionId: string,
  threadId: string,
  signal?: AbortSignal,
): AsyncGenerator<V3SSEEvent> {
  // Iteration 1: thinking → 2 tools, one fails
  yield { type: 'thinking_start', iteration: 1 };
  await sleep(60, signal);
  yield { type: 'thinking_delta', content: 'I need to look up demographics and statistics for this query. ' };
  yield {
    type: 'thinking_end',
    iteration: 1,
    tool_calls: [
      { name: 'stats_lookup', args: { period: '2024-H1' } },
      { name: 'demographics_lookup', args: { region: 'all', query } },
    ],
  };

  yield { type: 'tool_start', tool: 'stats_lookup' };
  if (signal?.aborted) return;
  await sleep(250, signal);
  yield { type: 'tool_end', tool: 'stats_lookup', success: true };

  yield { type: 'tool_start', tool: 'demographics_lookup' };
  if (signal?.aborted) return;
  await sleep(500, signal);
  // This one FAILS
  yield { type: 'tool_end', tool: 'demographics_lookup', success: false };

  // Iteration 2: recover and answer with partial data
  yield { type: 'thinking_start', iteration: 2 };
  await sleep(80, signal);
  yield { type: 'thinking_delta', content: 'demographics_lookup failed with timeout. I will use cached data to answer. ' };
  yield { type: 'thinking_end', iteration: 2, tool_calls: [] };

  yield* streamWords(ANSWERS.tool_fail, 30, signal);

  yield {
    type: 'run_complete',
    thread_id: threadId,
    session_id: sessionId,
    text: ANSWERS.tool_fail,
    evidence_items: [
      { source_type: 'api', content: '2024-H1 statistics (partial, mock)', score: 0.72 },
    ],
    metadata: { total_iterations: 2, total_tool_calls: 2, total_latency_ms: 1800 },
  };
}

async function* scenarioSimple(
  sessionId: string,
  threadId: string,
  signal?: AbortSignal,
): AsyncGenerator<V3SSEEvent> {
  // Single iteration — no tool calls, direct answer
  yield { type: 'thinking_start', iteration: 1 };
  await sleep(60, signal);
  yield { type: 'thinking_delta', content: 'This is a general question about GovOn. No tools needed — I can answer directly. ' };
  yield { type: 'thinking_end', iteration: 1, tool_calls: [] };

  yield* streamWords(ANSWERS.simple, 25, signal);

  yield {
    type: 'run_complete',
    thread_id: threadId,
    session_id: sessionId,
    text: ANSWERS.simple,
    evidence_items: [],
    metadata: { total_iterations: 1, total_tool_calls: 0, total_latency_ms: 400 },
  };
}

async function* scenarioLong(
  query: string,
  sessionId: string,
  threadId: string,
  signal?: AbortSignal,
): AsyncGenerator<V3SSEEvent> {
  // Iteration 1: thinking → 3 tool calls (heavy workload)
  yield { type: 'thinking_start', iteration: 1 };
  await sleep(80, signal);
  yield { type: 'thinking_delta', content: 'User is requesting a comprehensive analysis. I need multiple data sources. ' };
  yield { type: 'thinking_delta', content: 'Planning: stats_lookup for volume, keyword_analyzer for trends, demographics_lookup for regional breakdown. ' };
  yield {
    type: 'thinking_end',
    iteration: 1,
    tool_calls: [
      { name: 'stats_lookup', args: { period: '2024-full', detail: 'category' } },
      { name: 'keyword_analyzer', args: { query, depth: 'deep' } },
      { name: 'demographics_lookup', args: { region: 'all', period: '2024' } },
    ],
  };

  // 3 tools in sequence
  yield { type: 'tool_start', tool: 'stats_lookup' };
  if (signal?.aborted) return;
  await sleep(400, signal);
  yield { type: 'tool_end', tool: 'stats_lookup', success: true };

  yield { type: 'tool_start', tool: 'keyword_analyzer' };
  if (signal?.aborted) return;
  await sleep(350, signal);
  yield { type: 'tool_end', tool: 'keyword_analyzer', success: true };

  yield { type: 'tool_start', tool: 'demographics_lookup' };
  if (signal?.aborted) return;
  await sleep(500, signal);
  yield { type: 'tool_end', tool: 'demographics_lookup', success: true };

  // Iteration 2: compose long report
  yield { type: 'thinking_start', iteration: 2 };
  await sleep(60, signal);
  yield { type: 'thinking_delta', content: 'All three tools returned successfully. Composing a comprehensive report with tables and code blocks. ' };
  yield { type: 'thinking_end', iteration: 2, tool_calls: [] };

  // Stream the long answer more slowly to stress-test the display
  yield* streamWords(ANSWERS.long, 20, signal);

  yield {
    type: 'run_complete',
    thread_id: threadId,
    session_id: sessionId,
    text: ANSWERS.long,
    evidence_items: [
      { source_type: 'api', content: '2024 full-year complaint stats (mock)', score: 0.97 },
      { source_type: 'rag', content: 'Keyword trend analysis 2024 (mock)', score: 0.91 },
      { source_type: 'api', content: 'Regional demographics overlay (mock)', score: 0.88 },
    ],
    metadata: { total_iterations: 2, total_tool_calls: 3, total_latency_ms: 3200 },
  };
}

// ---------------------------------------------------------------------------
// MockGovOnClient
// ---------------------------------------------------------------------------

export class MockGovOnClient implements IClient {
  /**
   * Tracks pending approval requests by thread_id so that approve() can
   * resolve the waiting v2 stream and reuse the original session_id.
   */
  private readonly _pendingApprovals = new Map<
    string,
    { sessionId: string; resolve: () => void }
  >();

  async health(): Promise<Record<string, unknown>> {
    return { status: 'ok', mock: true };
  }

  async waitForReady(): Promise<boolean> {
    return true;
  }

  async *streamV3(
    query: string,
    sessionId?: string,
    _maxIterations?: number,
    signal?: AbortSignal,
  ): AsyncGenerator<V3SSEEvent> {
    const sid = sessionId ?? uuid();
    const tid = uuid();
    const scenario = resolveScenario(query);

    switch (scenario) {
      case 'approval':
        // v3 has no approval event — throw so useSSE falls through to v2
        throw new Error('v3 not supported for approval flow');

      case 'error':
        yield* scenarioError(sid, signal);
        return;

      case 'tool_fail':
        yield* scenarioToolFail(query, sid, tid, signal);
        return;

      case 'simple':
        yield* scenarioSimple(sid, tid, signal);
        return;

      case 'long':
        yield* scenarioLong(query, sid, tid, signal);
        return;

      case 'stats':
      default:
        yield* scenarioStats(query, sid, tid, signal);
        return;
    }
  }

  async *stream(
    query: string,
    sessionId?: string,
    _signal?: AbortSignal,
  ): AsyncGenerator<V2SSEEvent> {
    const sid = sessionId ?? uuid();
    const tid = uuid();
    const scenario = resolveScenario(query);

    if (scenario === 'approval') {
      // --- V2 approval flow ---
      yield { node: 'session_load', status: 'completed' };
      await sleep(100, _signal);

      yield {
        node: 'agent',
        status: 'completed',
        planned_tools: ['stats_lookup'],
        task_type: 'stats_query',
      };
      await sleep(150, _signal);

      // Trigger the approval prompt
      yield {
        node: 'approval_wait',
        status: 'awaiting_approval',
        thread_id: tid,
        session_id: sid,
        approval_request: {
          approval_id: uuid(),
          task_type: 'stats_query',
          description: '민원 데이터베이스에 직접 접근하여 통계를 조회합니다. 이 작업은 실시간 데이터를 사용하므로 승인이 필요합니다.',
          planned_tools: ['stats_lookup', 'demographics_lookup'],
        },
      };

      // Keep the generator alive until approve()/cancel() resolves it.
      // The TUI shows ApprovalPrompt and waits for y/n; once the user
      // decides, approve() resolves this promise and the stream ends.
      await new Promise<void>((resolve) => {
        this._pendingApprovals.set(tid, { sessionId: sid, resolve });
      });

      return;
    }

    // Default v2 fallback for non-approval scenarios
    yield { node: 'session_load', status: 'completed' };
    await sleep(100, _signal);

    yield {
      node: 'agent',
      status: 'completed',
      planned_tools: ['stats_lookup'],
    };
    await sleep(200, _signal);

    yield { node: 'tools', status: 'completed', tool_results: {} };
    await sleep(100, _signal);

    yield {
      node: 'persist',
      status: 'completed',
      final_text: ANSWERS[scenario] || ANSWERS.stats,
      evidence_items: [],
      session_id: sid,
      thread_id: tid,
    };
  }

  async run(
    query: string,
    sessionId?: string,
    _signal?: AbortSignal,
  ): Promise<AgentRunResponse> {
    await sleep(500, _signal);
    const sid = sessionId ?? uuid();
    const scenario = resolveScenario(query);
    const text = ANSWERS[scenario] || ANSWERS.stats;

    return {
      request_id: uuid(),
      session_id: sid,
      text,
      trace: {
        request_id: uuid(),
        session_id: sid,
        plan: ['stats_lookup'],
        plan_reason: `Mock blocking run for scenario: ${scenario}`,
        tool_results: [
          { tool: 'stats_lookup', success: true, latency_ms: 150, data: {} },
        ],
        total_latency_ms: 500,
      },
      thread_id: uuid(),
      evidence_items: [],
      metadata: { total_iterations: 1, total_tool_calls: 1, total_latency_ms: 500 },
    };
  }

  async runV3(
    query: string,
    sessionId?: string,
    _maxIterations?: number,
    _signal?: AbortSignal,
  ): Promise<Record<string, unknown>> {
    const result = await this.run(query, sessionId);
    return result as unknown as Record<string, unknown>;
  }

  async approve(threadId: string, approved: boolean): Promise<ApprovalResponse> {
    await sleep(200);

    // Retrieve and clean up the pending approval to reuse the original session_id
    const pending = this._pendingApprovals.get(threadId);
    const sid = pending?.sessionId ?? uuid();

    // Resolve the waiting v2 stream so the generator can finish
    if (pending) {
      pending.resolve();
      this._pendingApprovals.delete(threadId);
    }

    return {
      status: approved ? 'approved' : 'rejected',
      thread_id: threadId,
      session_id: sid,
      text: approved ? ANSWERS.approval : undefined,
      evidence_items: approved
        ? [{ source_type: 'api', content: 'Direct DB query result (mock)', score: 0.93 }]
        : [],
      approval_status: approved ? 'approved' : 'rejected',
    };
  }

  async cancel(threadId: string): Promise<Record<string, unknown>> {
    // Clean up any pending approval for this thread
    const pending = this._pendingApprovals.get(threadId);
    if (pending) {
      pending.resolve();
      this._pendingApprovals.delete(threadId);
    }
    return { status: 'cancelled', thread_id: threadId };
  }
}
