/**
 * GovOn local daemon API HTTP client.
 *
 * TypeScript port of src/cli/http_client.py.
 * Wraps the REST/SSE API of the local daemon (uvicorn).
 * Core endpoints: health / run / stream / approve / cancel.
 */

import { EventSourceParserStream } from 'eventsource-parser/stream';
import type { V3SSEEvent, V2SSEEvent, AgentRunResponse, ApprovalResponse } from './types.js';
import { getBaseUrl, TIMEOUTS } from './config.js';

// ---------------------------------------------------------------------------
// Internal timeout helpers
// ---------------------------------------------------------------------------

/** Milliseconds to AbortSignal with combined connect + read budget. */
function makeSignal(ms: number): AbortSignal {
  return AbortSignal.timeout(ms);
}

// ---------------------------------------------------------------------------
// GovOnClient
// ---------------------------------------------------------------------------

export class GovOnClient {
  private readonly _baseUrl: string;

  /** Cold-start polling timeout in ms (from config). */
  private static readonly _COLD_START_TIMEOUT_MS = TIMEOUTS.coldStart;

  /** Cold-start polling interval in ms (from config). */
  private static readonly _COLD_START_INTERVAL_MS = TIMEOUTS.coldStartInterval;

  /**
   * @param baseUrl - Daemon base URL (e.g. "http://127.0.0.1:8000").
   *                  Trailing slash is stripped automatically.
   */
  constructor(baseUrl: string = getBaseUrl()) {
    this._baseUrl = baseUrl.replace(/\/+$/, '');
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * GET /health — check daemon status.
   *
   * @returns Health response from the server.
   * @throws Error when daemon is unreachable.
   */
  async health(): Promise<Record<string, unknown>> {
    return this._get('/health', TIMEOUTS.default);
  }

  /**
   * Poll GET /health until the server responds with HTTP 200.
   *
   * Handles cold-start / sleeping remote servers (e.g. HF Space) by
   * printing Korean status messages to stderr and waiting patiently.
   *
   * @returns true when server is ready, false on timeout.
   */
  async waitForReady(): Promise<boolean> {
    const url = `${this._baseUrl}/health`;
    const deadline = Date.now() + GovOnClient._COLD_START_TIMEOUT_MS;
    const intervalMs = GovOnClient._COLD_START_INTERVAL_MS;

    let lastStatus = '';
    let attempt = 0;

    while (Date.now() < deadline) {
      let newStatus: string;

      try {
        const signal = makeSignal(10_000);
        const resp = await fetch(url, { signal });

        if (resp.status === 200) {
          if (attempt > 0) {
            process.stderr.write('\r✦ 서버 준비 완료.                              \n');
          }
          return true;
        }

        if (resp.status === 503) {
          newStatus = '⊛ 서버 시작 중… (503 응답 대기)';
        } else {
          newStatus = `⊛ 서버 응답 대기 중… (HTTP ${resp.status})`;
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);

        if (
          msg.includes('ECONNREFUSED') ||
          msg.includes('ENOTFOUND') ||
          msg.includes('fetch failed') ||
          msg.includes('connect')
        ) {
          newStatus = '⊛ 서버에 연결 중… (sleeping 상태에서 깨어나는 중)';
        } else if (msg.includes('TimeoutError') || msg.includes('timed out') || msg.includes('abort')) {
          newStatus = '⊛ 서버 응답 대기 중… (빌드 또는 모델 로딩 중)';
        } else {
          newStatus = '⊛ 서버에 연결 중… (sleeping 상태에서 깨어나는 중)';
        }
      }

      const elapsed = Math.round((Date.now() - (deadline - GovOnClient._COLD_START_TIMEOUT_MS)) / 1000);
      if (newStatus !== lastStatus || attempt % 6 === 0) {
        process.stderr.write(`\r⏳ ${newStatus} (${elapsed}s)`);
        lastStatus = newStatus;
      }

      attempt += 1;
      await sleep(intervalMs);
    }

    process.stderr.write('\r✘ 서버 연결 시간 초과.                            \n');
    return false;
  }

  /**
   * POST /v3/agent/stream — v3 ReAct fine-grained SSE streaming.
   *
   * @param query         - User input query.
   * @param sessionId     - Session ID to resume an existing session.
   * @param maxIterations - Maximum ReAct loop iterations.
   * @param signal        - Optional external AbortSignal; composed with the internal timeout.
   * @yields Parsed SSE event objects. Use the `type` key to distinguish them.
   */
  async *streamV3(
    query: string,
    sessionId?: string,
    maxIterations?: number,
    signal?: AbortSignal,
  ): AsyncGenerator<V3SSEEvent> {
    const body: Record<string, unknown> = { query };
    if (sessionId !== undefined) body['session_id'] = sessionId;
    if (maxIterations !== undefined) body['max_iterations'] = maxIterations;

    const url = `${this._baseUrl}/v3/agent/stream`;

    yield* this._sseStream<V3SSEEvent>(url, body, 'stream_v3', signal);
  }

  /**
   * POST /v2/agent/stream — per-node SSE streaming.
   *
   * @param query     - User input query.
   * @param sessionId - Session ID to resume an existing session.
   * @param signal    - Optional external AbortSignal; composed with the internal timeout.
   * @yields Parsed SSE event objects. Contains at least `node` and `status` keys.
   */
  async *stream(query: string, sessionId?: string, signal?: AbortSignal): AsyncGenerator<V2SSEEvent> {
    const body: Record<string, unknown> = { query };
    if (sessionId !== undefined) body['session_id'] = sessionId;

    const url = `${this._baseUrl}/v2/agent/stream`;

    yield* this._sseStream<V2SSEEvent>(url, body, 'stream', signal);
  }

  /**
   * POST /v2/agent/run — blocking agent execution.
   *
   * @param query     - User input query.
   * @param sessionId - Session ID to resume an existing session.
   * @param signal    - Optional external AbortSignal; composed with the internal timeout.
   * @returns Server response including thread_id, status, etc.
   */
  async run(query: string, sessionId?: string, signal?: AbortSignal): Promise<AgentRunResponse> {
    const body: Record<string, unknown> = { query };
    if (sessionId !== undefined) body['session_id'] = sessionId;

    return this._post('/v2/agent/run', body, TIMEOUTS.run, signal) as unknown as Promise<AgentRunResponse>;
  }

  /**
   * POST /v3/agent/run — v3 ReAct blocking execution.
   *
   * @param query         - User input query.
   * @param sessionId     - Session ID to resume an existing session.
   * @param maxIterations - Maximum ReAct loop iterations.
   * @param signal        - Optional external AbortSignal; composed with the internal timeout.
   * @returns Server response including metadata.
   */
  async runV3(
    query: string,
    sessionId?: string,
    maxIterations?: number,
    signal?: AbortSignal,
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { query };
    if (sessionId !== undefined) body['session_id'] = sessionId;
    if (maxIterations !== undefined) body['max_iterations'] = maxIterations;

    return this._post('/v3/agent/run', body, TIMEOUTS.run, signal);
  }

  /**
   * POST /v2/agent/approve — approve or reject a pending tool call.
   *
   * Note: uses QUERY PARAMETERS (thread_id, approved), not request body.
   *
   * @param threadId - Graph thread ID to approve or reject.
   * @param approved - true to approve, false to reject.
   * @returns Server response.
   */
  async approve(threadId: string, approved: boolean): Promise<ApprovalResponse> {
    const params = new URLSearchParams({
      thread_id: threadId,
      approved: String(approved).toLowerCase(),
    });
    return this._postParams('/v2/agent/approve', params, TIMEOUTS.default) as unknown as Promise<ApprovalResponse>;
  }

  /**
   * POST /v2/agent/cancel — cancel a running session.
   *
   * Note: uses QUERY PARAMETER (thread_id), not request body.
   *
   * @param threadId - Graph thread ID to cancel.
   * @returns Server response.
   */
  async cancel(threadId: string): Promise<Record<string, unknown>> {
    const params = new URLSearchParams({ thread_id: threadId });
    return this._postParams('/v2/agent/cancel', params, TIMEOUTS.default);
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  /** Shared SSE streaming implementation used by stream() and streamV3(). */
  private async *_sseStream<T>(
    url: string,
    body: Record<string, unknown>,
    label: string,
    externalSignal?: AbortSignal,
  ): AsyncGenerator<T> {
    // Compose the caller-provided signal with the internal read timeout so that
    // whichever fires first aborts the fetch.
    const timeoutSignal = AbortSignal.timeout(TIMEOUTS.read);
    const signal = externalSignal
      ? AbortSignal.any([externalSignal, timeoutSignal])
      : timeoutSignal;

    let response: Response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        msg.includes('ECONNREFUSED') ||
        msg.includes('ENOTFOUND') ||
        msg.includes('fetch failed') ||
        msg.includes('connect')
      ) {
        throw new Error(`daemon is not running. (${this._baseUrl})`);
      }
      throw err;
    }

    if (!response.ok) {
      throw new Error(`[${label}] HTTP ${response.status}: ${url}`);
    }

    if (!response.body) {
      throw new Error(`[${label}] Response body is null: ${url}`);
    }

    const sseStream = response.body
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new EventSourceParserStream());

    for await (const event of sseStream) {
      // EventSourceParserStream yields EventSourceMessage objects with { event?, data }
      if (event.data) {
        try {
          yield JSON.parse(event.data) as T;
        } catch {
          process.stderr.write(`[${label}] SSE JSON parse failed: len=${event.data.length}\n`);
          // Continue — do not break the stream on a single malformed frame
        }
      }
    }
  }

  /** GET helper with connection-error normalisation. */
  private async _get(path: string, timeoutMs: number): Promise<Record<string, unknown>> {
    const url = `${this._baseUrl}${path}`;
    let response: Response;
    try {
      response = await fetch(url, { signal: makeSignal(timeoutMs) });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        msg.includes('ECONNREFUSED') ||
        msg.includes('ENOTFOUND') ||
        msg.includes('fetch failed') ||
        msg.includes('connect')
      ) {
        throw new Error(`daemon is not running. (${this._baseUrl})`);
      }
      throw err;
    }

    if (!response.ok) {
      throw new Error(`[http_client] HTTP ${response.status}: ${url}`);
    }

    return response.json() as Promise<Record<string, unknown>>;
  }

  /** POST with JSON body helper. */
  private async _post(
    path: string,
    body: Record<string, unknown>,
    timeoutMs: number,
    externalSignal?: AbortSignal,
  ): Promise<Record<string, unknown>> {
    const url = `${this._baseUrl}${path}`;
    const timeoutSignal = makeSignal(timeoutMs);
    const signal = externalSignal
      ? AbortSignal.any([externalSignal, timeoutSignal])
      : timeoutSignal;
    let response: Response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        msg.includes('ECONNREFUSED') ||
        msg.includes('ENOTFOUND') ||
        msg.includes('fetch failed') ||
        msg.includes('connect')
      ) {
        throw new Error(`daemon is not running. (${this._baseUrl})`);
      }
      throw err;
    }

    if (!response.ok) {
      throw new Error(`[http_client] HTTP ${response.status}: ${url}`);
    }

    return response.json() as Promise<Record<string, unknown>>;
  }

  /** POST with query-parameter helper (used by /approve and /cancel). */
  private async _postParams(
    path: string,
    params: URLSearchParams,
    timeoutMs: number,
  ): Promise<Record<string, unknown>> {
    const url = `${this._baseUrl}${path}?${params.toString()}`;
    let response: Response;
    try {
      response = await fetch(url, {
        method: 'POST',
        signal: makeSignal(timeoutMs),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        msg.includes('ECONNREFUSED') ||
        msg.includes('ENOTFOUND') ||
        msg.includes('fetch failed') ||
        msg.includes('connect')
      ) {
        throw new Error(`daemon is not running. (${this._baseUrl})`);
      }
      throw err;
    }

    if (!response.ok) {
      throw new Error(`[http_client] HTTP ${response.status}: ${url}`);
    }

    return response.json() as Promise<Record<string, unknown>>;
  }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
