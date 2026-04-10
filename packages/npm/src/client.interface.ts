/**
 * IClient — shared interface for the GovOn daemon HTTP client.
 *
 * Both GovOnClient (real) and MockGovOnClient (offline dev) implement this
 * contract so that hooks and components remain agnostic of the backend.
 */

import type {
  V3SSEEvent,
  V2SSEEvent,
  AgentRunResponse,
  ApprovalResponse,
} from './types.js';

export interface IClient {
  health(): Promise<Record<string, unknown>>;
  waitForReady(): Promise<boolean>;

  streamV3(
    query: string,
    sessionId?: string,
    maxIterations?: number,
    signal?: AbortSignal,
  ): AsyncGenerator<V3SSEEvent>;

  stream(
    query: string,
    sessionId?: string,
    signal?: AbortSignal,
  ): AsyncGenerator<V2SSEEvent>;

  run(
    query: string,
    sessionId?: string,
    signal?: AbortSignal,
  ): Promise<AgentRunResponse>;

  runV3(
    query: string,
    sessionId?: string,
    maxIterations?: number,
    signal?: AbortSignal,
  ): Promise<Record<string, unknown>>;

  approve(threadId: string, approved: boolean): Promise<ApprovalResponse>;

  cancel(threadId: string): Promise<Record<string, unknown>>;
}
