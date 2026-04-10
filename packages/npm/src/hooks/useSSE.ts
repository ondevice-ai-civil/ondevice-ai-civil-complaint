/**
 * useSSE — React hook managing the SSE streaming lifecycle.
 *
 * Mirrors the fallback chain in src/cli/shell.py _try_process_query (lines 164-202):
 *   1. Try v3 fine-grained streaming  (client.streamV3)
 *   2. On failure → try v2 node-level streaming  (client.stream)
 *   3. On failure → try blocking run  (client.run)
 *
 * Cancellation is handled via an AbortController stored in a ref.
 * The controller is shared with the client by aborting the underlying fetch
 * through the stream generator — on abort the for-await loop throws and
 * the catch block in each path swallows the error and returns early.
 */

import { useCallback, useRef } from 'react';
import type { GovOnClient } from '../client.js';
import type { V3SSEEvent, V2SSEEvent, Action } from '../types.js';
import { NODE_STATUS_MESSAGES } from '../types.js';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface UseSSEOptions {
  client: GovOnClient;
  dispatch: React.Dispatch<Action>;
  sessionId: string | null;
}

export interface UseSSEReturn {
  /** Submit a user query. Resolves when the run finishes (success or error). */
  submit: (query: string) => Promise<void>;
  /** Abort the in-flight request immediately. */
  cancel: () => void;
}

// ---------------------------------------------------------------------------
// Hook implementation
// ---------------------------------------------------------------------------

export function useSSE({ client, dispatch, sessionId }: UseSSEOptions): UseSSEReturn {
  /** Holds the AbortController for the current in-flight request. */
  const abortRef = useRef<AbortController | null>(null);

  // -------------------------------------------------------------------------
  // cancel — abort the current streaming request
  // -------------------------------------------------------------------------

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  // -------------------------------------------------------------------------
  // submit — attempt v3 → v2 → blocking, dispatch actions along the way
  // -------------------------------------------------------------------------

  const submit = useCallback(
    async (query: string): Promise<void> => {
      // Cancel any previous in-flight request before starting a new one.
      cancel();

      const controller = new AbortController();
      abortRef.current = controller;

      // Stable message ID for the assistant slot opened before streaming begins.
      const assistantMsgId = crypto.randomUUID();
      const timestamp = new Date().toISOString();
      const sid = sessionId ?? undefined;

      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'SET_ERROR', payload: null });
      dispatch({
        type: 'START_ASSISTANT_MESSAGE',
        payload: { id: assistantMsgId, timestamp },
      });

      // ------------------------------------------------------------------
      // Path 1: v3 fine-grained streaming
      // ------------------------------------------------------------------
      const v3Success = await _tryV3(
        client,
        query,
        sid,
        assistantMsgId,
        controller,
        dispatch,
      );

      if (v3Success) {
        dispatch({ type: 'SET_API_VERSION', payload: 'v3' });
        dispatch({ type: 'SET_LOADING', payload: false });
        abortRef.current = null;
        return;
      }

      // Bail out immediately if the user cancelled.
      if (controller.signal.aborted) {
        dispatch({ type: 'SET_LOADING', payload: false });
        abortRef.current = null;
        return;
      }

      // ------------------------------------------------------------------
      // Path 2: v2 node-level streaming
      // ------------------------------------------------------------------
      const v2Success = await _tryV2(
        client,
        query,
        sid,
        assistantMsgId,
        controller,
        dispatch,
      );

      if (v2Success) {
        dispatch({ type: 'SET_API_VERSION', payload: 'v2' });
        dispatch({ type: 'SET_LOADING', payload: false });
        abortRef.current = null;
        return;
      }

      if (controller.signal.aborted) {
        dispatch({ type: 'SET_LOADING', payload: false });
        abortRef.current = null;
        return;
      }

      // ------------------------------------------------------------------
      // Path 3: blocking run
      // ------------------------------------------------------------------
      await _tryBlocking(client, query, sid, assistantMsgId, controller, dispatch);

      dispatch({ type: 'SET_LOADING', payload: false });
      abortRef.current = null;
    },
    [client, dispatch, sessionId, cancel],
  );

  return { submit, cancel };
}

// ---------------------------------------------------------------------------
// Internal path helpers  (module-level to keep the hook body lean)
// ---------------------------------------------------------------------------

/**
 * Attempt v3 SSE streaming. Returns true when the run completed successfully,
 * false when any error occurs so the caller can fall through to the next path.
 */
async function _tryV3(
  client: GovOnClient,
  query: string,
  sessionId: string | undefined,
  assistantMsgId: string,
  controller: AbortController,
  dispatch: React.Dispatch<Action>,
): Promise<boolean> {
  try {
    for await (const raw of client.streamV3(query, sessionId)) {
      // Respect cancellation mid-stream.
      if (controller.signal.aborted) return false;

      const event = raw as V3SSEEvent;

      switch (event.type) {
        case 'thinking_start':
          dispatch({
            type: 'START_THINKING_STEP',
            payload: { iteration: event.iteration },
          });
          break;

        case 'thinking_delta':
          dispatch({
            type: 'APPEND_STREAMING_THINKING',
            payload: event.content,
          });
          break;

        case 'thinking_end':
          dispatch({
            type: 'END_THINKING_STEP',
            payload: { iteration: event.iteration, tool_calls: event.tool_calls },
          });
          break;

        case 'tool_start':
          dispatch({
            type: 'MARK_TOOL_START',
            payload: { tool: event.tool },
          });
          break;

        case 'tool_end':
          dispatch({
            type: 'MARK_TOOL_END',
            payload: { tool: event.tool, success: event.success },
          });
          break;

        case 'response_delta':
          dispatch({
            type: 'APPEND_STREAMING_CONTENT',
            payload: event.content,
          });
          break;

        case 'run_complete':
          dispatch({
            type: 'FINALIZE_ASSISTANT_MESSAGE',
            payload: {
              messageId: assistantMsgId,
              content: event.text,
              evidence: event.evidence_items,
              metadata: event.metadata,
              sessionId: event.session_id,
              threadId: event.thread_id,
            },
          });
          // run_complete signals a clean finish — return success immediately.
          return true;

        case 'error':
          dispatch({ type: 'SET_ERROR', payload: event.error });
          // An explicit error event from v3 is still a "handled" terminal state;
          // return false so the caller falls through to v2/blocking.
          return false;

        default:
          // Unknown event type — ignore and continue streaming.
          break;
      }
    }

    // Generator exhausted without a run_complete event — treat as failure.
    return false;
  } catch {
    // Network error, HTTP non-2xx, abort, or JSON parse failure from the client.
    return false;
  }
}

/**
 * Attempt v2 node-level SSE streaming. Returns true on clean completion,
 * false on error so the caller can fall through to the blocking path.
 */
async function _tryV2(
  client: GovOnClient,
  query: string,
  sessionId: string | undefined,
  assistantMsgId: string,
  controller: AbortController,
  dispatch: React.Dispatch<Action>,
): Promise<boolean> {
  try {
    for await (const raw of client.stream(query, sessionId)) {
      if (controller.signal.aborted) return false;

      const event = raw as V2SSEEvent;
      const { node, status } = event;

      // Update the status bar label for recognised node transitions.
      const statusMsg = NODE_STATUS_MESSAGES[node];
      if (statusMsg !== undefined) {
        dispatch({ type: 'SET_STATUS_LABEL', payload: statusMsg });
      }

      if (node === 'agent' && status === 'completed') {
        if (event.planned_tools && event.planned_tools.length > 0) {
          // Agent has decided which tools to call next.
          dispatch({
            type: 'SET_STATUS_LABEL',
            payload: `도구 실행 예정: ${event.planned_tools.join(', ')}`,
          });
        }

        // When final_text is present, the agent run finished on this node.
        if (event.final_text !== undefined) {
          dispatch({
            type: 'FINALIZE_ASSISTANT_MESSAGE',
            payload: {
              messageId: assistantMsgId,
              content: event.final_text,
              evidence: event.evidence_items ?? [],
              metadata: {},
              sessionId: event.session_id ?? '',
              threadId: event.thread_id ?? '',
            },
          });
          return true;
        }
      }

      if (node === 'tools' && status === 'completed') {
        // Tool node finished — update status label; individual tool tracking
        // is not available at v2 granularity so we only set the label.
        dispatch({ type: 'SET_STATUS_LABEL', payload: '도구 실행 완료' });
      }

      if (node === 'approval_wait' && status === 'awaiting_approval') {
        if (event.approval_request !== undefined) {
          dispatch({
            type: 'SET_PENDING_APPROVAL',
            payload: event.approval_request,
          });
        }
      }

      if (node === 'persist' && status === 'completed') {
        if (event.final_text !== undefined) {
          dispatch({
            type: 'FINALIZE_ASSISTANT_MESSAGE',
            payload: {
              messageId: assistantMsgId,
              content: event.final_text,
              evidence: event.evidence_items ?? [],
              metadata: {},
              sessionId: event.session_id ?? '',
              threadId: event.thread_id ?? '',
            },
          });
          return true;
        }
      }

      if (node === 'error') {
        dispatch({
          type: 'SET_ERROR',
          payload: event.error ?? 'Unknown error from v2 stream',
        });
        return false;
      }
    }

    // Generator exhausted without a persist/agent run_complete event.
    return false;
  } catch {
    return false;
  }
}

/**
 * Attempt the blocking /v2/agent/run endpoint as a last resort.
 * Dispatches FINALIZE on success, SET_ERROR on failure.
 */
async function _tryBlocking(
  client: GovOnClient,
  query: string,
  sessionId: string | undefined,
  assistantMsgId: string,
  controller: AbortController,
  dispatch: React.Dispatch<Action>,
): Promise<void> {
  if (controller.signal.aborted) return;

  try {
    const result = await client.run(query, sessionId);

    if (controller.signal.aborted) return;

    dispatch({
      type: 'FINALIZE_ASSISTANT_MESSAGE',
      payload: {
        messageId: assistantMsgId,
        content: result.text,
        evidence: result.evidence_items ?? [],
        metadata: result.metadata ?? {},
        sessionId: result.session_id,
        threadId: result.thread_id ?? '',
      },
    });
  } catch (err) {
    if (controller.signal.aborted) return;

    const msg = err instanceof Error ? err.message : String(err);
    dispatch({ type: 'SET_ERROR', payload: msg });
  }
}
