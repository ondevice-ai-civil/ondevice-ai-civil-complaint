/**
 * useSSE — React hook managing the SSE streaming lifecycle.
 *
 * Mirrors the fallback chain in src/cli/shell.py _try_process_query (lines 164-202):
 *   1. Try v3 fine-grained streaming  (client.streamV3)
 *   2. On failure → try v2 node-level streaming  (client.stream)
 *   3. On failure → try blocking run  (client.run)
 *
 * Cancellation is handled via an AbortController stored in a ref.
 * The controller.signal is forwarded to every client call so that aborting the
 * controller immediately cancels the underlying fetch (no zombie requests).
 *
 * An unmount guard (mountedRef) prevents dispatching into unmounted components.
 */

import { useCallback, useEffect, useRef } from 'react';
import type { IClient } from '../client.interface.js';
import type { V3SSEEvent, V2SSEEvent, Action } from '../types.js';
import { NODE_STATUS_MESSAGES } from '../types.js';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface UseSSEOptions {
  client: IClient;
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

  /**
   * Unmount guard — set to false when the component unmounts.
   * All async callbacks check this before dispatching to avoid state updates
   * on unmounted components (which would leak memory and produce React warnings).
   */
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  /**
   * Safe dispatch — drops the action if the component has already unmounted.
   * All internal helpers receive this wrapper instead of the raw dispatch.
   */
  const safeDispatch = useCallback(
    (action: Action) => {
      if (mountedRef.current) dispatch(action);
    },
    [dispatch],
  );

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

      // Guard against stale cleanup: if a newer submit() starts while this
      // one is still unwinding, we must not null out the newer controller.
      const isCurrentRun = () => abortRef.current === controller;

      // Stable message ID for the assistant slot opened before streaming begins.
      const assistantMsgId = crypto.randomUUID();
      const timestamp = new Date().toISOString();
      const sid = sessionId ?? undefined;

      safeDispatch({ type: 'SET_LOADING', payload: true });
      safeDispatch({ type: 'SET_ERROR', payload: null });
      safeDispatch({
        type: 'START_ASSISTANT_MESSAGE',
        payload: { id: assistantMsgId, timestamp },
      });

      // ------------------------------------------------------------------
      // Path 1: v3 fine-grained streaming
      // ------------------------------------------------------------------
      const { success: v3Success, contentDispatched: v3ContentDispatched } = await _tryV3(
        client,
        query,
        sid,
        assistantMsgId,
        controller,
        safeDispatch,
      );

      if (v3Success) {
        safeDispatch({ type: 'SET_API_VERSION', payload: 'v3' });
        safeDispatch({ type: 'SET_LOADING', payload: false });
        if (isCurrentRun()) abortRef.current = null;
        return;
      }

      // Bail out immediately if the user cancelled.
      if (controller.signal.aborted) {
        safeDispatch({ type: 'SET_LOADING', payload: false });
        if (isCurrentRun()) abortRef.current = null;
        return;
      }

      // If v3 emitted content tokens before failing, the message buffer is
      // already partially written. Falling through to v2 would corrupt it by
      // appending v2 content on top of the partial v3 content. Treat this as
      // a terminal error instead.
      if (v3ContentDispatched) {
        safeDispatch({
          type: 'SET_ERROR',
          payload: 'Streaming interrupted after partial response — please retry.',
        });
        safeDispatch({ type: 'SET_LOADING', payload: false });
        if (isCurrentRun()) abortRef.current = null;
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
        safeDispatch,
      );

      if (v2Success) {
        safeDispatch({ type: 'SET_API_VERSION', payload: 'v2' });
        safeDispatch({ type: 'SET_LOADING', payload: false });
        if (isCurrentRun()) abortRef.current = null;
        return;
      }

      if (controller.signal.aborted) {
        safeDispatch({ type: 'SET_LOADING', payload: false });
        if (isCurrentRun()) abortRef.current = null;
        return;
      }

      // ------------------------------------------------------------------
      // Path 3: blocking run
      // ------------------------------------------------------------------
      await _tryBlocking(client, query, sid, assistantMsgId, controller, safeDispatch);

      safeDispatch({ type: 'SET_LOADING', payload: false });
      if (isCurrentRun()) abortRef.current = null;
    },
    [client, safeDispatch, sessionId, cancel],
  );

  return { submit, cancel };
}

// ---------------------------------------------------------------------------
// Internal path helpers  (module-level to keep the hook body lean)
// ---------------------------------------------------------------------------

/** Result shape returned by _tryV3 so callers can inspect partial-write state. */
interface V3Result {
  /** true when run_complete was received cleanly. */
  success: boolean;
  /**
   * true when at least one response_delta or thinking_delta was dispatched
   * before the function returned. Used to decide whether to fall through or
   * surface an error when v3 fails mid-stream.
   */
  contentDispatched: boolean;
}

/**
 * Attempt v3 SSE streaming. Returns a V3Result describing the outcome.
 *
 * Returns success=true only on a clean run_complete event.
 * Returns success=false, contentDispatched=false on network/HTTP exceptions
 *   (safe to fall through to v2).
 * Returns success=false, contentDispatched=false when an explicit `error`
 *   event is received — the error is already dispatched and the caller
 *   should NOT fall through (the event is a terminal server-side error).
 *   We signal this by returning success=false but we still set contentDispatched
 *   to true so the caller surfaces the error rather than trying v2.
 */
async function _tryV3(
  client: IClient,
  query: string,
  sessionId: string | undefined,
  assistantMsgId: string,
  controller: AbortController,
  dispatch: React.Dispatch<Action>,
): Promise<V3Result> {
  let contentDispatched = false;

  try {
    for await (const raw of client.streamV3(query, sessionId, undefined, controller.signal)) {
      // Respect cancellation mid-stream.
      if (controller.signal.aborted) return { success: false, contentDispatched };

      const event = raw as V3SSEEvent;

      switch (event.type) {
        case 'thinking_start':
          dispatch({
            type: 'START_THINKING_STEP',
            payload: { iteration: event.iteration },
          });
          break;

        case 'thinking_delta':
          contentDispatched = true;
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
          contentDispatched = true;
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
          return { success: true, contentDispatched };

        case 'error':
          // An explicit server-side error event is a terminal state.
          // Dispatch the error and signal the caller NOT to fall through by
          // returning contentDispatched=true (treated as "buffer tainted").
          dispatch({ type: 'SET_ERROR', payload: event.error });
          return { success: false, contentDispatched: true };

        default:
          // Unknown event type — ignore and continue streaming.
          break;
      }
    }

    // Generator exhausted without a run_complete event — treat as failure.
    // contentDispatched reflects whether any tokens were written.
    return { success: false, contentDispatched };
  } catch (_err) {
    // Network error, HTTP non-2xx, abort, or JSON parse failure from the client.
    return { success: false, contentDispatched };
  }
}

/**
 * Attempt v2 node-level SSE streaming. Returns true on clean completion,
 * false on error so the caller can fall through to the blocking path.
 */
async function _tryV2(
  client: IClient,
  query: string,
  sessionId: string | undefined,
  assistantMsgId: string,
  controller: AbortController,
  dispatch: React.Dispatch<Action>,
): Promise<boolean> {
  try {
    for await (const raw of client.stream(query, sessionId, controller.signal)) {
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
  } catch (_err) {
    // Network/HTTP error or abort — try next path.
    return false;
  }
}

/**
 * Attempt the blocking /v2/agent/run endpoint as a last resort.
 * Dispatches FINALIZE on success, SET_ERROR on failure.
 */
async function _tryBlocking(
  client: IClient,
  query: string,
  sessionId: string | undefined,
  assistantMsgId: string,
  controller: AbortController,
  dispatch: React.Dispatch<Action>,
): Promise<void> {
  if (controller.signal.aborted) return;

  try {
    const result = await client.run(query, sessionId, controller.signal);

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
    dispatch({ type: 'SET_API_VERSION', payload: 'v2' });
  } catch (err) {
    if (controller.signal.aborted) return;

    const msg = err instanceof Error ? err.message : String(err);
    dispatch({ type: 'SET_ERROR', payload: msg });
  }
}
