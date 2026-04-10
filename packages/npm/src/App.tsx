/**
 * App.tsx — main application component.
 *
 * Port of src/cli/tui/app.py GovOnApp class.
 * Composes all UI widgets into a full-screen Ink layout and wires state
 * management, server polling, SSE streaming, and input history together.
 *
 * Layout (top → bottom):
 *   1. Banner           (welcome screen, rendered once into scrollback)
 *   2. Message history  (Ink <Static> — completed messages go to scrollback)
 *   3. Streaming area   (current thinking + tools + streaming content)
 *   4. Approval prompt  (shown when pendingApproval is set)
 *   5. Separator line
 *   6. InputBar
 *   7. Status footer
 */

import React, { useReducer, useCallback, useMemo, useEffect } from 'react';
import { Box, Text, Static, useApp, useInput, useStdout } from 'ink';
import { createClient, isMockMode } from './clientFactory.js';
import { getBaseUrl, THEME_COLORS } from './config.js';
import type { AppState, Action, Message } from './types.js';
import { useDaemon } from './hooks/useDaemon.js';
import { useSSE } from './hooks/useSSE.js';
import { useHistory } from './hooks/useHistory.js';
import { Banner } from './components/Banner.js';
import { InputBar } from './components/InputBar.js';
import { Spinner } from './components/Spinner.js';
import { ThinkingBlock } from './components/ThinkingBlock.js';
import { MarkdownView } from './components/MarkdownView.js';
import { MessageBubble } from './components/MessageBubble.js';
import { ToolPanel } from './components/ToolPanel.js';
import { ApprovalPrompt } from './components/ApprovalPrompt.js';
import { MetadataBar } from './components/MetadataBar.js';

// ---------------------------------------------------------------------------
// Module-level constants
// ---------------------------------------------------------------------------

/** Sentinel object placed first in the Static items array to render the banner
 *  exactly once into the terminal scrollback. Hoisted to module level so that
 *  its reference is stable across renders and Static never re-prints it. */
const BANNER_ITEM = { _banner: true as const };

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface AppProps {
  version: string;
  initialQuery?: string;
}

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: AppState = {
  messages: [],
  isLoading: false,
  streamingContent: '',
  streamingThinking: '',
  pendingThinking: [],
  activeTools: [],
  sessionId: null,
  threadId: null,
  pendingApproval: null,
  statusLabel: null,
  error: null,
  theme: 'dark',
  apiBase: getBaseUrl(),
  apiVersion: 'v3',
};

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };

    case 'ADD_USER_MESSAGE': {
      const msg: Message = {
        ...action.payload,
        role: 'user',
        content: action.payload.content,
      };
      return { ...state, messages: [...state.messages, msg] };
    }

    case 'START_ASSISTANT_MESSAGE': {
      const msg: Message = {
        id: action.payload.id,
        role: 'assistant',
        content: '',
        timestamp: action.payload.timestamp,
        streaming: true,
      };
      return {
        ...state,
        messages: [...state.messages, msg],
        streamingContent: '',
        streamingThinking: '',
        pendingThinking: [],
        activeTools: [],
      };
    }

    case 'APPEND_STREAMING_CONTENT':
      return {
        ...state,
        streamingContent: state.streamingContent + action.payload,
      };

    case 'APPEND_STREAMING_THINKING':
      return {
        ...state,
        streamingThinking: state.streamingThinking + action.payload,
      };

    case 'START_THINKING_STEP': {
      const { iteration } = action.payload;
      // Only add if we don't already have an entry for this iteration
      const already = state.pendingThinking.some((s) => s.iteration === iteration);
      if (already) return state;
      // Initialize with empty content — the stale streamingThinking buffer
      // must not be snapshotted here because END_THINKING_STEP will append
      // whatever the stream emits between start and end into this slot.
      return {
        ...state,
        pendingThinking: [
          ...state.pendingThinking,
          { iteration, content: '' },
        ],
        streamingThinking: '',
      };
    }

    case 'END_THINKING_STEP': {
      const { iteration, tool_calls } = action.payload;
      const updated = state.pendingThinking.map((step) =>
        step.iteration === iteration
          ? { ...step, content: step.content + state.streamingThinking, tool_calls }
          : step,
      );
      return {
        ...state,
        pendingThinking: updated,
        streamingThinking: '',
      };
    }

    case 'MARK_TOOL_START':
      return {
        ...state,
        activeTools: [
          ...state.activeTools,
          { tool: action.payload.tool, pending: true },
        ],
      };

    case 'MARK_TOOL_END': {
      const { tool, success } = action.payload;
      // Use findIndex + targeted update so only the first pending entry for
      // this tool name is mutated — prevents clobbering later runs of the
      // same tool if it appears more than once in a single iteration.
      const idx = state.activeTools.findIndex((t) => t.tool === tool && t.pending);
      if (idx === -1) return state;
      const updated = [...state.activeTools];
      updated[idx] = { ...updated[idx], pending: false, success };
      return { ...state, activeTools: updated };
    }

    case 'FINALIZE_ASSISTANT_MESSAGE': {
      const {
        messageId,
        content,
        evidence,
        metadata,
        sessionId,
        threadId,
      } = action.payload;

      const messages = state.messages.map((msg) =>
        msg.id === messageId
          ? {
              ...msg,
              content,
              streaming: false,
              thinking: state.pendingThinking,
              tools: state.activeTools,
              evidence,
              metadata,
            }
          : msg,
      );

      return {
        ...state,
        messages,
        streamingContent: '',
        streamingThinking: '',
        pendingThinking: [],
        activeTools: [],
        sessionId,
        threadId,
        isLoading: false,
        statusLabel: null,
        error: null,
      };
    }

    case 'SET_PENDING_APPROVAL':
      return { ...state, pendingApproval: action.payload };

    case 'SET_STATUS_LABEL':
      return { ...state, statusLabel: action.payload };

    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };

    case 'SET_THEME':
      return { ...state, theme: action.payload };

    case 'SET_API_BASE':
      return { ...state, apiBase: action.payload };

    case 'SET_API_VERSION':
      return { ...state, apiVersion: action.payload };

    case 'RESET_SESSION':
      return { ...initialState, apiBase: state.apiBase, theme: state.theme };

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// App component
// ---------------------------------------------------------------------------

export function App({ version, initialQuery }: AppProps) {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const [state, dispatch] = useReducer(reducer, initialState);

  // Clear terminal on resize to prevent ghost lines from Ink's line-count
  // mismatch. Ink erases only previousLineCount lines, but after a width
  // change text reflows and the line count shifts — leaving stale rows.
  useEffect(() => {
    if (!stdout) return;
    const onResize = () => {
      stdout.write('\x1b[2J\x1b[H');
    };
    stdout.on('resize', onResize);
    return () => { stdout.off('resize', onResize); };
  }, [stdout]);

  // Client is stable for the lifetime of the app (mock or real based on GOVON_MOCK)
  const client = useMemo(() => createClient(state.apiBase), [state.apiBase]);

  // Wait for server readiness
  const daemon = useDaemon(client);

  // SSE streaming
  const { submit, cancel } = useSSE({
    client,
    dispatch,
    sessionId: state.sessionId,
  });

  // Input history (up/down arrow navigation)
  const { push: pushHistory } = useHistory();

  // Handle query submission from InputBar
  const handleSubmit = useCallback(
    (query: string) => {
      // Guard: do not allow submission while a request is in flight
      if (state.isLoading) return;

      const userMsgId = crypto.randomUUID();
      const timestamp = new Date().toISOString();

      dispatch({
        type: 'ADD_USER_MESSAGE',
        payload: { id: userMsgId, content: query, timestamp },
      });

      pushHistory(query);
      submit(query);
    },
    [state.isLoading, dispatch, pushHistory, submit],
  );

  // Approval handling
  const handleApproval = useCallback(
    async (approved: boolean) => {
      if (!state.threadId || !state.pendingApproval) return;
      dispatch({ type: 'SET_PENDING_APPROVAL', payload: null });
      try {
        const result = await client.approve(state.threadId, approved);
        if (approved && result.text) {
          // The approve endpoint returns the final answer when the agent
          // completes after approval. Finalize the current assistant message
          // with that result rather than leaving streaming dangling.
          const streamingMsg = state.messages.find((m) => m.streaming);
          if (streamingMsg) {
            dispatch({
              type: 'FINALIZE_ASSISTANT_MESSAGE',
              payload: {
                messageId: streamingMsg.id,
                content: result.text,
                evidence: result.evidence_items ?? [],
                metadata: {},
                sessionId: result.session_id,
                threadId: result.thread_id,
              },
            });
          }
        }
        dispatch({ type: 'SET_LOADING', payload: false });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        dispatch({ type: 'SET_ERROR', payload: msg });
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    },
    [client, state.threadId, state.pendingApproval, state.messages, dispatch],
  );

  // Submit initial query on first render once daemon is ready
  const initialSubmittedRef = React.useRef(false);
  React.useEffect(() => {
    if (daemon.ready && initialQuery && !initialSubmittedRef.current) {
      initialSubmittedRef.current = true;
      handleSubmit(initialQuery);
    }
  }, [daemon.ready, initialQuery, handleSubmit]);

  // Keyboard shortcuts: Ctrl+D/Ctrl+C exits, Esc cancels in-flight request
  useInput((input: string, key: { ctrl: boolean; escape: boolean }) => {
    if (key.ctrl && input === 'd') exit();
    if (key.ctrl && input === 'c') exit();
    if (key.escape && state.isLoading) cancel();
  });

  // ALL hooks must be called ABOVE this line — React requires the same
  // number of hooks on every render. Early returns go BELOW.

  // Completed messages (go to terminal scrollback via <Static>)
  const completedMessages = state.messages.filter((m) => !m.streaming);

  // Memoize the items array so that the reference only changes when
  // completedMessages changes. Using a stable module-level BANNER_ITEM object
  // prevents Static from detecting a new array on every render and
  // re-printing the banner into the scrollback.
  const staticItems = useMemo(
    () => [BANNER_ITEM, ...completedMessages],
    [completedMessages],
  );


  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  // Server not yet ready
  if (daemon.waiting) {
    return (
      <Box flexDirection="column">
        <Banner version={version} />
        <Box marginTop={1}>
          <Text color={THEME_COLORS.muted}>{'서버 연결 중…'}</Text>
        </Box>
      </Box>
    );
  }

  if (daemon.error) {
    return (
      <Box flexDirection="column">
        <Banner version={version} />
        <Box marginTop={1}>
          <Text color={THEME_COLORS.error}>{'서버 오류: '}{daemon.error}</Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      {/* ── 1. Banner + completed messages in scrollback ── */}
      <Static items={staticItems}>
        {(item: { _banner?: boolean } | Message) => {
          if ('_banner' in item && item._banner) {
            return <Banner key="__banner__" version={version} />;
          }
          const msg = item as Message;
          return <MessageBubble key={msg.id} message={msg} />;
        }}
      </Static>

      {/* ── 2. Streaming area (live, below scrollback) ── */}
      {state.isLoading && (
        <Box flexDirection="column" marginTop={1}>
          {/* Thinking in progress */}
          {state.streamingThinking.length > 0 && (
            <ThinkingBlock content={state.streamingThinking} streaming />
          )}

          {/* Active tool invocations */}
          {state.activeTools.length > 0 && (
            <ToolPanel tools={state.activeTools} />
          )}

          {/* Streaming response text */}
          {state.streamingContent.length > 0 ? (
            <Box marginLeft={2}>
              <MarkdownView content={state.streamingContent} streaming />
            </Box>
          ) : (
            <Box marginLeft={2}>
              <Spinner />
            </Box>
          )}

          {/* Status label from v2 node events */}
          {state.statusLabel && (
            <Box marginLeft={2}>
              <Text color={THEME_COLORS.muted} dimColor>
                {state.statusLabel}
              </Text>
            </Box>
          )}
        </Box>
      )}

      {/* ── 3. Error display ── */}
      {state.error && !state.isLoading && (
        <Box marginTop={1} marginLeft={2}>
          <Text color={THEME_COLORS.error}>{'오류: '}{state.error}</Text>
        </Box>
      )}

      {/* ── 4. Approval prompt ── */}
      {state.pendingApproval && (
        <ApprovalPrompt
          request={state.pendingApproval}
          onApprove={handleApproval}
        />
      )}

      {/* ── 5. Separator ── */}
      <Box marginTop={1}>
        <Text color={THEME_COLORS.muted} dimColor>{'───'}</Text>
      </Box>

      {/* ── 6. Input bar — unmounted during approval to avoid useInput conflicts ── */}
      {!state.pendingApproval && (
        <InputBar onSubmit={handleSubmit} disabled={state.isLoading} />
      )}

      {/* ── 7. Status footer ── */}
      <Box marginTop={1}>
        <Text dimColor color={THEME_COLORS.dimmed}>
          {isMockMode ? '[MOCK] esc 취소 · Ctrl+D 종료 · /help 도움말' : 'esc 취소 · Ctrl+D 종료 · /help 도움말'}
        </Text>
      </Box>
    </Box>
  );
}
