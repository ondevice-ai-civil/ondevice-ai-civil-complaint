/**
 * types.ts — All shared TypeScript types for the GovOn npm TUI.
 *
 * Covers:
 *  - V3 SSE event discriminated union (fine-grained streaming)
 *  - V2 SSE event interface (coarse node-level streaming)
 *  - Chat message model (user / assistant with thinking & tools)
 *  - ApprovalRequest / ApprovalResponse
 *  - EvidenceItem / RunMetadata
 *  - AgentRunRequest / AgentRunResponse / AgentTrace / ToolResult
 *  - Theme
 *  - AppState + Action discriminated union for the reducer
 *  - Exported const maps: NODE_STATUS_MESSAGES, TASK_TYPE_LABELS,
 *    TASK_TYPE_STYLES, TOOL_DISPLAY_NAMES
 */

// ---------------------------------------------------------------------------
// 1. Shared sub-types
// ---------------------------------------------------------------------------

/** A single piece of evidence returned by the agent after a run. */
export interface EvidenceItem {
  /** Origin of the evidence. */
  source_type: "rag" | "api" | "llm_generated";
  /** Human-readable content excerpt or summary. */
  content: string;
  /** Relevance / confidence score in [0, 1]. */
  score?: number;
  /** Arbitrary extra fields from the backend (e.g. chunk_id, url). */
  metadata?: Record<string, unknown>;
}

/** Metadata attached to a completed agent run (v3 run_complete event). */
export interface RunMetadata {
  total_iterations?: number;
  total_tool_calls?: number;
  total_latency_ms?: number;
  node_latencies?: Record<string, number>;
}

/** A single tool call record embedded in a thinking_end event. */
export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// 2. V3 SSE event discriminated union  (POST /v3/agent/stream)
// ---------------------------------------------------------------------------

export interface V3ThinkingStartEvent {
  type: "thinking_start";
  /** ReAct loop iteration number (1-based). */
  iteration: number;
}

export interface V3ThinkingDeltaEvent {
  type: "thinking_delta";
  /** Partial LLM token chunk emitted during tool-call reasoning. */
  content: string;
}

export interface V3ThinkingEndEvent {
  type: "thinking_end";
  /** Tool calls the agent decided to make after this reasoning step. */
  tool_calls: ToolCall[];
  /** ReAct loop iteration number (1-based). */
  iteration: number;
}

export interface V3ToolStartEvent {
  type: "tool_start";
  /** Name of the tool being invoked. */
  tool: string;
}

export interface V3ToolEndEvent {
  type: "tool_end";
  /** Name of the tool that finished. */
  tool: string;
  /** Whether the tool call succeeded. */
  success?: boolean;
}

export interface V3ResponseDeltaEvent {
  type: "response_delta";
  /** Partial final-answer token chunk (no tool_calls in flight). */
  content: string;
}

export interface V3RunCompleteEvent {
  type: "run_complete";
  thread_id: string;
  session_id: string;
  /** Full final answer text. */
  text: string;
  evidence_items: EvidenceItem[];
  metadata: RunMetadata;
}

export interface V3ErrorEvent {
  type: "error";
  error: string;
}

/** Discriminated union of all V3 SSE event shapes. */
export type V3SSEEvent =
  | V3ThinkingStartEvent
  | V3ThinkingDeltaEvent
  | V3ThinkingEndEvent
  | V3ToolStartEvent
  | V3ToolEndEvent
  | V3ResponseDeltaEvent
  | V3RunCompleteEvent
  | V3ErrorEvent;

// ---------------------------------------------------------------------------
// 3. V2 SSE event interface  (POST /v2/agent/stream)
// ---------------------------------------------------------------------------

/** A pending human-approval request embedded in a V2 event. */
export interface ApprovalRequest {
  /** Unique identifier for this approval gate. */
  approval_id: string;
  /** Task type key (see TASK_TYPE_LABELS for human-readable label). */
  task_type: string;
  /** Short description of what the agent wants to do. */
  description?: string;
  /** Tools the agent plans to call if approved. */
  planned_tools?: string[];
  /** Arbitrary structured context the UI may render. */
  context?: Record<string, unknown>;
}

/** Coarse node-level event emitted by the V2 streaming endpoint. */
export interface V2SSEEvent {
  /** Graph node name (e.g. "agent", "tools", "approval_wait"). */
  node: string;
  status: "completed" | "awaiting_approval" | "error" | (string & {});
  /** Full answer text, present when node === "agent" and run is done. */
  final_text?: string;
  evidence_items?: EvidenceItem[];
  /** Task type key for display purposes. */
  task_type?: string;
  approval_request?: ApprovalRequest;
  thread_id?: string;
  session_id?: string;
  error?: string;
  /** Raw tool outputs keyed by tool name. */
  tool_results?: Record<string, unknown>;
  /** Tools the agent plans to call next. */
  planned_tools?: string[];
}

// ---------------------------------------------------------------------------
// 4. Agent request / response types
// ---------------------------------------------------------------------------

/** Request body sent to both /v2/agent/stream and /v3/agent/stream. */
export interface AgentRunRequest {
  /** User query (1–10 000 chars). */
  query: string;
  /** Existing session to continue; omit to start a new session. */
  session_id?: string;
  /** Whether to stream the response via SSE (default: true). */
  stream?: boolean;
  /** Force specific tools to be called. */
  force_tools?: string[];
  /** Max tokens for the LLM response (default 512, max 4096). */
  max_tokens?: number;
  /** Sampling temperature (default 0.7, range 0–2). */
  temperature?: number;
  /** Max ReAct iterations (default 10, range 1–20). */
  max_iterations?: number;
}

/** A single tool invocation record inside an AgentTrace. Matches backend ToolResultSchema. */
export interface ToolResult {
  tool: string;
  success: boolean;
  latency_ms: number;
  data: Record<string, unknown>;
  error?: string;
}

/** Full reasoning trace for a single agent run. Matches backend AgentTraceSchema. */
export interface AgentTrace {
  request_id: string;
  session_id: string;
  plan: string[];
  plan_reason: string;
  tool_results: ToolResult[];
  total_latency_ms: number;
  error?: string;
}

/** Non-streaming response returned by the agent endpoint. */
export interface AgentRunResponse {
  request_id: string;
  session_id: string;
  text: string;
  trace: AgentTrace;
  /** Present in v2 stream approval events, absent in blocking run. */
  thread_id?: string;
  /** Present in v2 blocking run responses. */
  graph_run_id?: string;
  evidence_items?: EvidenceItem[];
  metadata?: RunMetadata;
}

/** Response from POST /v2/agent/approve. */
export interface ApprovalResponse {
  status: "approved" | "rejected" | "error";
  thread_id: string;
  session_id: string;
  graph_run_id?: string;
  text?: string;
  evidence_items?: EvidenceItem[];
  /** "approved" | "rejected" mirroring top-level status. */
  approval_status?: string;
  error?: string;
}

// ---------------------------------------------------------------------------
// 5. Chat message model
// ---------------------------------------------------------------------------

/** A reasoning/thinking step displayed inside an assistant message. */
export interface ThinkingStep {
  /** Which ReAct iteration this step belongs to. */
  iteration: number;
  /** Accumulated thinking text for this step. */
  content: string;
  /** Tool calls decided at the end of this step. */
  tool_calls?: ToolCall[];
}

/** A single tool invocation displayed inside an assistant message. */
export interface ToolInvocation {
  tool: string;
  /** Whether the tool call is still running. */
  pending: boolean;
  success?: boolean;
}

/** A single message in the chat history. */
export interface Message {
  id: string;
  role: "user" | "assistant";
  /** Main text content (may be partial while streaming). */
  content: string;
  /** Thinking steps shown in a collapsible section. */
  thinking?: ThinkingStep[];
  /** Tool invocations shown inline. */
  tools?: ToolInvocation[];
  /** Evidence items shown after the answer. */
  evidence?: EvidenceItem[];
  /** Run metadata shown after the answer. */
  metadata?: RunMetadata;
  /** ISO-8601 timestamp. */
  timestamp: string;
  /** Whether streaming is still in progress. */
  streaming?: boolean;
  /** Error text, if this message represents a failure. */
  error?: string;
}

// ---------------------------------------------------------------------------
// 6. Theme
// ---------------------------------------------------------------------------

export type Theme = "light" | "dark" | "system";

// ---------------------------------------------------------------------------
// 7. AppState and Action discriminated union (reducer)
// ---------------------------------------------------------------------------

/** Global application state managed by the reducer. */
export interface AppState {
  /** All messages in the current session. */
  messages: Message[];
  /** Whether the agent is currently processing a request. */
  isLoading: boolean;
  /** Current streaming text buffer (partial assistant response). */
  streamingContent: string;
  /** Current streaming thinking buffer. */
  streamingThinking: string;
  /** Pending thinking steps in the current run. */
  pendingThinking: ThinkingStep[];
  /** Active tool invocations in the current run. */
  activeTools: ToolInvocation[];
  /** Session ID from the backend (null until first run). */
  sessionId: string | null;
  /** Thread ID from the backend (null until first run). */
  threadId: string | null;
  /** Pending approval request, if any. */
  pendingApproval: ApprovalRequest | null;
  /** Current node/phase label shown in the status bar. */
  statusLabel: string | null;
  /** Last error message to surface in the UI. */
  error: string | null;
  /** Active color theme. */
  theme: Theme;
  /** Base URL of the GovOn backend. */
  apiBase: string;
  /** SSE protocol version in use. */
  apiVersion: "v2" | "v3";
}

// ---------- Actions ---------------------------------------------------------

export interface SetLoadingAction {
  type: "SET_LOADING";
  payload: boolean;
}

export interface AddUserMessageAction {
  type: "ADD_USER_MESSAGE";
  payload: Pick<Message, "id" | "content" | "timestamp">;
}

/** Start a new assistant message slot for streaming. */
export interface StartAssistantMessageAction {
  type: "START_ASSISTANT_MESSAGE";
  payload: Pick<Message, "id" | "timestamp">;
}

export interface AppendStreamingContentAction {
  type: "APPEND_STREAMING_CONTENT";
  payload: string;
}

export interface AppendStreamingThinkingAction {
  type: "APPEND_STREAMING_THINKING";
  payload: string;
}

export interface StartThinkingStepAction {
  type: "START_THINKING_STEP";
  payload: { iteration: number };
}

export interface EndThinkingStepAction {
  type: "END_THINKING_STEP";
  payload: { iteration: number; tool_calls: ToolCall[] };
}

export interface MarkToolStartAction {
  type: "MARK_TOOL_START";
  payload: { tool: string };
}

export interface MarkToolEndAction {
  type: "MARK_TOOL_END";
  payload: { tool: string; success?: boolean };
}

export interface FinalizeAssistantMessageAction {
  type: "FINALIZE_ASSISTANT_MESSAGE";
  payload: {
    messageId: string;
    content: string;
    evidence: EvidenceItem[];
    metadata: RunMetadata;
    sessionId: string;
    threadId: string;
  };
}

export interface SetPendingApprovalAction {
  type: "SET_PENDING_APPROVAL";
  payload: ApprovalRequest | null;
}

export interface SetStatusLabelAction {
  type: "SET_STATUS_LABEL";
  payload: string | null;
}

export interface SetErrorAction {
  type: "SET_ERROR";
  payload: string | null;
}

export interface SetThemeAction {
  type: "SET_THEME";
  payload: Theme;
}

export interface SetApiBaseAction {
  type: "SET_API_BASE";
  payload: string;
}

export interface SetApiVersionAction {
  type: "SET_API_VERSION";
  payload: "v2" | "v3";
}

export interface ResetSessionAction {
  type: "RESET_SESSION";
}

/** Discriminated union of all reducer actions. */
export type Action =
  | SetLoadingAction
  | AddUserMessageAction
  | StartAssistantMessageAction
  | AppendStreamingContentAction
  | AppendStreamingThinkingAction
  | StartThinkingStepAction
  | EndThinkingStepAction
  | MarkToolStartAction
  | MarkToolEndAction
  | FinalizeAssistantMessageAction
  | SetPendingApprovalAction
  | SetStatusLabelAction
  | SetErrorAction
  | SetThemeAction
  | SetApiBaseAction
  | SetApiVersionAction
  | ResetSessionAction;

// ---------------------------------------------------------------------------
// 8. Exported const maps  (sourced from renderer.py and approval_ui.py)
// ---------------------------------------------------------------------------

/**
 * Human-readable status messages for each V2 graph node.
 * Displayed in the status bar while the corresponding node is active.
 */
export const NODE_STATUS_MESSAGES: Readonly<Record<string, string>> = {
  session_load: "세션 로드 중…",
  agent: "에이전트 추론 중…",
  approval_wait: "승인 대기 중…",
  tools: "도구 실행 중…",
  persist: "저장 중…",
} as const;

/**
 * Korean display labels for each task type key.
 * Used in approval dialogs and history views.
 */
export const TASK_TYPE_LABELS: Readonly<Record<string, string>> = {
  domain_adapter: "도메인 어댑터 응답",
  revise_response: "답변 수정",
  lookup_stats: "통계 조회",
  issue_detection: "이슈 탐지",
  stats_query: "통계 조회",
  keyword_analysis: "키워드 분석",
  demographics_query: "인구통계 조회",
  default: "일반 작업",
} as const;

/**
 * Terminal / Ink color name for each task type's badge.
 * Values correspond to Ink's `<Text color="...">` prop.
 */
export const TASK_TYPE_STYLES: Readonly<Record<string, string>> = {
  domain_adapter: "cyan",
  revise_response: "blue",
  lookup_stats: "magenta",
  issue_detection: "yellow",
  stats_query: "magenta",
  keyword_analysis: "yellow",
  demographics_query: "blueBright",
  default: "cyan",
} as const;

/**
 * Human-readable display names for each tool identifier.
 * Used in the tool-call inline UI within assistant messages.
 */
export const TOOL_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  stats_lookup: "민원 통계",
  keyword_analyzer: "키워드 분석",
  demographics_lookup: "인구통계",
} as const;
