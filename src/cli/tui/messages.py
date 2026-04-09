"""Custom Textual Message types for SSE event bridging.

These messages are posted from the SSE worker thread to the main Textual
event loop, allowing async widget updates from synchronous httpx streams.
"""

from __future__ import annotations

from typing import Any

from textual.message import Message


class SSENodeUpdate(Message):
    """Agent node status changed (e.g. session_load, agent, tools)."""

    def __init__(self, node: str, status: str) -> None:
        super().__init__()
        self.node = node
        self.status = status


class SSEThinkingDelta(Message):
    """Incremental thinking text from the LLM."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content


class SSEResponseDelta(Message):
    """Incremental response token from the LLM final answer."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content


class SSEToolStart(Message):
    """A tool execution has started."""

    def __init__(self, tool_name: str) -> None:
        super().__init__()
        self.tool_name = tool_name


class SSEToolEnd(Message):
    """A tool execution has completed."""

    def __init__(self, tool_name: str, latency_ms: float = 0) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.latency_ms = latency_ms


class SSEApproval(Message):
    """Server is requesting human approval for a tool execution plan."""

    def __init__(self, request: dict[str, Any], thread_id: str) -> None:
        super().__init__()
        self.request = request
        self.thread_id = thread_id


class SSEComplete(Message):
    """SSE stream finished. Carries the final result payload."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__()
        self.result = result


class SSEError(Message):
    """An error occurred during SSE streaming."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class ApprovalResult(Message):
    """User responded to an approval request."""

    def __init__(self, approved: bool, thread_id: str) -> None:
        super().__init__()
        self.approved = approved
        self.thread_id = thread_id


class QuerySubmitted(Message):
    """User submitted a query from the input bar."""

    def __init__(self, query: str) -> None:
        super().__init__()
        self.query = query
