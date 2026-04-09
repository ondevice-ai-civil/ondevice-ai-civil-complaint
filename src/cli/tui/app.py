"""GovOn Textual TUI application.

Provides a Claude Code-style terminal interface with scrollable message
history, streaming markdown, collapsible tool panels, and an input bar.
Runs in inline mode so output stays in terminal scrollback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Rule, Static

from src.cli.tui.messages import (
    ApprovalResult,
    SSEApproval,
    SSEComplete,
    SSEError,
    SSEResponseDelta,
    SSEThinkingDelta,
    SSEToolEnd,
    SSEToolStart,
)
from src.cli.tui.widgets.approval_modal import ApprovalModal
from src.cli.tui.widgets.input_bar import InputBar
from src.cli.tui.widgets.markdown_view import MarkdownView
from src.cli.tui.widgets.message_bubble import MessageBubble
from src.cli.tui.widgets.metadata_bar import MetadataBar
from src.cli.tui.widgets.spinner import SpinnerWidget
from src.cli.tui.widgets.thinking_block import ThinkingBlock
from src.cli.tui.widgets.tool_panel import ToolCallPanel
from src.cli.tui.widgets.welcome_screen import WelcomeScreen

if TYPE_CHECKING:
    from src.cli.http_client import GovOnClient


class GovOnApp(App):
    """Main Textual application for GovOn CLI."""

    CSS_PATH = "govon_app.tcss"

    BINDINGS = [
        ("escape", "cancel_query", "\ucde8\uc18c"),
        ("ctrl+d", "quit", "\uc885\ub8cc"),
    ]

    def __init__(
        self,
        client: GovOnClient,
        session_id: str | None = None,
        query: str | None = None,
        version: str = "dev",
        runtime_url: str | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.session_id = session_id
        self._initial_query = query
        self._version = version
        self._runtime_url = runtime_url
        self._current_bubble: MessageBubble | None = None
        self._current_spinner: SpinnerWidget | None = None
        self._current_thinking: ThinkingBlock | None = None
        self._current_markdown: MarkdownView | None = None

    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        with VerticalScroll(id="message-history"):
            yield WelcomeScreen(
                version=self._version,
                runtime_url=self._runtime_url,
            )
        yield Rule(id="separator")
        yield InputBar(id="input-bar")
        yield Footer(id="status-footer")

    def on_mount(self) -> None:
        """Focus the input bar on startup."""
        self.query_one("#input-bar", InputBar).focus()
        if self._initial_query:
            self._submit_query(self._initial_query)

    def on_input_submitted(self, event: InputBar.Submitted) -> None:
        """Handle Enter key in the input bar."""
        query = event.value.strip()
        if not query:
            return
        event.input.clear()
        self._submit_query(query)

    def _submit_query(self, query: str) -> None:
        """Process a user query — route to command handler or SSE worker."""
        from src.cli.commands import handle_command, is_command

        history = self.query_one("#message-history", VerticalScroll)

        if is_command(query):
            try:
                result = handle_command(query)
            except SystemExit:
                self.exit()
                return
            if result is not None:
                history.mount(Static(result))
            return

        # Mount user message bubble
        user_bubble = MessageBubble(role="user")
        history.mount(user_bubble)
        user_bubble.mount(Static(f"[bold]\u276f[/bold] {query}"))

        # Mount assistant bubble with spinner
        self._current_bubble = MessageBubble(role="assistant")
        history.mount(self._current_bubble)
        self._current_spinner = SpinnerWidget()
        self._current_bubble.mount(self._current_spinner)
        self._current_thinking = None
        self._current_markdown = None

        # Start SSE streaming in background thread
        self.run_worker(
            self._consume_sse(query),
            thread=True,
            exclusive=True,
            group="sse",
        )

    def _stop_spinner(self) -> None:
        """Remove the spinner widget when first content arrives."""
        if self._current_spinner is not None:
            self._current_spinner.stop()
            self._current_spinner.remove()
            self._current_spinner = None

    async def _consume_sse(self, query: str) -> None:
        """Worker: consume SSE events from the sync httpx client."""
        try:
            for event in self.client.stream_v3(query, self.session_id):
                event_type = event.get("type", "")

                if event_type == "error":
                    self.call_from_thread(
                        self.post_message, SSEError(event.get("error", "unknown error"))
                    )
                    return

                elif event_type == "thinking_delta":
                    content = event.get("content", "")
                    if content:
                        self.call_from_thread(self.post_message, SSEThinkingDelta(content))

                elif event_type == "response_delta":
                    content = event.get("content", "")
                    if content:
                        self.call_from_thread(self.post_message, SSEResponseDelta(content))

                elif event_type == "tool_start":
                    tool_name = event.get("tool", "")
                    if tool_name:
                        self.call_from_thread(self.post_message, SSEToolStart(tool_name))

                elif event_type == "tool_end":
                    tool_name = event.get("tool", "")
                    latency = event.get("latency_ms", 0)
                    self.call_from_thread(self.post_message, SSEToolEnd(tool_name, latency))

                elif event_type == "awaiting_approval":
                    request = event.get("approval_request") or {}
                    thread_id = event.get("thread_id", "")
                    self.call_from_thread(self.post_message, SSEApproval(request, thread_id))
                    return  # pause SSE consumption until approval resolves

                elif event_type == "run_complete":
                    sid = event.get("session_id") or event.get("thread_id")
                    if sid:
                        self.session_id = sid
                    self.call_from_thread(self.post_message, SSEComplete(event))

        except Exception as exc:
            self.call_from_thread(self.post_message, SSEError(str(exc)))

    # -- SSE event handlers --------------------------------------------------

    def on_sse_thinking_delta(self, message: SSEThinkingDelta) -> None:
        """Append thinking text — spinner disappears on first token."""
        self._stop_spinner()
        if self._current_bubble is None:
            return
        if self._current_thinking is None:
            self._current_thinking = ThinkingBlock()
            self._current_bubble.mount(self._current_thinking)
        self._current_thinking.append(message.content)

    def on_sse_response_delta(self, message: SSEResponseDelta) -> None:
        """Stream response tokens into markdown view."""
        self._stop_spinner()
        if self._current_bubble is None:
            return
        if self._current_markdown is None:
            self._current_markdown = MarkdownView()
            self._current_bubble.mount(self._current_markdown)
        self._current_markdown.append_delta(message.content)

    def on_sse_tool_start(self, message: SSEToolStart) -> None:
        """Mount a tool call panel."""
        self._stop_spinner()
        if self._current_bubble is None:
            return
        panel = ToolCallPanel(tool_name=message.tool_name, id=f"tool-{message.tool_name}")
        self._current_bubble.mount(panel)

    def on_sse_tool_end(self, message: SSEToolEnd) -> None:
        """Update tool call panel to completed state."""
        if self._current_bubble is None:
            return
        try:
            panel = self._current_bubble.query_one(f"#tool-{message.tool_name}", ToolCallPanel)
            panel.complete(message.latency_ms)
        except Exception:
            # Panel not found — render inline fallback
            latency = f" ({message.latency_ms:.0f}ms)" if message.latency_ms else ""
            self._current_bubble.mount(
                Static(
                    f"[green]\u2514\u2500 \u2726 [bold]{message.tool_name}[/bold] \uc644\ub8cc{latency}[/green]"
                )
            )

    def on_sse_error(self, message: SSEError) -> None:
        """Show error in the current bubble or history."""
        self._stop_spinner()
        target = self._current_bubble or self.query_one("#message-history", VerticalScroll)
        target.mount(Static(f"[red bold]\uc624\ub958:[/red bold] {message.error}"))

    def on_sse_complete(self, message: SSEComplete) -> None:
        """Finalize response rendering."""
        self._stop_spinner()

        if self._current_bubble is None:
            return

        # Render final text if not already streamed via response_delta
        text = message.result.get("text") or message.result.get("response") or ""
        if text and self._current_markdown is None:
            md = MarkdownView()
            self._current_bubble.mount(md)
            md.set_content(text)

        # Show metadata
        metadata = message.result.get("metadata", {})
        if metadata:
            bar = MetadataBar()
            self._current_bubble.mount(bar)
            bar.set_metadata(metadata)

        # Scroll to bottom
        history = self.query_one("#message-history", VerticalScroll)
        history.scroll_end()

        # Reset current state for next query
        self._current_bubble = None
        self._current_spinner = None
        self._current_thinking = None
        self._current_markdown = None

    def on_sse_approval(self, message: SSEApproval) -> None:
        """Mount an approval modal for human-in-the-loop tool approval."""
        self._stop_spinner()
        if self._current_bubble is None:
            return
        modal = ApprovalModal(
            approval_request=message.request,
            thread_id=message.thread_id,
        )
        self._current_bubble.mount(modal)

    def on_approval_result(self, message: ApprovalResult) -> None:
        """Handle user approval/rejection and resume SSE if approved."""
        try:
            self.client.approve(message.thread_id, approved=message.approved)
        except Exception:
            pass

        if message.approved:
            # Resume SSE streaming after approval
            self.run_worker(
                self._consume_sse_resume(message.thread_id),
                thread=True,
                exclusive=True,
                group="sse",
            )

    async def _consume_sse_resume(self, thread_id: str) -> None:
        """Resume SSE streaming after approval by re-streaming from the server."""
        # The server continues the agent execution after approve().
        # We need to stream the remaining events.
        # Use stream() (v2) which handles post-approval continuation.
        try:
            for event in self.client.stream(None, self.session_id):
                event_type = event.get("type", event.get("status", ""))

                if event_type == "error" or event.get("node") == "error":
                    self.call_from_thread(
                        self.post_message,
                        SSEError(event.get("error", "unknown error")),
                    )
                    return

                if event_type in ("completed", "done", "success") or event.get("text"):
                    self.call_from_thread(
                        self.post_message,
                        SSEComplete(event),
                    )
                    return

        except Exception as exc:
            self.call_from_thread(self.post_message, SSEError(str(exc)))

    def action_cancel_query(self) -> None:
        """Cancel the active SSE worker on Esc."""
        self.workers.cancel_group(self, "sse")
        self._stop_spinner()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
