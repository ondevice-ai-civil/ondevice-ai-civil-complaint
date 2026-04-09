"""GovOn Textual TUI application.

Provides a Claude Code-style terminal interface with scrollable message
history, streaming markdown, collapsible tool panels, and an input bar.
Runs in inline mode so output stays in terminal scrollback.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Rule, Static

if TYPE_CHECKING:
    from src.cli.http_client import GovOnClient

CSS_PATH = Path(__file__).parent / "govon_app.tcss"


class GovOnApp(App):
    """Main Textual application for GovOn CLI."""

    CSS_PATH = "govon_app.tcss"

    BINDINGS = [
        ("escape", "cancel_query", "취소"),
        ("ctrl+d", "quit", "종료"),
    ]

    def __init__(
        self,
        client: GovOnClient,
        session_id: str | None = None,
        query: str | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.session_id = session_id
        self._initial_query = query

    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield VerticalScroll(id="message-history")
        yield Rule(id="separator")
        yield Input(placeholder="\u276f ", id="input-bar")
        yield Footer(id="status-footer")

    def on_mount(self) -> None:
        """Focus the input bar on startup."""
        self.query_one("#input-bar", Input).focus()
        if self._initial_query:
            self._submit_query(self._initial_query)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input bar."""
        query = event.value.strip()
        if not query:
            return
        event.input.clear()
        self._submit_query(query)

    def _submit_query(self, query: str) -> None:
        """Process a user query — route to command handler or SSE worker."""
        from src.cli.commands import handle_command, is_command

        if is_command(query):
            try:
                result = handle_command(query)
            except SystemExit:
                self.exit()
                return
            if result is not None:
                history = self.query_one("#message-history", VerticalScroll)
                history.mount(Static(result))
            return

        # Mount user message bubble
        history = self.query_one("#message-history", VerticalScroll)
        history.mount(Static(f"[bold]❯[/bold] {query}"))

        # Start SSE streaming in background thread
        self.run_worker(
            self._consume_sse(query),
            thread=True,
            exclusive=True,
            group="sse",
        )

    async def _consume_sse(self, query: str) -> None:
        """Worker: consume SSE events from the sync httpx client."""
        from src.cli.tui.messages import (
            SSEComplete,
            SSEError,
            SSEResponseDelta,
            SSEThinkingDelta,
            SSEToolEnd,
            SSEToolStart,
        )

        try:
            for event in self.client.stream_v3(query, self.session_id):
                event_type = event.get("type", "")

                if event_type == "error":
                    self.call_from_thread(
                        self.post_message,
                        SSEError(event.get("error", "unknown error")),
                    )
                    return

                if event_type == "thinking_delta":
                    content = event.get("content", "")
                    if content:
                        self.call_from_thread(
                            self.post_message,
                            SSEThinkingDelta(content),
                        )

                elif event_type == "response_delta":
                    content = event.get("content", "")
                    if content:
                        self.call_from_thread(
                            self.post_message,
                            SSEResponseDelta(content),
                        )

                elif event_type == "tool_start":
                    tool_name = event.get("tool", "")
                    if tool_name:
                        self.call_from_thread(
                            self.post_message,
                            SSEToolStart(tool_name),
                        )

                elif event_type == "tool_end":
                    tool_name = event.get("tool", "")
                    self.call_from_thread(
                        self.post_message,
                        SSEToolEnd(tool_name),
                    )

                elif event_type == "run_complete":
                    sid = event.get("session_id") or event.get("thread_id")
                    if sid:
                        self.session_id = sid
                    self.call_from_thread(
                        self.post_message,
                        SSEComplete(event),
                    )

        except Exception as exc:
            self.call_from_thread(
                self.post_message,
                SSEError(str(exc)),
            )

    # -- SSE event handlers --------------------------------------------------

    def on_sse_thinking_delta(self, event: object) -> None:
        """Append thinking text to the message history."""
        from src.cli.tui.messages import SSEThinkingDelta

        msg = event if isinstance(event, SSEThinkingDelta) else None
        if msg is None:
            return
        history = self.query_one("#message-history", VerticalScroll)
        history.mount(Static(f"[dim italic]{msg.content}[/dim italic]"))

    def on_sse_response_delta(self, event: object) -> None:
        """Append response token to the message history."""
        from src.cli.tui.messages import SSEResponseDelta

        msg = event if isinstance(event, SSEResponseDelta) else None
        if msg is None:
            return
        history = self.query_one("#message-history", VerticalScroll)
        history.mount(Static(msg.content))

    def on_sse_tool_start(self, event: object) -> None:
        """Show tool execution start."""
        from src.cli.tui.messages import SSEToolStart

        msg = event if isinstance(event, SSEToolStart) else None
        if msg is None:
            return
        history = self.query_one("#message-history", VerticalScroll)
        history.mount(Static(f"[cyan]\u250c\u2500 \u2699 {msg.tool_name}[/cyan]"))

    def on_sse_tool_end(self, event: object) -> None:
        """Show tool execution end."""
        from src.cli.tui.messages import SSEToolEnd

        msg = event if isinstance(event, SSEToolEnd) else None
        if msg is None:
            return
        history = self.query_one("#message-history", VerticalScroll)
        latency = f" ({msg.latency_ms:.0f}ms)" if msg.latency_ms else ""
        history.mount(
            Static(f"[green]\u2514\u2500 \u2726 {msg.tool_name} \uc644\ub8cc{latency}[/green]")
        )

    def on_sse_error(self, event: object) -> None:
        """Show error message."""
        from src.cli.tui.messages import SSEError

        msg = event if isinstance(event, SSEError) else None
        if msg is None:
            return
        history = self.query_one("#message-history", VerticalScroll)
        history.mount(Static(f"[red bold]\uc624\ub958:[/red bold] {msg.error}"))

    def on_sse_complete(self, event: object) -> None:
        """Finalize response rendering."""
        from src.cli.tui.messages import SSEComplete

        msg = event if isinstance(event, SSEComplete) else None
        if msg is None:
            return

        # Render final text if present and not already streamed via deltas
        text = msg.result.get("text") or msg.result.get("response") or ""
        if text:
            history = self.query_one("#message-history", VerticalScroll)
            from textual.widgets import Markdown

            history.mount(Markdown(text))

        # Show metadata
        metadata = msg.result.get("metadata", {})
        if metadata:
            iterations = metadata.get("total_iterations", 0)
            tool_calls = metadata.get("total_tool_calls", 0)
            latency = metadata.get("total_latency_ms", 0)
            history = self.query_one("#message-history", VerticalScroll)
            history.mount(
                Static(
                    f"[dim]\u23af iterations={iterations}  "
                    f"tools={tool_calls}  "
                    f"latency={latency:.0f}ms[/dim]"
                )
            )

    def action_cancel_query(self) -> None:
        """Cancel the active SSE worker on Esc."""
        self.workers.cancel_group(self, "sse")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
