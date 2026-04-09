"""Metadata bar showing execution statistics."""

from __future__ import annotations

from textual.widgets import Static


class MetadataBar(Static):
    """Single-line display of execution metadata (iterations, tools, latency)."""

    DEFAULT_CSS = """
    MetadataBar {
        width: 100%;
        height: 1;
        color: $text-muted;
        margin: 0 0;
    }
    """

    def set_metadata(self, metadata: dict) -> None:
        """Update the metadata display from a result metadata dict."""
        iterations = metadata.get("total_iterations", 0)
        tool_calls = metadata.get("total_tool_calls", 0)
        latency = metadata.get("total_latency_ms", 0)
        self.update(
            f"\u23af iterations={iterations}  " f"tools={tool_calls}  " f"latency={latency:.0f}ms"
        )
