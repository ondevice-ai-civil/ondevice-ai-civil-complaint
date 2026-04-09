"""Message bubble container for user and assistant messages."""

from __future__ import annotations

from textual.containers import Vertical


class MessageBubble(Vertical):
    """Container for a single conversation turn (user or assistant).

    Usage::

        bubble = MessageBubble(role="user")
        bubble.mount(Static("user query text"))

        bubble = MessageBubble(role="assistant")
        bubble.mount(SpinnerWidget())  # later replaced by MarkdownView
    """

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        margin: 1 0 0 0;
        padding: 0 0;
    }
    """

    def __init__(self, role: str = "assistant", **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.role = role
        self.add_class(role)
