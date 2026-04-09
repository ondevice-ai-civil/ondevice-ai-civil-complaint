"""Textual TUI package for GovOn CLI.

Provides a full terminal UI when Textual is installed. Falls back to the
legacy rich+prompt_toolkit REPL when Textual is not available.
"""

HAS_TEXTUAL = False
try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:
    pass
