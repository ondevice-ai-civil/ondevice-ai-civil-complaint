"""CLI logging configuration for GovOn.

Follows the industry standard used by Claude Code, Codex CLI, and Gemini CLI:
- All internal/debug logs are written to files only (~/.govon/logs/)
- Zero console noise in normal operation
- Opt-in debug output via setup_logging(debug=True)
- 7-day log retention with automatic cleanup
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_DIR = Path.home() / ".govon" / "logs"
_LOG_RETENTION_DAYS = 7
_LOG_FORMAT_FILE = (
    "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} | {message}"
)
_LOG_FORMAT_STDERR = "{time:HH:mm:ss} | {level} | {message}"

# ---------------------------------------------------------------------------
# Module-level initialization: silence loguru's default stderr handler
# immediately on import so no debug output leaks before setup_logging() runs.
# ---------------------------------------------------------------------------
logger.remove()  # remove the default stderr handler (id=0)


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the CLI process.

    Parameters
    ----------
    debug:
        When True, also emit WARNING+ log records to stderr so the operator
        can see internal diagnostics.  When False (default), all log output
        goes to the rotating file only — the terminal UI stays clean.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File handler: all levels, structured format, 7-day rotation
    logger.add(
        str(_LOG_DIR / "cli-{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format=_LOG_FORMAT_FILE,
        rotation="00:00",       # rotate at midnight
        retention=f"{_LOG_RETENTION_DAYS} days",
        encoding="utf-8",
        enqueue=True,           # non-blocking writes
        catch=True,             # suppress handler exceptions
    )

    if debug:
        # Optional stderr handler for operator/developer use
        logger.add(
            sys.stderr,
            level="WARNING",
            format=_LOG_FORMAT_STDERR,
            colorize=True,
            catch=True,
        )


def cleanup_old_logs() -> int:
    """Delete log files older than 7 days from the log directory.

    Returns
    -------
    int
        Number of files deleted.
    """
    import time

    if not _LOG_DIR.exists():
        return 0

    cutoff = time.time() - (_LOG_RETENTION_DAYS * 86_400)
    deleted = 0
    for log_file in _LOG_DIR.glob("cli-*.log"):
        try:
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
                deleted += 1
        except OSError:
            pass  # ignore permission errors or race conditions

    return deleted
