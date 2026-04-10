"""GovOn CLI launcher — finds and executes the npm TUI.

This module is the entry point for `pip install govon`. It attempts to
run the govon npm package via npx, or guides the user to install Node.js.

Note: Node SEA (Single Executable Application) is not yet compatible with
Ink's top-level await (yoga-layout WASM). Until Node SEA stabilizes ESM
support, pip-installed govon delegates to npx/node.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _find_npx() -> str | None:
    """Find npx in PATH."""
    return shutil.which("npx")


def _find_node() -> str | None:
    """Find node in PATH."""
    return shutil.which("node")


def main() -> None:
    """Entry point — run govon via npx or installed global npm package."""
    npx = _find_npx()

    if npx:
        # Try npx govon (uses globally installed or fetches from npm)
        try:
            result = subprocess.run(
                [npx, "--yes", "govon@latest", *sys.argv[1:]],
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as exc:
            print(f"Error running npx govon: {exc}", file=sys.stderr)
            sys.exit(1)

    # No npx found — guide the user
    node = _find_node()
    if node:
        print(
            "Error: npx not found but node is installed.\n"
            "Install govon globally: npm install -g govon\n"
            "Then run: govon",
            file=sys.stderr,
        )
    else:
        print(
            "Error: Node.js is not installed.\n"
            "GovOn TUI requires Node.js 18+.\n"
            "\n"
            "Install Node.js:\n"
            "  macOS:  brew install node\n"
            "  Linux:  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -\n"
            "  All:    https://nodejs.org/\n"
            "\n"
            "Then run: npx govon",
            file=sys.stderr,
        )
    sys.exit(1)
