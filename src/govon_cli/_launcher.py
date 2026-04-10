"""GovOn CLI launcher — finds and executes the bundled Node SEA binary.

This module is the entry point for `pip install govon`. It locates the
platform-specific Node SEA binary bundled inside the wheel and exec's it,
replacing the Python process entirely.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# Map Python platform identifiers to the binary naming convention
_ARCH_MAP = {
    "x86_64": "x64",
    "amd64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


def _resolve_binary() -> Path:
    """Locate the SEA binary for the current platform."""
    system = platform.system().lower()  # darwin, linux, windows
    machine = platform.machine().lower()
    arch = _ARCH_MAP.get(machine, machine)

    name = f"govon-{system}-{arch}"
    if system == "windows":
        name += ".exe"

    bin_dir = Path(__file__).parent / "_bin"
    binary = bin_dir / name

    if not binary.exists():
        # List available binaries for diagnostics
        available = [f.name for f in bin_dir.iterdir()] if bin_dir.is_dir() else []
        print(
            f"Error: No GovOn CLI binary for {system}-{arch}.\n"
            f"Available: {available or 'none'}\n"
            f"Try: npm install -g govon (requires Node.js 18+)",
            file=sys.stderr,
        )
        sys.exit(1)

    return binary


def main() -> None:
    """Entry point — exec the Node SEA binary, replacing this process."""
    binary = _resolve_binary()

    # Make executable on Unix (wheels may strip execute bits)
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | 0o755)

    # Replace the Python process with the SEA binary
    os.execv(str(binary), [str(binary)] + sys.argv[1:])
