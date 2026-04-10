"""GovOn CLI launcher — finds and executes the bundled Node SEA binary.

This module is the entry point for `pip install govon`. It locates the
platform-specific Node SEA binary bundled inside the wheel and exec's it,
replacing the Python process entirely (on Unix) or spawning it (on Windows).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

# Map Python platform.machine() to binary naming convention
_ARCH_MAP: dict[str, str] = {
    "x86_64": "x64",
    "amd64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
}

# Map Python platform.system() to binary naming convention
_SYSTEM_MAP: dict[str, str] = {
    "windows": "win",
    "darwin": "darwin",
    "linux": "linux",
}

_SUPPORTED_PLATFORMS = "darwin-x64, darwin-arm64, linux-x64, linux-arm64, win-x64"


def _resolve_binary() -> Path:
    """Locate the SEA binary for the current platform."""
    system_raw = platform.system().lower()
    system = _SYSTEM_MAP.get(system_raw, system_raw)

    machine = platform.machine().lower()
    arch = _ARCH_MAP.get(machine)
    if arch is None:
        print(
            f"Error: Unsupported architecture '{machine}'.\n"
            f"GovOn supports x64 and arm64 only.\n"
            f"Try: npm install -g govon (requires Node.js 18+)",
            file=sys.stderr,
        )
        sys.exit(1)

    name = f"govon-{system}-{arch}"
    if system == "win":
        name += ".exe"

    bin_dir = Path(__file__).parent / "_bin"
    binary = bin_dir / name

    if not binary.exists():
        print(
            f"Error: No GovOn CLI binary for {system}-{arch}.\n"
            f"Supported platforms: {_SUPPORTED_PLATFORMS}\n"
            f"Try: npm install -g govon (requires Node.js 18+)",
            file=sys.stderr,
        )
        sys.exit(1)

    return binary


def main() -> None:
    """Entry point — exec the Node SEA binary, replacing this process."""
    binary = _resolve_binary()

    if os.name == "nt":
        # Windows does not support true process replacement via os.execv.
        # Use subprocess.run and propagate the exit code.
        result = subprocess.run([str(binary), *sys.argv[1:]])
        sys.exit(result.returncode)
    else:
        # Make executable on Unix (wheels may strip execute bits)
        mode = binary.stat().st_mode
        if not (mode & 0o111):
            binary.chmod(mode | 0o755)
        # Replace the Python process entirely
        os.execv(str(binary), [str(binary)] + sys.argv[1:])
