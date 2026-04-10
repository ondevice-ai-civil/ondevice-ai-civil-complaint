"""GovOn CLI launcher — finds and executes the Node SEA binary.

This module is the entry point for `pip install govon`. On first run it
downloads the platform-specific SEA binary from GitHub Releases into
~/.govon/bin/ and then exec's it.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

# Map Python platform identifiers to binary naming convention
_ARCH_MAP: dict[str, str] = {
    "x86_64": "x64",
    "amd64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
}

_SYSTEM_MAP: dict[str, str] = {
    "windows": "win",
    "darwin": "darwin",
    "linux": "linux",
}

_SUPPORTED_PLATFORMS = "darwin-arm64, linux-x64, linux-arm64, win-x64"

# Binary cache directory
_BIN_DIR = Path.home() / ".govon" / "bin"


def _get_version() -> str:
    """Read the installed package version."""
    try:
        from importlib.metadata import version

        return version("govon")
    except Exception:
        return "0.0.0"


def _resolve_platform() -> tuple[str, str]:
    """Detect current platform and architecture."""
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

    return system, arch


def _binary_name(system: str, arch: str) -> str:
    """Construct the binary filename for the given platform."""
    name = f"govon-{system}-{arch}"
    if system == "win":
        name += ".exe"
    return name


def _download_binary(version: str, system: str, arch: str, dest: Path) -> None:
    """Download the SEA binary from GitHub Releases."""
    name = _binary_name(system, arch)
    url = f"https://github.com/GovOn-Org/GovOn/releases/download/v{version}/{name}"

    print(f"Downloading GovOn CLI binary for {system}-{arch}...", file=sys.stderr)
    print(f"  {url}", file=sys.stderr)

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        print(
            f"Error: Failed to download binary: {exc}\n"
            f"Try: npm install -g govon (requires Node.js 18+)",
            file=sys.stderr,
        )
        # Clean up partial download
        dest.unlink(missing_ok=True)
        sys.exit(1)

    # Make executable on Unix
    if os.name != "nt":
        dest.chmod(dest.stat().st_mode | 0o755)

    print(f"  Installed to {dest}", file=sys.stderr)


def _resolve_binary() -> Path:
    """Locate or download the SEA binary for the current platform."""
    version = _get_version()
    system, arch = _resolve_platform()
    name = _binary_name(system, arch)

    # Check local cache first
    binary = _BIN_DIR / version / name
    if binary.exists():
        return binary

    # Check bundled _bin/ (for platform-specific wheels)
    bundled = Path(__file__).parent / "_bin" / name
    if bundled.exists():
        return bundled

    # Download from GitHub Releases
    _download_binary(version, system, arch, binary)
    return binary


def main() -> None:
    """Entry point — exec the Node SEA binary, replacing this process."""
    binary = _resolve_binary()

    if os.name == "nt":
        result = subprocess.run([str(binary), *sys.argv[1:]])
        sys.exit(result.returncode)
    else:
        # Ensure executable
        mode = binary.stat().st_mode
        if not (mode & 0o111):
            binary.chmod(mode | 0o755)
        os.execv(str(binary), [str(binary)] + sys.argv[1:])
