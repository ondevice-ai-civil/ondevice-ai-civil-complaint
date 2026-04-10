#!/usr/bin/env bash
# GovOn CLI installer — installs via npm globally.
# Usage: curl -fsSL https://raw.githubusercontent.com/GovOn-Org/GovOn/main/scripts/install.sh | bash
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
RESET='\033[0m'

info()  { echo -e "${GREEN}${BOLD}==>${RESET} ${BOLD}$*${RESET}"; }
warn()  { echo -e "${YELLOW}${BOLD}warning:${RESET} $*"; }
error() { echo -e "${RED}${BOLD}error:${RESET} $*" >&2; }

# Check for Node.js
if ! command -v node &>/dev/null; then
  error "Node.js is not installed."
  echo ""
  echo "Install Node.js first:"
  echo "  macOS:   brew install node"
  echo "  Linux:   curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -"
  echo "  Windows: https://nodejs.org/"
  exit 1
fi

NODE_VERSION=$(node --version)
NODE_MAJOR=$(echo "$NODE_VERSION" | sed 's/v//' | cut -d. -f1)

if [ "$NODE_MAJOR" -lt 18 ]; then
  error "Node.js 18+ is required. Found: $NODE_VERSION"
  exit 1
fi

info "Found Node.js $NODE_VERSION"

# Check for npm
if ! command -v npm &>/dev/null; then
  error "npm is not found. Please install Node.js with npm."
  exit 1
fi

# Install govon globally
info "Installing govon via npm..."
npm install -g govon

# Verify
if command -v govon &>/dev/null; then
  GOVON_VERSION=$(govon --version 2>/dev/null || echo "unknown")
  echo ""
  info "GovOn CLI installed successfully!"
  echo "  Version: $GOVON_VERSION"
  echo "  Path:    $(which govon)"
  echo ""
  echo "  Run ${BOLD}govon${RESET} to start."
else
  warn "govon was installed but is not in PATH."
  echo "  Try: npx govon"
fi
