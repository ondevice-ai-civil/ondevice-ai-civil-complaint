#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Detect platform
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ARCH="x64" ;;
  aarch64|arm64) ARCH="arm64" ;;
esac
PLATFORM="${OS}-${ARCH}"

echo "Building Node SEA binary for ${PLATFORM}..."

# Step 1: Build single-file ESM bundle with all deps inlined
echo "  [1/5] Bundling with esbuild..."
npx esbuild src/index.tsx \
  --bundle \
  --platform=node \
  --target=node22 \
  --format=esm \
  --jsx=automatic \
  --outfile=dist/sea-bundle.js \
  --define:process.env.NODE_ENV=\"production\" \
  --external:react-devtools-core

# Step 2: Generate SEA preparation blob
echo "  [2/5] Generating SEA blob..."
node --experimental-sea-config sea-config.json

# Step 3: Copy node binary
BINARY_NAME="govon-${PLATFORM}"
if [ "$OS" = "windows" ] || [[ "$OS" == mingw* ]]; then
  BINARY_NAME="${BINARY_NAME}.exe"
fi
echo "  [3/5] Copying node binary -> dist/${BINARY_NAME}"
cp "$(command -v node)" "dist/${BINARY_NAME}"

# Step 4: Remove existing signature on macOS (required before injection)
if [ "$OS" = "darwin" ]; then
  echo "  [4/5] Removing macOS code signature..."
  codesign --remove-signature "dist/${BINARY_NAME}"
fi

# Step 5: Inject SEA blob
echo "  [4/5] Injecting SEA blob..."
npx postject "dist/${BINARY_NAME}" NODE_SEA_BLOB dist/sea-prep.blob \
  --sentinel-fuse NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2

# Step 6: Re-sign on macOS
if [ "$OS" = "darwin" ]; then
  echo "  [5/5] Re-signing macOS binary..."
  codesign --sign - "dist/${BINARY_NAME}"
fi

# Verify
echo ""
echo "SEA binary created: dist/${BINARY_NAME}"
ls -lh "dist/${BINARY_NAME}"
echo "Test: dist/${BINARY_NAME} --version"
"dist/${BINARY_NAME}" --version || echo "(version check may fail if index.tsx requires terminal)"
