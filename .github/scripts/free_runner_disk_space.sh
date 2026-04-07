#!/usr/bin/env bash

set -euo pipefail

echo "Disk usage before cleanup:"
df -h

# GitHub-hosted Ubuntu runners ship with large SDK directories that are not
# needed for this repository's Docker validation path.
for path in /usr/share/dotnet /opt/ghc /usr/local/lib/android; do
  if [ -d "${path}" ]; then
    echo "Removing ${path}"
    sudo rm -rf "${path}"
  fi
done

docker system prune -af >/dev/null 2>&1 || true

echo "Disk usage after cleanup:"
df -h
