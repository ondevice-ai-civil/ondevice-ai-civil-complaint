#!/usr/bin/env bash
# generate-changelog.sh — Generate categorised changelog between two git tags
# Usage: ./scripts/generate-changelog.sh [PREVIOUS_TAG] [CURRENT_TAG]
#
# If PREVIOUS_TAG is empty, all commits up to CURRENT_TAG are included.
# Output is written to stdout in Markdown format.

set -euo pipefail

PREV_TAG="${1:-}"
CURR_TAG="${2:-HEAD}"
REPO_URL="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-govon-org/govon}"

# --- collect raw commits ------------------------------------------------
if [[ -z "$PREV_TAG" ]]; then
  RAW=$(git log "$CURR_TAG" --pretty=format:"%H %s" 2>/dev/null || true)
else
  RAW=$(git log "${PREV_TAG}..${CURR_TAG}" --pretty=format:"%H %s" 2>/dev/null || true)
fi

if [[ -z "$RAW" ]]; then
  echo "_No changes detected._"
  exit 0
fi

# --- helpers --------------------------------------------------------------
extract_message() {
  local prefix="$1"
  local msg="$2"
  # Strip "prefix:" or "prefix(scope):" from the beginning
  local result
  result=$(echo "$msg" | sed -E "s/^${prefix}(\([^)]*\))?:\s*//" )
  echo "$result"
}

# --- categorise commits --------------------------------------------------
FEAT=()
FIX=()
DOCS=()
REFACTOR=()
TEST=()
CHORE=()
OTHER=()

while IFS= read -r line; do
  sha="${line%% *}"
  short_sha="${sha:0:7}"
  msg="${line#* }"

  # Extract PR number if present (e.g., "(#123)")
  pr_link=""
  if [[ "$msg" =~ \(\#([0-9]+)\) ]]; then
    pr_num="${BASH_REMATCH[1]}"
    pr_link=" ([#${pr_num}](${REPO_URL}/pull/${pr_num}))"
    msg=$(echo "$msg" | sed "s/ *(#${pr_num})//")
  fi

  commit_link="[\`${short_sha}\`](${REPO_URL}/commit/${sha})"

  # Conventional Commits parsing via case + sed
  case "$msg" in
    feat:*|feat\(*)
      clean_msg=$(extract_message "feat" "$msg")
      FEAT+=("- ${clean_msg}${pr_link} (${commit_link})")
      ;;
    fix:*|fix\(*)
      clean_msg=$(extract_message "fix" "$msg")
      FIX+=("- ${clean_msg}${pr_link} (${commit_link})")
      ;;
    docs:*|docs\(*)
      clean_msg=$(extract_message "docs" "$msg")
      DOCS+=("- ${clean_msg}${pr_link} (${commit_link})")
      ;;
    refactor:*|refactor\(*)
      clean_msg=$(extract_message "refactor" "$msg")
      REFACTOR+=("- ${clean_msg}${pr_link} (${commit_link})")
      ;;
    test:*|test\(*)
      clean_msg=$(extract_message "test" "$msg")
      TEST+=("- ${clean_msg}${pr_link} (${commit_link})")
      ;;
    chore:*|chore\(*)
      clean_msg=$(extract_message "chore" "$msg")
      CHORE+=("- ${clean_msg}${pr_link} (${commit_link})")
      ;;
    *)
      OTHER+=("- ${msg}${pr_link} (${commit_link})")
      ;;
  esac
done <<< "$RAW"

# --- output markdown -----------------------------------------------------
print_section() {
  local title="$1"
  shift
  if [[ $# -le 0 ]]; then
    return
  fi
  echo "### ${title}"
  echo ""
  for item in "$@"; do
    echo "$item"
  done
  echo ""
}

print_section "New Features" "${FEAT[@]+"${FEAT[@]}"}"
print_section "Bug Fixes" "${FIX[@]+"${FIX[@]}"}"
print_section "Documentation" "${DOCS[@]+"${DOCS[@]}"}"
print_section "Refactoring" "${REFACTOR[@]+"${REFACTOR[@]}"}"
print_section "Tests" "${TEST[@]+"${TEST[@]}"}"
print_section "Chores" "${CHORE[@]+"${CHORE[@]}"}"
print_section "Other Changes" "${OTHER[@]+"${OTHER[@]}"}"

# --- contributors ---------------------------------------------------------
echo "### Contributors"
echo ""
if [[ -z "$PREV_TAG" ]]; then
  git log "$CURR_TAG" --pretty=format:"%aN" | sort -u | while IFS= read -r name; do
    echo "- ${name}"
  done
else
  git log "${PREV_TAG}..${CURR_TAG}" --pretty=format:"%aN" | sort -u | while IFS= read -r name; do
    echo "- ${name}"
  done
fi
echo ""
