#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# GovOn Blue/Green Deployment Script
#
# Usage:
#   ./scripts/deploy.sh deploy <image-tag>     Deploy new version
#   ./scripts/deploy.sh rollback               Rollback to previous version
#   ./scripts/deploy.sh status                 Show current deployment status
#   ./scripts/deploy.sh health                 Check health of active deployment
# ──────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.prod.yml"
STATE_FILE="${PROJECT_DIR}/.deploy-state"
HEALTH_TIMEOUT=120
HEALTH_INTERVAL=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ──────────────────────────────────────────────
# State management
# ──────────────────────────────────────────────

get_active_slot() {
  if [ -f "$STATE_FILE" ]; then
    cat "$STATE_FILE"
  else
    echo "none"
  fi
}

get_inactive_slot() {
  local active
  active=$(get_active_slot)
  if [ "$active" = "blue" ]; then
    echo "green"
  else
    echo "blue"
  fi
}

get_slot_port() {
  local slot=$1
  if [ "$slot" = "blue" ]; then echo 8001; else echo 8002; fi
}

# ──────────────────────────────────────────────
# Health check with retry
# ──────────────────────────────────────────────

wait_for_health() {
  local port=$1
  local elapsed=0
  log_info "헬스체크 대기 중 (포트: ${port}, 타임아웃: ${HEALTH_TIMEOUT}초)..."

  while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
      echo ""
      log_info "헬스체크 통과 (${elapsed}초 소요)"
      return 0
    fi
    sleep $HEALTH_INTERVAL
    elapsed=$((elapsed + HEALTH_INTERVAL))
    printf "."
  done

  echo ""
  log_error "헬스체크 실패 (${HEALTH_TIMEOUT}초 타임아웃)"
  return 1
}

# ──────────────────────────────────────────────
# Prerequisites check
# ──────────────────────────────────────────────

check_prerequisites() {
  if ! command -v docker &>/dev/null; then
    log_error "Docker가 설치되어 있지 않습니다."
    exit 1
  fi

  if ! docker compose version &>/dev/null; then
    log_error "Docker Compose가 설치되어 있지 않습니다."
    exit 1
  fi

  if [ ! -f "$COMPOSE_FILE" ]; then
    log_error "Compose 파일을 찾을 수 없습니다: ${COMPOSE_FILE}"
    exit 1
  fi
}

# ──────────────────────────────────────────────
# Deploy new version
# ──────────────────────────────────────────────

cmd_deploy() {
  local image_tag="${1:-latest}"
  local active
  local target
  local target_port

  active=$(get_active_slot)
  target=$(get_inactive_slot)
  target_port=$(get_slot_port "$target")

  check_prerequisites

  log_info "=== GovOn 배포 시작: v${image_tag} ==="
  log_info "현재 활성 슬롯: ${active}"
  log_info "배포 대상 슬롯: ${target}"
  echo ""

  # Set the tag for the target slot
  if [ "$target" = "blue" ]; then
    export BLUE_TAG="$image_tag"
  else
    export GREEN_TAG="$image_tag"
  fi

  # Pull new image
  log_info "이미지 풀링: ghcr.io/govon-org/govon:${image_tag}..."
  docker pull "ghcr.io/govon-org/govon:${image_tag}"

  # Create volume directories
  mkdir -p "${PROJECT_DIR}/models" "${PROJECT_DIR}/data" "${PROJECT_DIR}/agents" "${PROJECT_DIR}/configs"

  # Start target slot
  log_info "${target} 슬롯 시작 중..."
  docker compose -f "$COMPOSE_FILE" --profile "$target" up -d

  # Wait for health
  if wait_for_health "$target_port"; then
    log_info "${target} 배포가 정상 작동합니다!"

    # Update state
    echo "$target" > "$STATE_FILE"
    log_info "활성 슬롯 변경: ${active} -> ${target}"

    # Stop previous slot
    if [ "$active" != "none" ]; then
      log_info "이전 ${active} 슬롯 중지 중..."
      docker compose -f "$COMPOSE_FILE" --profile "$active" down
    fi

    echo ""
    log_info "=== 배포 완료 ==="
    cmd_status
  else
    log_error "배포 실패! 롤백 수행 중..."
    docker compose -f "$COMPOSE_FILE" --profile "$target" down
    log_error "실패한 배포를 정리했습니다. 이전 버전이 계속 활성 상태입니다."
    exit 1
  fi
}

# ──────────────────────────────────────────────
# Rollback to previous version
# ──────────────────────────────────────────────

cmd_rollback() {
  local active
  local previous
  local prev_port

  active=$(get_active_slot)
  previous=$(get_inactive_slot)
  prev_port=$(get_slot_port "$previous")

  check_prerequisites

  if [ "$active" = "none" ]; then
    log_error "롤백할 활성 배포가 없습니다."
    exit 1
  fi

  log_warn "=== 롤백 시작: ${active} -> ${previous} ==="

  # Start previous slot
  docker compose -f "$COMPOSE_FILE" --profile "$previous" up -d

  if wait_for_health "$prev_port"; then
    # Stop current active
    docker compose -f "$COMPOSE_FILE" --profile "$active" down
    echo "$previous" > "$STATE_FILE"
    echo ""
    log_info "=== 롤백 완료. 활성 슬롯: ${previous} ==="
    cmd_status
  else
    log_error "롤백 실패! 수동 조치가 필요합니다."
    log_error "현재 활성 슬롯(${active})은 그대로 유지됩니다."
    docker compose -f "$COMPOSE_FILE" --profile "$previous" down
    exit 1
  fi
}

# ──────────────────────────────────────────────
# Show deployment status
# ──────────────────────────────────────────────

cmd_status() {
  local active
  active=$(get_active_slot)
  local blue_status
  local green_status

  blue_status=$(docker ps --filter name=govon-blue --format '{{.Status}}' 2>/dev/null || echo "stopped")
  green_status=$(docker ps --filter name=govon-green --format '{{.Status}}' 2>/dev/null || echo "stopped")

  [ -z "$blue_status" ] && blue_status="stopped"
  [ -z "$green_status" ] && green_status="stopped"

  echo ""
  echo "========================================"
  echo "       GovOn 배포 상태"
  echo "========================================"
  echo " 활성 슬롯  : ${active}"
  echo " Blue  (8001): ${blue_status}"
  echo " Green (8002): ${green_status}"
  echo "========================================"
}

# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

cmd_health() {
  local active
  local port

  active=$(get_active_slot)
  if [ "$active" = "none" ]; then
    log_error "활성 배포가 없습니다."
    exit 1
  fi

  port=$(get_slot_port "$active")

  if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
    log_info "활성 배포(${active})가 정상입니다."
  else
    log_error "활성 배포(${active})가 비정상입니다!"
    exit 1
  fi
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

case "${1:-help}" in
  deploy)   cmd_deploy "${2:-latest}" ;;
  rollback) cmd_rollback ;;
  status)   cmd_status ;;
  health)   cmd_health ;;
  *)
    echo "GovOn Blue/Green 배포 스크립트"
    echo ""
    echo "사용법: $0 {deploy <tag>|rollback|status|health}"
    echo ""
    echo "명령어:"
    echo "  deploy <tag>   새 버전 배포 (기본값: latest)"
    echo "  rollback       이전 버전으로 롤백"
    echo "  status         현재 배포 상태 확인"
    echo "  health         활성 배포 헬스체크"
    exit 1
    ;;
esac
