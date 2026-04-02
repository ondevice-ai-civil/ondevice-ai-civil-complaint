#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_FILE="${PROJECT_DIR}/govon-image.tar.gz"
ENV_TEMPLATE="${PROJECT_DIR}/.env.airgap.example"
ENV_FILE="${PROJECT_DIR}/.env"
API_KEY_PLACEHOLDER="CHANGE_ME_TO_SECURE_RANDOM_KEY"
BM25_INDEX_HMAC_KEY_PLACEHOLDER="CHANGE_ME_TO_SECURE_HMAC_KEY"

extract_env_value() {
    local key="$1"
    local file="$2"

    awk -F= -v key="$key" '
        $0 ~ "^[[:space:]]*" key "=" {
            sub(/^[^=]*=/, "", $0)
            print $0
            exit
        }
    ' "$file"
}

require_secure_env_value() {
    local key="$1"
    local placeholder="$2"
    local value

    value="$(extract_env_value "$key" "$ENV_FILE")"
    if [ -z "$value" ] || [ "$value" = "$placeholder" ]; then
        echo "[ERROR] ${key} 값이 비어 있거나 예시 placeholder 그대로입니다."
        echo "        ${ENV_FILE}에서 ${key}를 안전한 임의 문자열로 수정한 뒤 다시 실행하세요."
        exit 1
    fi
}

echo "=== GovOn 오프라인 배포 스크립트 ==="

# 1. Docker 설치 확인
if ! command -v docker &>/dev/null; then
    echo "[ERROR] Docker가 설치되어 있지 않습니다."
    echo "설치 가이드: https://docs.docker.com/engine/install/"
    exit 1
fi
echo "[OK] Docker: $(docker --version)"

# 2. Docker Compose 확인
if ! docker compose version &>/dev/null; then
    echo "[ERROR] Docker Compose가 설치되어 있지 않습니다."
    exit 1
fi
echo "[OK] Docker Compose: $(docker compose version --short)"

# 3. NVIDIA Container Toolkit 확인 (경고만)
if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "[OK] NVIDIA Container Toolkit 감지됨"
else
    echo "[WARNING] NVIDIA Container Toolkit이 감지되지 않았습니다."
    echo "GPU 가속이 필요합니다: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# 4. 이미지 파일 확인 및 로드
if [ ! -f "$IMAGE_FILE" ]; then
    echo "[ERROR] 이미지 파일을 찾을 수 없습니다: $IMAGE_FILE"
    exit 1
fi
echo "Docker 이미지 로드 중... (시간이 소요될 수 있습니다)"
gunzip -c "$IMAGE_FILE" | docker load
echo "[OK] 이미지 로드 완료"

# 5. 환경변수 템플릿 준비
if [ ! -f "$ENV_FILE" ] && [ -f "$ENV_TEMPLATE" ]; then
    cp "$ENV_TEMPLATE" "$ENV_FILE"
    echo "[OK] .env 파일을 .env.airgap.example 기준으로 생성했습니다."
    echo "     API_KEY, BM25_INDEX_HMAC_KEY, CORS_ORIGINS 등을 수정한 뒤 재실행하세요."
fi

if [ -z "${MODEL_PATH:-}" ] && [ ! -f "$ENV_FILE" ]; then
    echo "[INFO] MODEL_PATH가 설정되지 않았습니다."
    echo "  오프라인 환경에서는 컨테이너 내부 경로를 지정하세요:"
    echo "  export MODEL_PATH=/app/models/GovOn-EXAONE-AWQ-v2"
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "[ERROR] 환경변수 파일을 찾을 수 없습니다: $ENV_FILE"
    exit 1
fi

require_secure_env_value "API_KEY" "$API_KEY_PLACEHOLDER"
require_secure_env_value "BM25_INDEX_HMAC_KEY" "$BM25_INDEX_HMAC_KEY_PLACEHOLDER"

# 6. 볼륨 디렉토리 생성
echo "볼륨 디렉토리 생성 중..."
mkdir -p \
    "${PROJECT_DIR}/models" \
    "${PROJECT_DIR}/data" \
    "${PROJECT_DIR}/agents" \
    "${PROJECT_DIR}/configs" \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/.cache"
echo "[OK] 볼륨 디렉토리 준비 완료"

# 7. 컨테이너 실행
echo "컨테이너 시작 중..."
docker compose --env-file "${ENV_FILE}" -f "${PROJECT_DIR}/docker-compose.offline.yml" up -d
echo "[OK] 컨테이너 시작됨"

# 8. 헬스체크 대기
echo "서버 시작 대기 중... (최대 120초)"
for i in $(seq 1 24); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "=============================="
        echo "[SUCCESS] GovOn 서버가 정상 시작되었습니다!"
        echo "API 주소: http://localhost:8000"
        echo "헬스체크: http://localhost:8000/health"
        echo "=============================="
        exit 0
    fi
    printf "."
    sleep 5
done

echo ""
echo "[ERROR] 서버 시작 실패 (120초 타임아웃)"
echo "로그 확인: docker compose --env-file ${ENV_FILE} -f ${PROJECT_DIR}/docker-compose.offline.yml logs"
exit 1
