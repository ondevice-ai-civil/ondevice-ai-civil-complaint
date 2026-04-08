#!/usr/bin/env bash
# GovOn Runtime Entrypoint
# 1) vLLM OpenAI-compatible 서버를 백그라운드로 기동
# 2) health check로 준비 완료 대기
# 3) FastAPI 서버 실행 (foreground, GPU 접근 차단)
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL_PATH:-LGAI-EXAONE/EXAONE-4.0-32B-AWQ}"
GPU_UTIL="${GPU_UTILIZATION:-0.90}"
MAX_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${MODEL_DTYPE:-half}"
KV_DTYPE="${KV_CACHE_DTYPE:-auto}"
SKIP_MODEL="${SKIP_MODEL_LOAD:-false}"

# SKIP_MODEL_LOAD 시 vLLM 서버 없이 FastAPI만 실행
if [ "$SKIP_MODEL" = "true" ] || [ "$SKIP_MODEL" = "1" ]; then
    echo "[entrypoint] SKIP_MODEL_LOAD=true: FastAPI만 실행"
    CUDA_VISIBLE_DEVICES="" exec python3.10 -m src.inference.api_server
fi

# --- vLLM 서버 기동 ---
VLLM_ARGS=(
    --model "$MODEL"
    --port "$VLLM_PORT"
    --host 127.0.0.1
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_LEN"
    --kv-cache-dtype "$KV_DTYPE"
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser hermes
)

# LoRA 어댑터 설정 (ADAPTER_PATHS 환경변수에서 파싱)
if [ -n "${ADAPTER_PATHS:-}" ]; then
    VLLM_ARGS+=(--enable-lora --max-loras 4 --max-lora-rank 64)
    # ADAPTER_PATHS 형식: "civil=repo/path,legal=repo/path"
    # vLLM 0.19: --lora-modules를 한 번만 사용, 여러 어댑터는 배열 전개로 개별 인자 전달
    IFS=',' read -ra PAIRS <<< "$ADAPTER_PATHS"
    LORA_MODULES=()
    for pair in "${PAIRS[@]}"; do
        name="${pair%%=*}"
        path="${pair#*=}"
        LORA_MODULES+=("${name}=${path}")
    done
    VLLM_ARGS+=(--lora-modules "${LORA_MODULES[@]}")
fi

echo "[entrypoint] vLLM 서버 기동: port=$VLLM_PORT model=$MODEL"
echo "[entrypoint] args: ${VLLM_ARGS[*]}"

python3.10 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
VLLM_PID=$!

# --- vLLM health check ---
# CUDA_VISIBLE_DEVICES="": health check python 프로세스에서 GPU 접근 차단
#   → torch/vllm import 시 CUDA 초기화 hang 방지
# except Exception: bare except(except:) 사용 금지
#   → sys.exit()이 raise하는 SystemExit을 잡아버려 항상 실패 반환
# timeout 10: 프로세스-레벨 타임아웃 (urllib timeout과 별개)
echo "[entrypoint] vLLM 서버 준비 대기 중..."
MAX_WAIT=900
WAITED=0
INTERVAL=5

# nvidia/cuda 이미지에 coreutils(timeout)가 없을 수 있으므로 조건부 사용
if command -v timeout &>/dev/null; then
    TIMEOUT_CMD="timeout 10"
else
    TIMEOUT_CMD=""
fi

_health_check() {
    CUDA_VISIBLE_DEVICES="" $TIMEOUT_CMD python3.10 -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://localhost:${VLLM_PORT}/health', timeout=5)
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
" 2>&1
    return $?
}

while [ $WAITED -lt $MAX_WAIT ]; do
    if _health_check; then
        echo "[entrypoint] vLLM 서버 준비 완료 (${WAITED}s)"
        break
    fi
    # vLLM 프로세스가 죽었는지 확인
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[entrypoint] ERROR: vLLM 프로세스 종료됨"
        wait $VLLM_PID; VLLM_EXIT=$?
        echo "[entrypoint] vLLM exit code=$VLLM_EXIT"
        exit $VLLM_EXIT
    fi
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[entrypoint] ERROR: vLLM 서버 시작 타임아웃 (${MAX_WAIT}s)"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# --- FastAPI 서버 실행 (foreground) ---
# CUDA_VISIBLE_DEVICES="": FastAPI는 httpx로 vLLM API만 호출하므로 GPU 불필요
#   → vLLM import 시 CUDA context 생성 방지, GPU 메모리 절약
# exec 대신 백그라운드 실행 후 wait: SIGTERM을 vLLM/FastAPI 양쪽에 전파하기 위함
cleanup() {
    echo "[entrypoint] Shutting down..."
    kill $FASTAPI_PID 2>/dev/null || true
    kill $VLLM_PID 2>/dev/null || true
    wait $FASTAPI_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

echo "[entrypoint] FastAPI 서버 기동: port=${PORT:-7860}"
CUDA_VISIBLE_DEVICES="" python3.10 -m src.inference.api_server &
FASTAPI_PID=$!

# 두 자식 중 먼저 종료된 프로세스를 감지하여 나머지도 정리
wait -n $FASTAPI_PID $VLLM_PID 2>/dev/null || true
EXITED=$?
echo "[entrypoint] 프로세스 종료 감지 (exit=$EXITED), cleanup 진행"
