#!/usr/bin/env bash
# GovOn Runtime Entrypoint
# 1) vLLM OpenAI-compatible 서버를 백그라운드로 기동
# 2) health check로 준비 완료 대기
# 3) FastAPI 서버 실행 (foreground)
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
FASTAPI_PORT="${PORT:-7860}"
MODEL="${MODEL_PATH:-LGAI-EXAONE/EXAONE-4.0-32B-AWQ}"
GPU_UTIL="${GPU_UTILIZATION:-0.90}"
MAX_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${MODEL_DTYPE:-half}"
KV_DTYPE="${KV_CACHE_DTYPE:-auto}"
SKIP_MODEL="${SKIP_MODEL_LOAD:-false}"

# SKIP_MODEL_LOAD 시 vLLM 서버 없이 FastAPI만 실행
if [ "$SKIP_MODEL" = "true" ] || [ "$SKIP_MODEL" = "1" ]; then
    echo "[entrypoint] SKIP_MODEL_LOAD=true: FastAPI만 실행"
    exec python3.10 -m src.inference.api_server
fi

# --- vLLM 서버 기동 ---
VLLM_ARGS=(
    --model "$MODEL"
    --port "$VLLM_PORT"
    --host 0.0.0.0
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
    IFS=',' read -ra PAIRS <<< "$ADAPTER_PATHS"
    for pair in "${PAIRS[@]}"; do
        name="${pair%%=*}"
        path="${pair#*=}"
        VLLM_ARGS+=(--lora-modules "${name}=${path}")
    done
fi

echo "[entrypoint] vLLM 서버 기동: port=$VLLM_PORT model=$MODEL"
echo "[entrypoint] args: ${VLLM_ARGS[*]}"

python3.10 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
VLLM_PID=$!

# --- vLLM health check ---
echo "[entrypoint] vLLM 서버 준비 대기 중..."
MAX_WAIT=900  # 최대 15분 (모델 다운로드 + CUDA graph 캡처 포함)
WAITED=0
INTERVAL=5

while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[entrypoint] vLLM 서버 준비 완료 (${WAITED}s)"
        break
    fi
    # vLLM 프로세스가 죽었는지 확인
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[entrypoint] ERROR: vLLM 프로세스 종료됨"
        exit 1
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
echo "[entrypoint] FastAPI 서버 기동: port=$FASTAPI_PORT"
exec python3.10 -m src.inference.api_server
