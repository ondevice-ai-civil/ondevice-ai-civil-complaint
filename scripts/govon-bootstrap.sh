#!/usr/bin/env bash
# GovOn daemon bootstrap script
# Usage: ./scripts/govon-bootstrap.sh [start|stop|status|health]
#
# 환경변수:
#   GOVON_HOME   — GovOn 홈 디렉터리 (기본: ~/.govon)
#   GOVON_PORT   — daemon 포트 (기본: 8000)
#   SKIP_MODEL_LOAD — 모델 로드 건너뛰기 (경고 표시됨)

set -euo pipefail

PYTHON_CMD=""

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
GOVON_HOME="${GOVON_HOME:-$HOME/.govon}"
GOVON_PORT="${GOVON_PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${GOVON_PORT}/health"
PID_FILE="${GOVON_HOME}/daemon.pid"
LOG_FILE="${GOVON_HOME}/daemon.log"

# ---------------------------------------------------------------------------
# 색상 출력 헬퍼
# ---------------------------------------------------------------------------
_info()    { echo "[INFO]  $*"; }
_warn()    { echo "[WARN]  $*" >&2; }
_error()   { echo "[ERROR] $*" >&2; }
_success() { echo "[OK]    $*"; }

# ---------------------------------------------------------------------------
# Pre-flight 검사
# ---------------------------------------------------------------------------
_preflight_checks() {
    # SKIP_MODEL_LOAD 경고
    if [ "${SKIP_MODEL_LOAD:-}" = "true" ] || [ "${SKIP_MODEL_LOAD:-}" = "1" ]; then
        _warn "SKIP_MODEL_LOAD가 설정되어 있습니다. 모델이 로드되지 않으며 일부 기능이 비활성화됩니다."
    fi

    # GPU 감지 경고
    if command -v nvidia-smi &>/dev/null; then
        if ! nvidia-smi &>/dev/null 2>&1; then
            _warn "nvidia-smi 실행에 실패했습니다. GPU를 사용할 수 없을 수 있습니다."
        fi
    else
        _warn "nvidia-smi를 찾을 수 없습니다. CPU 전용 모드로 실행됩니다. (성능이 크게 저하될 수 있습니다)"
    fi
}

# ---------------------------------------------------------------------------
# Python / govon 설치 확인
# ---------------------------------------------------------------------------
_check_python() {
    if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
        _error "Python을 찾을 수 없습니다. Python 3.10 이상을 설치하세요."
        exit 1
    fi
    PYTHON_CMD="$(command -v python3 || command -v python)"
    _info "Python: $("$PYTHON_CMD" --version 2>&1)"
}

_check_govon() {
    # govon CLI 또는 src.cli.shell 모듈 가용 여부 확인
    if command -v govon &>/dev/null; then
        GOVON_CMD="govon"
        _info "govon 명령어 발견: $(command -v govon)"
    elif $PYTHON_CMD -c "import src.cli.shell" 2>/dev/null; then
        GOVON_CMD="$PYTHON_CMD -m src.cli.shell"
        _info "govon 모듈(src.cli.shell) 발견"
    else
        _error "govon이 설치되어 있지 않습니다. 'pip install govon[cli]' 또는 'pip install -e .[cli]'를 실행하세요."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# PID 유틸리티
# ---------------------------------------------------------------------------
_read_pid() {
    if [ -f "$PID_FILE" ]; then
        awk '{print $1}' "$PID_FILE" 2>/dev/null || echo ""
    fi
}

_pid_alive() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# ---------------------------------------------------------------------------
# health 확인
# ---------------------------------------------------------------------------
_health_check() {
    curl -sf --max-time 5 "$HEALTH_URL" &>/dev/null
}

# ---------------------------------------------------------------------------
# 명령: start
# ---------------------------------------------------------------------------
cmd_start() {
    _preflight_checks
    _check_python
    _check_govon

    # 이미 실행 중인지 확인
    local existing_pid
    existing_pid="$(_read_pid)"
    if _pid_alive "$existing_pid" && _health_check; then
        _success "GovOn daemon이 이미 실행 중입니다. (PID=$existing_pid, 포트=$GOVON_PORT)"
        exit 0
    fi

    # ~/.govon 디렉터리 생성
    mkdir -p "$GOVON_HOME"

    _info "GovOn daemon을 시작합니다. (포트=$GOVON_PORT, 로그=$LOG_FILE)"

    # daemon 기동
    if [ "$GOVON_CMD" = "govon" ]; then
        # govon CLI를 통한 기동 (govon --start 지원 시 사용; 없으면 직접 uvicorn 호출)
        if govon --help 2>&1 | grep -q -- "--start" 2>/dev/null; then
            govon --start >> "$LOG_FILE" 2>&1 &
        else
            # 직접 uvicorn으로 기동
            $PYTHON_CMD -m uvicorn src.inference.api_server:app \
                --host 127.0.0.1 \
                --port "$GOVON_PORT" >> "$LOG_FILE" 2>&1 &
        fi
    else
        $PYTHON_CMD -m uvicorn src.inference.api_server:app \
            --host 127.0.0.1 \
            --port "$GOVON_PORT" >> "$LOG_FILE" 2>&1 &
    fi

    local daemon_pid=$!
    echo "$daemon_pid $(date +%s)" > "$PID_FILE"
    _info "daemon PID=$daemon_pid 기록 완료."

    # 빠른 실패 감지: 2초 후 프로세스가 이미 종료되었는지 확인
    sleep 2
    if ! kill -0 "$daemon_pid" 2>/dev/null; then
        _error "daemon이 기동 직후 종료되었습니다. 로그를 확인하세요: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi

    # health check 대기 (최대 120초)
    local elapsed=0
    local max_wait=120
    _info "health check 대기 중..."
    while [ $elapsed -lt $max_wait ]; do
        if _health_check; then
            _success "GovOn daemon 기동 완료. (PID=$daemon_pid, 포트=$GOVON_PORT)"
            exit 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    _error "health check timeout (${max_wait}s). 로그를 확인하세요: $LOG_FILE"
    exit 1
}

# ---------------------------------------------------------------------------
# 명령: stop
# ---------------------------------------------------------------------------
cmd_stop() {
    local pid
    pid="$(_read_pid)"

    if [ -z "$pid" ]; then
        _warn "PID 파일이 없습니다. daemon이 실행 중이 아닌 것으로 간주합니다."
        exit 0
    fi

    if ! _pid_alive "$pid"; then
        _warn "PID=$pid 프로세스가 없습니다. PID 파일을 제거합니다."
        rm -f "$PID_FILE"
        exit 0
    fi

    # govon CLI --stop 지원 여부 확인
    if command -v govon &>/dev/null && govon --help 2>&1 | grep -q -- "--stop" 2>/dev/null; then
        govon --stop
    else
        _info "SIGTERM 전송: PID=$pid"
        kill -TERM "$pid"

        local elapsed=0
        while [ $elapsed -lt 10 ]; do
            if ! _pid_alive "$pid"; then
                _success "GovOn daemon이 정상 종료되었습니다. (PID=$pid)"
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
            elapsed=$((elapsed + 1))
        done

        _warn "timeout — SIGKILL 전송: PID=$pid"
        kill -KILL "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        _success "GovOn daemon이 강제 종료되었습니다. (PID=$pid)"
    fi
}

# ---------------------------------------------------------------------------
# 명령: status
# ---------------------------------------------------------------------------
cmd_status() {
    local pid
    pid="$(_read_pid)"

    if [ -z "$pid" ]; then
        echo "GovOn daemon: 중지됨 (PID 파일 없음)"
        exit 1
    fi

    if ! _pid_alive "$pid"; then
        echo "GovOn daemon: 중지됨 (PID=$pid — 프로세스 없음)"
        rm -f "$PID_FILE"
        exit 1
    fi

    if _health_check; then
        echo "GovOn daemon: 실행 중 (PID=$pid, 포트=$GOVON_PORT)"
        exit 0
    else
        echo "GovOn daemon: 프로세스는 살아 있지만 health check 실패 (PID=$pid, URL=$HEALTH_URL)"
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# 명령: health
# ---------------------------------------------------------------------------
cmd_health() {
    _info "GET $HEALTH_URL"
    if curl -sf --max-time 10 "$HEALTH_URL"; then
        echo ""
        _success "health check 통과."
        exit 0
    else
        _error "health check 실패. daemon이 실행 중인지 확인하세요."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
COMMAND="${1:-help}"

case "$COMMAND" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    health)
        cmd_health
        ;;
    help|--help|-h)
        echo "사용법: $0 [start|stop|status|health]"
        echo ""
        echo "명령어:"
        echo "  start   — GovOn daemon을 기동합니다"
        echo "  stop    — GovOn daemon을 중지합니다"
        echo "  status  — daemon 실행 상태를 확인합니다"
        echo "  health  — /health 엔드포인트를 probe합니다"
        echo ""
        echo "환경변수:"
        echo "  GOVON_HOME=$GOVON_HOME"
        echo "  GOVON_PORT=$GOVON_PORT"
        echo "  SKIP_MODEL_LOAD (설정 시 경고 표시)"
        exit 0
        ;;
    *)
        _error "알 수 없는 명령: $COMMAND"
        echo "사용법: $0 [start|stop|status|health]"
        exit 1
        ;;
esac
