#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "=== GovOn Smoke Test ==="
echo "대상: $BASE_URL"
echo ""

PASS=0
FAIL=0

# Test 1: Health check
echo -n "[TEST] GET /health ... "
HEALTH_RESPONSE=$(curl -sf "${BASE_URL}/health" 2>/dev/null) || { echo "FAIL (연결 실패)"; FAIL=$((FAIL+1)); }
if [ -n "${HEALTH_RESPONSE:-}" ]; then
    STATUS=$(echo "$HEALTH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
    if [ "$STATUS" = "healthy" ]; then
        echo "PASS"
        PASS=$((PASS+1))
    else
        echo "FAIL (status: ${STATUS:-unknown})"
        FAIL=$((FAIL+1))
    fi
fi

# Test 2: Health response structure
echo -n "[TEST] /health 응답 구조 ... "
if echo "$HEALTH_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'status' in d" 2>/dev/null; then
    echo "PASS"
    PASS=$((PASS+1))
else
    echo "FAIL"
    FAIL=$((FAIL+1))
fi

echo ""
echo "=============================="
echo "결과: PASS=${PASS}, FAIL=${FAIL}"
if [ "$FAIL" -gt 0 ]; then
    echo "상태: FAILED"
    exit 1
else
    echo "상태: PASSED"
    exit 0
fi
