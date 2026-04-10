# Mock Test Prompts

Offline mock mode for TUI development without a running backend.

```bash
cd packages/npm
npm run build && npm run dev:mock
```

## Prompts

| Prompt | Scenario | UI Elements | SSE Path |
|--------|----------|-------------|----------|
| `민원 통계` | v3 normal streaming | ThinkingBlock(2 iter) → ToolPanel(stats_lookup ✦, keyword_analyzer ✦) → answer stream → MetadataBar | v3 → `run_complete` |
| `승인 테스트` | v2 approval flow | v3 throw → v2 fallback → ApprovalPrompt box (y/n) → approve response | v3 fail → v2 → `approval_wait` |
| `에러 테스트` | server error | ThinkingBlock mid-stream → red error message | v3 → `error` event |
| `도구 실패` | partial tool failure | stats_lookup ✦ ok → demographics_lookup ✘ fail (red) → partial answer | v3 → `tool_end` success=false |
| `간단 질문` | direct answer, no tools | ThinkingBlock(1 iter, "No tools needed") → answer stream only | v3 → thinking → response |
| `긴 답변` | long markdown stress test | 3 tools → long report (tables, code blocks, lists) streaming | v3 → 3 tools → long stream |

Any input not matching the above keywords falls back to the `민원 통계` scenario.

## Matching Rules

```
query.includes('승인')           → approval
query.includes('에러' | '오류')  → error
query.includes('실패')           → tool_fail
query.includes('간단')           → simple
query.includes('긴 답변' | '길게') → long
(default)                        → stats
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOVON_MOCK=true` | Enable mock mode (required) |
| `GOVON_RUNTIME_URL` | Ignored in mock mode |
