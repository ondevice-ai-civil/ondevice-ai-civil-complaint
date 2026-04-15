# GovOn 발표자료 시각화 프롬프트

> 생성 일자: 2026-04-15  
> 분석 기준: `main` @ `de5622fe1`  
> 용도: 발표자료용 아키텍처/플로우 다이어그램 생성 (Mermaid, Excalidraw, Figma, draw.io, Gamma 등)

---

## ▶ 사용 방법

이 문서는 **하나의 대형 프롬프트 패키지**입니다. 아래 각 섹션을 통째로 복사하여 이미지 생성·다이어그램 AI(예: Claude, GPT-4o, Gamma, Napkin AI, Eraser.io, Mermaid Chart AI)에 전달하면, 발표자료용 시각 자료가 생성됩니다.

- **섹션 0 (SYSTEM)**: 모든 다이어그램 공통 지침
- **섹션 1~9**: 각 다이어그램별 독립 프롬프트 (원하는 것만 사용)

---

## 0. SYSTEM PROMPT (공통 지침)

```
You are a senior information designer creating slides for a technical presentation
about "GovOn" — an Agentic Transformation (AX) platform that unifies Korea's
fragmented government DX infrastructure through a central LLM agent that wraps
agency APIs as LLM-callable tools.

AUDIENCE: Technical reviewers (professors, engineers), non-technical stakeholders
          (public sector decision makers). Bilingual KO/EN labels acceptable;
          prefer concise English labels with Korean sub-captions for Korean terms.

STYLE CONSTRAINTS:
- Clean, modern, minimalist. Flat design, no 3D, no skeuomorphism.
- Color palette: deep navy (#0F2A44) primary, soft cyan (#2FB6B0) accent,
  warm amber (#E8A94B) for highlights, neutral grays for infrastructure.
- Use icons from Lucide / Heroicons / Tabler when representing services
  (database, brain, terminal, shield, graph, cloud).
- Rounded rectangles (radius 8-12px), 1.5px stroke, subtle drop shadow only
  on foreground elements.
- Typography: sans-serif (Inter / Pretendard). Labels ≤ 4 words when possible.
- Arrows: thin (1.5px), rounded arrowheads. Solid = synchronous/data flow,
  dashed = async/event/SSE, dotted = optional/conditional.
- Callouts: numbered circles (①②③) for sequencing.
- NEVER use emoji in production diagrams. NEVER write Claude/Anthropic attribution.

OUTPUT FORMAT:
- Default: Mermaid syntax (graph TD / sequenceDiagram / flowchart LR).
- If Mermaid cannot express it cleanly, output Excalidraw JSON or a Figma-style
  structured description with absolute coordinates.
- Always include a short "legend" block explaining symbols.

FACTUAL CONSTRAINTS:
- Every component MUST come from the spec blocks below. Do NOT invent modules.
- If a detail is missing, write "[unspecified]" rather than hallucinate.
- File paths and line numbers in parentheses are optional; include them only
  when they aid the technical audience.
```

---

## 1. 🏛️ 전체 시스템 아키텍처 (High-Level Architecture)

```
TASK: Draw a high-level system architecture diagram for GovOn.

LAYOUT: Horizontal, 4 layers top→bottom.

LAYERS & COMPONENTS:

1) CLIENT LAYER (top)
   - GovOn CLI (React + Ink TUI, Node.js 18+, ~10MB bundle)
     └─ distribution: npm, Homebrew, native SEA binary
   - Next.js Web UI (Next 16.2.1 + React 19.2.4 + Tailwind 4) — "R1+ roadmap"

2) API GATEWAY LAYER
   - FastAPI server (src/inference/api_server.py)
     - Endpoints: /health, /v3/agent/run, /v3/agent/stream (SSE),
       /v3/agent/approve, /v3/agent/cancel, /v2/agent/stream (legacy)
     - Auth: X-API-Key header (slowapi rate limit 30/min)

3) AGENT CORE LAYER (center, largest block)
   - LangGraph ReAct v3 Engine (src/inference/graph/)
     Nodes: session_load → agent → [tools ⇄ agent loop] → synthesize
   - Session Store: SQLite (~/.govon/sessions.sqlite3, schema v2)
   - Adapter Registry (config/adapters.yaml)
   - Tool Registry (7 tools, see diagram §3)
   - Runtime Config (config/govon.yaml, 4 profiles: local/single/container/airgap)

4) INFERENCE & DATA LAYER (bottom)
   - vLLM server (OpenAI-compatible, port 8000)
     Model: EXAONE 4.0-32B-AWQ + Multi-LoRA (max 4 adapters, rank 64)
   - LoRA Adapters (HuggingFace Hub):
     • public_admin → umyunsang/govon-civil-adapter
     • legal        → siwo/govon-legal-adapter
   - External APIs: 공공데이터포털 (Korean Open Data Portal)
   - Vector Store: FAISS-CPU + BM25 (rank-bm25)

INFRASTRUCTURE SIDEBAR (right):
   - Docker image (nvidia/cuda:12.6.3, Python 3.10)
   - Deploy targets: HuggingFace Spaces (A100 80GB), Google Cloud Run,
     on-prem (Compose), airgap
   - Observability: Grafana Cloud (DORA dashboard — umyunsang.grafana.net/d/govon-dora/)
   - CI/CD: GitHub Actions (ci.yml, deploy-hfspace.yml, docker-publish.yml,
     dora-metrics.yml)

ANNOTATIONS:
   - Tag the Agent Core layer "ReAct v3 (LangGraph 1.1.6)"
   - Show "SSE stream" between API Gateway and Client
   - Show "HTTP+JSON" between Agent Core and vLLM

RENDER AS: Mermaid `graph TD` with subgraphs per layer.
```

---

## 2. 🧠 LangGraph ReAct v3 그래프 토폴로지

```
TASK: Draw the LangGraph state machine for GovOn's ReAct v3 agent.

NODES (rounded rectangles):
  START (circle)
  session_load        — "Load multi-turn history; trim + extractive summary
                        when tokens > 60% of 4500 budget; keep recent 6 msgs"
                        (src/inference/graph/nodes.py:191-249)
  agent               — "LLM ReAct loop; emits tool_calls; bound to 7 tools"
                        (make_agent_node_v3)
  tools               — "LangGraph ToolNode; executes all pending_tool_calls;
                        appends ToolMessages; iteration_count++"
                        (make_tools_node_v3)
  synthesize          — "Final answer + evidence items + SQLite persist"
                        (make_persist_node)
  END (circle)

EDGES:
  START       → session_load
  session_load → agent
  agent       → route_agent_v3 (diamond / decision node)
  route_agent_v3 --[pending_tool_calls.length > 0
                    AND iteration_count < max_iterations (=10)]--> tools
  tools       → agent           (loop back — label "ReAct cycle")
  route_agent_v3 --[else]-----> synthesize
  synthesize  → END

STATE BADGE (sidebar, show GovOnGraphState fields):
  session_id, request_id, messages (Annotated[...add_messages]),
  pending_tool_calls, tool_call_history, iteration_count, max_iterations=10,
  approval_status, final_text, evidence_items, node_latencies

CONTEXT MANAGEMENT CALLOUT (attach to session_load):
  • _MAX_MESSAGE_TOKENS = 4500
  • _KEEP_RECENT = 6
  • _SUMMARY_THRESHOLD_RATIO = 0.6
  • _TOOL_CLEAR_AFTER_ITERATION = 2
  • _TOOL_KEEP_RECENT = 2

RENDER AS: Mermaid `stateDiagram-v2` OR `flowchart TD` with a conditional diamond.
```

---

## 3. 🧰 도구 생태계 (Tool Registry)

```
TASK: Create a "tool catalog" diagram showing all 7 GovOn tools grouped by category.

GROUP 1 — SEARCH (src/inference/graph/tools/search_tools.py)
  • api_lookup — 공공데이터포털 민원분석정보 조회, timeout 10s, retry 1
                 (requires_approval: false)

GROUP 2 — ANALYSIS (src/inference/graph/tools/analysis_tools.py)
  • issue_detector        — Trending issue detection (timeout 15s)
  • stats_lookup          — Domain statistics lookup
  • keyword_analyzer      — Keyword frequency analysis
  • demographics_lookup   — Regional demographic lookup

GROUP 3 — ADAPTER / LoRA (src/inference/graph/tools/adapter_tools.py)
  • public_admin_adapter  — Administrative draft generation
                            (LoRA id=1, requires_approval: true) 🛡
  • legal_adapter         — Legal interpretation + citations
                            (LoRA id=2, requires_approval: true) 🛡

VISUAL:
  - Three vertical columns, one per group.
  - Each tool = card with: icon, name (mono font), 1-line purpose, badges
    (timeout, approval-required, LoRA-id).
  - Approval-required tools get a shield badge and amber border.
  - Top-bar: "Dynamic tool registration" note — adapter_tools.py reads
    config/adapters.yaml at build time; changing YAML adds/removes tools
    without code changes.

RENDER AS: Mermaid `graph LR` with subgraphs, OR Excalidraw-style card layout.
```

---

## 4. 🔄 E2E 요청 플로우 (Sequence Diagram)

```
TASK: Draw a sequence diagram for a single user turn that triggers tool calls.

ACTORS (columns, left→right):
  User
  CLI (Ink TUI, packages/npm/src/App.tsx)
  Client SDK (packages/npm/src/client.ts)
  FastAPI (src/inference/api_server.py)
  LangGraph (builder.py)
  ToolNode
  External (공공데이터포털 API)
  vLLM (EXAONE + LoRA)
  SQLite (SessionStore)

EXAMPLE QUERY:
  "도로 파손 민원에 대한 행정 대응 절차"

MESSAGES (numbered):
  ① User types query → CLI handleSubmit()
  ② CLI → Client SDK: streamV3(query, session_id, max_iterations=10)
  ③ Client → FastAPI: POST /v3/agent/stream (SSE, X-API-Key)
  ④ FastAPI → LangGraph: graph.ainvoke({...})
  ⑤ LangGraph → SQLite: SessionStore.load(session_id) — returns prior messages
  ⑥ LangGraph (session_load): trim + summarize if over budget
  ⑦ LangGraph (agent) → vLLM: chat/completions (tools bound)
  ⑧ vLLM → LangGraph: AIMessage{tool_calls=[api_lookup, public_admin_adapter]}
  ⑨ LangGraph (tools) → External: GET 공공데이터포털 /complaint-analysis
  ⑩ External → LangGraph: JSON results
  ⑪ LangGraph (tools) → vLLM: adapter inference (LoRA=public_admin)
  ⑫ vLLM → LangGraph: draft response
  ⑬ LangGraph (agent, iter 2) → vLLM: final synthesis
  ⑭ vLLM → LangGraph: AIMessage (no tool_calls)
  ⑮ LangGraph (synthesize) → SQLite: persist turn + graph_run
  ⑯ FastAPI → Client: SSE events stream
       thinking_start → thinking_delta* → thinking_end
       → tool_start/tool_end (x2) → response_delta*
       → run_complete{text, evidence_items, metadata}
  ⑰ Client → CLI: dispatch reducer actions
  ⑱ CLI renders: Static scrollback + ThinkingBlock + ToolPanel + MarkdownView

STYLE:
  - Activation bars on each actor while processing.
  - Dashed arrows for SSE events (async).
  - Solid arrows for sync HTTP.
  - Group the ReAct loop (steps ⑦-⑫) in a `loop` frame with label
    "ReAct cycle — iteration_count < max_iterations (=10)".

RENDER AS: Mermaid `sequenceDiagram`.
```

---

## 5. 📡 SSE 이벤트 스트림 타임라인

```
TASK: Draw a horizontal time-axis diagram showing the SSE event stream
      a CLI receives for one query that triggers 2 tool calls over 2 iterations.

TIME AXIS: left→right, marks at 0s, 0.5s, 1.2s, 2.1s, 3.0s, 3.5s (illustrative).

SWIMLANES (rows):
  Row 1 — "Thinking" events (blue)
  Row 2 — "Tool" events (amber)
  Row 3 — "Response" events (green)
  Row 4 — "Terminal" event (navy)

EVENTS IN ORDER (from types.ts:51-99):
  t=0.00  thinking_start { iteration: 1 }
  t=0.1~  thinking_delta { content: "..." } × N
  t=0.50  thinking_end { tool_calls: [api_lookup, public_admin_adapter], iteration: 1 }
  t=0.55  tool_start { tool: "api_lookup" }
  t=1.15  tool_end   { tool: "api_lookup", success: true }
  t=1.20  tool_start { tool: "public_admin_adapter" }
  t=2.05  tool_end   { tool: "public_admin_adapter", success: true }
  t=2.10  thinking_start { iteration: 2 }
  t=2.2~  thinking_delta × N
  t=2.80  response_delta { content: "..." } × N
  t=3.50  run_complete { text, evidence_items, metadata: {total_iterations:2} }

VISUAL HINTS:
  - Render deltas as rapid-fire small ticks (not individual blocks).
  - Box the "Iteration 1" and "Iteration 2" regions distinctly.
  - Add callout on run_complete showing final metadata keys.

RENDER AS: Mermaid `gantt` OR custom SVG timeline.
```

---

## 6. 🤝 Human-in-the-Loop 승인 플로우

```
TASK: Flowchart for tool-call approval when a requires_approval tool is requested.

START:
  Agent emits tool_calls containing requires_approval=true tool
  (e.g., legal_adapter — flagged in config/adapters.yaml)

NODES:
  1. Detect approval-required tool → route to approval_wait node
  2. LangGraph `interrupt()` — pauses execution, persists checkpoint
  3. Server emits SSE event: approval_request { session_id, thread_id, tools[] }
  4. CLI shows ApprovalPrompt.tsx:
       "Type: Draft response
        Goal: Legal interpretation
        Tools: legal_adapter
        [Approve] [Reject] [Cancel]"
  5. User decision → POST /v3/agent/approve { session_id, thread_id, decision }
  6. Decision branching (diamond):
       approved  → resume graph → tools node (execute)
       rejected  → resume graph → agent node (re-think without this tool)
       cancelled → resume graph → synthesize node (graceful termination)

STATE OVERLAY:
  approval_status: "pending" | "approved" | "rejected" | "cancelled"

RENDER AS: Mermaid `flowchart TD` with diamond for decision branching.
          Use dashed arrows for SSE and HTTP callbacks.
```

---

## 7. 🎓 LoRA 학습 파이프라인

```
TASK: Horizontal pipeline diagram for the domain LoRA training workflow
      (reference: training/legal_adapter/, experiment_results.md).

STAGES (left → right, 6 boxes):

  [1] Data Source
      - AI Hub 71841 (civil-law QA)
      - AI Hub 71843 (IP-law QA)
      - AI Hub 71848 (criminal-law QA)
      - Total: 270K QA pairs

  [2] Preprocessing (src/data_collection_preprocessing/)
      - pipeline.py → parsers.py → KoNLPy (mecab) tokenization
      - Upload to HF Datasets: umyunsang/govon-legal-response-data

  [3] Training Environment
      - HuggingFace Spaces, A100 SXM4 80GB
      - Script: training/legal_adapter/train_qlora.py
      - Framework: Unsloth + TRL SFTTrainer
      - Quantization: 4-bit NF4 on EXAONE 4.0-32B
      - LoRA: rank=16, alpha=32, 7 target modules
              (q, k, v, o, gate, up, down)
      - Precision: BF16 + TF32
      - Batch: 8 × gradient_acc 8 = 64 effective

  [4] Training Run
      - 365 steps, ~7 hours
      - Loss: 2.334 → 0.889

  [5] Artifact
      - Adapter weights pushed to HF: siwo/govon-legal-adapter
      - Registered in config/adapters.yaml with lora_id=2

  [6] Runtime Loading
      - vLLM --enable-lora --max-loras=4 --max-lora-rank=64
      - Adapter Registry loads on API server start
      - Invoked via legal_adapter tool with requires_approval=true

RENDER AS: Mermaid `flowchart LR` with 6 rectangles connected by arrows.
           Add a datasets/weights cloud icon for HF Hub.
```

---

## 8. 🚢 CI/CD & 배포 파이프라인

```
TASK: Draw the end-to-end CI/CD and deploy pipeline.

SOURCES (left):
  Developer → PR → `main` branch (single-trunk strategy, no develop)

GITHUB ACTIONS WORKFLOWS (parallel lanes):
  • ci.yml — PR gate
      detect-changes (path filter) →
      ├─ test-inference (pytest + coverage)
      ├─ test-npm (vitest, packages/npm)
      ├─ docker-build smoke
      └─ e2e (Playwright, optional)
  • docker-publish.yml — on push to main → GHCR/ECR image push
  • deploy-hfspace.yml — on push to main → HF Space sync
                         (umyunsang/govon-runtime, A100)
  • deploy-cloud-run.yml — on release → Google Cloud Run
  • publish-npm.yml — on release → npm registry (govon-cli)
  • compose-smoke.yml — manual/PR → docker-compose verify
  • dora-metrics.yml — daily cron 00:00 UTC (09:00 KST)
      └─ metrics/scripts/generate_report.py →
         weekly-YYYYMMDD.md + latest-dora.html + PNG →
         push to Grafana Cloud (Prometheus line protocol)

DEPLOY TARGETS (right):
  • HuggingFace Space (A100 80GB) — primary runtime
  • Google Cloud Run — managed container host
  • On-prem Docker Compose — deploy/compose/docker-compose.prod.yml
  • Airgap deployment — offline image bundle

OBSERVABILITY (bottom):
  Grafana Cloud → DORA dashboard (umyunsang.grafana.net/d/govon-dora/)
  Current status badges (README):
    Deploy Freq 30/wk (Elite) • Lead Time 0.1h (Elite)
    Change Fail 29.6% (High) • MTTR 0.0h (Elite)

RENDER AS: Mermaid `flowchart LR` with 3 column bands (source → CI → deploy),
           each workflow as its own rounded rectangle with the trigger annotated
           on the incoming edge.
```

---

## 9. 📦 구성(Config) 단일 진리원(Single Source of Truth)

```
TASK: Diagram showing how two YAMLs govern the entire runtime.

CENTER PIECE (large):
  config/govon.yaml — "Unified hyperparameter spec"
    generation.{max_tokens, temperature, agent_temperature}
    serving.{gpu_memory_utilization, max_model_len, max_loras, max_lora_rank}
    context.{agent_input_budget, keep_recent_messages, summary_threshold_ratio,
             max_iterations}
    tools.defaults.{timeout_sec, max_retries}
    tools.overrides.{api_lookup, issue_detector, ...}
    rate_limit.default

  config/adapters.yaml — "Domain adapter registry"
    adapters.public_admin.{path, domain, lora_id=1, requires_approval=true,
                           tool_description}
    adapters.legal.{path=siwo/govon-legal-adapter, lora_id=2,
                    requires_approval=true, tool_description}

OVERRIDE MECHANISM (top):
  Environment variables: GOVON_<SECTION>_<KEY>
  e.g., GOVON_GENERATION_MAX_TOKENS=1024
        GOVON_CONTEXT_MAX_ITERATIONS=15

CONSUMERS (spokes out):
  • runtime_config.py → profile resolution (local/single/container/airgap)
  • adapter_registry.py → loads adapters + builds dynamic tools
  • graph/nodes.py → reads context budget, summary threshold, tool-clear rules
  • graph/tools/__init__.py → applies tool timeouts & retries
  • vLLM entrypoint.sh → passes serving.* flags to `vllm.entrypoints.openai.api_server`
  • FastAPI api_server.py → applies rate_limit & API_KEY

STYLE:
  - Central hub-and-spoke diagram.
  - Two YAML files as central gears.
  - Each consumer as a labeled outer node with arrow pointing inward.

RENDER AS: Mermaid `flowchart` with central subgraph or Excalidraw radial layout.
```

---

## 10. 🧬 개념 스택: "무엇이 GovOn을 가능하게 하는가"

```
TASK: Conceptual "stack sandwich" slide showing how open components compose
      into the GovOn value proposition.

LAYERS (bottom → top, each a wide horizontal band):

  ▲ VALUE — "Unified Agentic Interface to Korean Government DX"
     (single chat, cross-agency tool chaining, domain-tuned responses,
      human-in-the-loop safety)

  │ EXPERIENCE — GovOn CLI (Ink TUI) + Next.js Web UI
               SSE streaming, approval UX, multi-turn session

  │ AGENT RUNTIME — LangGraph 1.1.6 ReAct v3
                   (7 tools, Multi-LoRA dispatch, context management,
                    SQLite checkpointing)

  │ MODEL — EXAONE 4.0-32B-AWQ + domain LoRA adapters
           (public_admin, legal — 270K QA trained, Unsloth+TRL)

  │ INFERENCE — vLLM (OpenAI-compatible, Multi-LoRA serving,
                     A100 80GB, hermes tool-call parser)

  │ DATA — Korean Open Data Portal API, AI Hub datasets,
          FAISS + BM25 hybrid retrieval, KoNLPy preprocessing

  │ PLATFORM — Docker / CUDA 12.6 / Python 3.10 / Node 18+
              HuggingFace Spaces, Google Cloud Run, on-prem Compose

  ▼ OBSERVABILITY — GitHub Actions CI/CD, Grafana Cloud DORA dashboard,
                    public transparency (MIT-licensed source)

STYLE:
  - Each band has a left-side icon and a right-side short tagline.
  - Color gradient from warm (top, user-facing) to cool (bottom, infrastructure).
  - Use this as the "elevator pitch" slide.

RENDER AS: Vertical stacked-band diagram (Figma or Excalidraw).
```

---

## 📎 부록 — 다이어그램 제작 시 빠른 참조 치트시트

| 숫자 | 의미 |
|---|---|
| 7 | 도구 개수 (api_lookup, issue_detector, stats_lookup, keyword_analyzer, demographics_lookup, public_admin_adapter, legal_adapter) |
| 10 | ReAct 최대 iteration (`config/govon.yaml: context.max_iterations`) |
| 4500 | 메시지 토큰 예산 (`_MAX_MESSAGE_TOKENS`) |
| 6 | 최근 메시지 항상 유지 개수 (`_KEEP_RECENT`) |
| 0.6 | 요약 트리거 비율 (`_SUMMARY_THRESHOLD_RATIO`) |
| 2 | iteration ≥ 2 이후 구형 ToolMessage clear (`_TOOL_CLEAR_AFTER_ITERATION`) |
| 32B | EXAONE 4.0 파라미터 수 (AWQ 4-bit) |
| 4 | 동시 LoRA 어댑터 최대 (`max_loras`) |
| 64 | LoRA rank 최대 (`max_lora_rank`) |
| 8192 | max_model_len (single/container/airgap 프로필) |
| 0.90 | GPU memory utilization (production) |
| 30/wk | 현재 배포 빈도 (DORA Elite) |
| 0.1h | 현재 리드 타임 (DORA Elite) |
| 270K | 법률 LoRA 학습 데이터 크기 (AI Hub 71841+71843+71848) |
| 365 | 법률 LoRA 학습 steps (~7h on A100) |
| 2.334 → 0.889 | 법률 LoRA 학습 loss 변화 |

| 경로 | 역할 |
|---|---|
| `src/inference/api_server.py` | FastAPI 엔드포인트 전체 |
| `src/inference/graph/builder.py` | ReAct v2/v3 그래프 생성 |
| `src/inference/graph/state.py` | `GovOnGraphState` 정의 |
| `src/inference/graph/nodes.py` | session_load/agent/tools/synthesize 노드 |
| `src/inference/graph/tools/__init__.py` | `build_all_tools()` 동적 도구 조립 |
| `src/inference/adapter_registry.py` | LoRA 어댑터 레지스트리 |
| `src/inference/session_context.py` | SQLite 세션 저장소 (schema v2) |
| `src/inference/runtime_config.py` | 4개 프로필 + 환경변수 오버라이드 |
| `packages/npm/src/App.tsx` | Ink TUI 메인 컴포넌트 |
| `packages/npm/src/client.ts` | SSE 스트리밍 클라이언트 |
| `packages/npm/src/types.ts` | V3SSEEvent 등 타입 정의 |
| `config/govon.yaml` | 통합 하이퍼파라미터 |
| `config/adapters.yaml` | 도메인 어댑터 메타데이터 |
| `deploy/docker/Dockerfile` | CUDA 12.6 + Python 3.10 이미지 |
| `metrics/scripts/generate_report.py` | DORA 보고서 생성 |
| `.github/workflows/ci.yml` | PR 게이트 |
| `.github/workflows/dora-metrics.yml` | 일일 DORA 수집 |
| `training/legal_adapter/train_qlora.py` | 법률 어댑터 학습 스크립트 |
