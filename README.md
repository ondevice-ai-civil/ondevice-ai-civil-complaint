# GovOn — AI 민원 답변 도우미

> **"도로 파손 민원에 대한 답변 초안을 작성해줘"** — 한마디면 AI가 법령을 찾고, 유사 사례를 조회하고, 공문서 형식의 초안을 생성합니다.

[![PyPI](https://img.shields.io/pypi/v/govon?logo=pypi&logoColor=white)](https://pypi.org/project/govon/)
[![npm](https://img.shields.io/npm/v/govon?logo=npm)](https://www.npmjs.com/package/govon)
[![Homebrew](https://img.shields.io/badge/brew-govon--org/govon-FBB040?logo=homebrew)](https://github.com/GovOn-Org/homebrew-govon)
[![Python](https://img.shields.io/pypi/pyversions/govon)](https://pypi.org/project/govon/)
[![Docs](https://img.shields.io/badge/Docs-Portal-blue?logo=readthedocs)](https://govon-org.github.io/GovOn/)

<!-- DORA-BADGES:START -->
![DORA Grade](https://img.shields.io/badge/DORA-Elite-brightgreen)
![Deploy Freq](https://img.shields.io/badge/Deploy_Freq-30%2Fweek-blue)
![Lead Time](https://img.shields.io/badge/Lead_Time-0.3h-brightgreen)
![CFR](https://img.shields.io/badge/CFR-29.0%2525-yellow)
![MTTR](https://img.shields.io/badge/MTTR-0.0h-brightgreen)
<!-- DORA-BADGES:END -->

---

## 설치

### CLI 클라이언트 (원격 서버 접속용)

```bash
# pip (권장)
pip install govon

# npm
npm install -g govon

# Homebrew (macOS / Linux)
brew tap govon-org/govon && brew install govon
```

### 로컬 서버 (GPU 환경, 선택)

```bash
# pip extras — CLI + 백엔드 전체 (vLLM, torch, FastAPI, LangGraph)
pip install govon[server]

# 또는 Docker (권장)
govon server pull
govon server start
```

---

## 빠른 시작

### 1. 원격 서버에 접속하기 (가장 쉬운 방법)

```bash
# HF Space 런타임 URL 설정
export GOVON_RUNTIME_URL=https://umyunsang-govon-runtime.hf.space

# CLI 실행
govon
```

```
GovOn Shell — 무엇을 도와드릴까요?

> 도로 파손 민원에 대한 답변 초안을 작성해줘

┌─ 작업 승인 요청 ─────────────────┐
│  유형: 답변 초안 작성              │
│  목표: 도로 파손 민원 답변 생성     │
│  작업:                            │
│   • 민원 처리 근거 확인            │
│   • 유사 사례 및 담당 부서 조회     │
│                                   │
│  ● 승인  ○ 거절                   │
└───────────────────────────────────┘
```

### 2. 단발 실행 (파이프라인 연동)

```bash
# 한 줄 질문 → 답변 → 종료
govon "우리 지역 도로 파손 민원 현황 알려줘"
```

### 3. API 직접 호출

```bash
curl -X POST $GOVON_RUNTIME_URL/v3/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "도로 파손 민원 현황 알려줘",
    "session_id": "demo-1",
    "max_iterations": 10
  }'
```

### 4. 로컬 서버로 실행

```bash
# Docker로 백엔드 시작
govon server pull          # 이미지 다운로드
govon server start         # 백엔드 시작 (docker compose up -d)
govon server status        # 상태 확인

# CLI가 자동으로 localhost에 접속
govon
```

---

## 사용 예시

### 민원 답변 초안 작성

```
> 도로 파손 민원에 대한 답변 초안을 작성해줘

[승인] → AI가 관련 법령 검색 + 유사 사례 조회 + 공문서 형식 초안 생성
```

### 민원 통계 분석

```
> 이번 달 민원 통계와 주요 이슈 분석해줘

→ stats_lookup + issue_detector 자동 호출 → 통계 요약 + 급증 이슈 리포트
```

### 멀티턴 대화

```
> 도로 파손 민원 현황 알려줘
→ (현황 응답)

> 그 중 가장 많은 유형은?
→ (이전 대화 맥락 유지하여 응답)

> 그 유형에 대한 답변 초안을 작성해줘
→ (3턴 맥락을 종합하여 초안 생성)
```

---

## 서버 관리

`govon server` 명령으로 Docker 기반 백엔드를 관리합니다.

| 명령 | 설명 |
|------|------|
| `govon server pull [TAG]` | Docker 이미지 다운로드 |
| `govon server start` | 백엔드 시작 (`docker compose up -d`) |
| `govon server stop` | 백엔드 중지 (`docker compose down`) |
| `govon server status` | 컨테이너 상태 + `/health` 체크 |
| `govon server logs` | 실시간 로그 스트리밍 |

```bash
# GPU 서버에서 전체 스택 실행
govon server pull v1.0.6
govon server start
govon server status

# 로그 확인
govon server logs
```

---

## 도구 목록

AI 에이전트가 자율적으로 선택하는 7개 도구:

| 도구 | 역할 | 예시 질문 |
|------|------|---------|
| `api_lookup` | 민원 데이터 조회 | "우리 구 도로 민원 현황" |
| `issue_detector` | 이슈 탐지 | "최근 급증한 민원 유형" |
| `stats_lookup` | 통계 조회 | "월별 민원 처리 건수" |
| `keyword_analyzer` | 키워드 분석 | "이번 달 키워드 트렌드" |
| `demographics_lookup` | 인구통계 조회 | "해당 지역 인구 구성" |
| `public_admin_adapter` | 민원답변 초안 생성 | "도로 파손 민원 답변 작성" |
| `legal_adapter` | 법률 근거 보강 | "관련 법령과 판례" |

도구 실행 전 **승인 UI**가 표시되어, 사용자가 확인 후 실행합니다.

---

## 아키텍처

```mermaid
graph LR
    subgraph Client ["사용자"]
        CLI["govon CLI<br/>pip / npm / brew"]
    end

    subgraph Server ["서버 (HF Space 또는 Docker)"]
        API["FastAPI"]
        AGENT["ReAct Agent<br/>+ 7 Tools"]
        LLM["EXAONE 4.0-32B<br/>+ civil LoRA<br/>+ legal LoRA"]
    end

    CLI -- "HTTP/SSE" --> API --> AGENT --> LLM

    style Client fill:#e0f2fe,stroke:#0284c7
    style Server fill:#fef3c7,stroke:#d97706
```

**CLI는 가볍고, 서버는 강력합니다.**
- CLI: httpx, rich, prompt-toolkit (~10MB)
- 서버: EXAONE 4.0-32B + vLLM + Multi-LoRA (A100 80GB)

---

## 환경 설정

| 환경변수 | 설명 | 기본값 |
|---------|------|--------|
| `GOVON_RUNTIME_URL` | 서버 URL | `http://localhost:7860` |
| `API_KEY` | API 인증 키 | (없으면 인증 없이 접속) |
| `HOST_PORT` | 로컬 서버 포트 | `8000` |

```bash
# .env 또는 셸에서 설정
export GOVON_RUNTIME_URL=https://umyunsang-govon-runtime.hf.space
export API_KEY=your-api-key
```

---

## 요구 사항

### CLI

- Python 3.10+ 또는 Node.js 18+ 또는 Homebrew
- 인터넷 연결 (원격 서버 접속 시)

### 로컬 서버

- NVIDIA GPU (A100 80GB 권장)
- Docker + NVIDIA Container Toolkit
- 또는 `pip install govon[server]` + CUDA 12.x

---

## 문서

| 문서 | 설명 |
|------|------|
| [사용자 가이드](docs/guide/user-guide.md) | 설치, CLI 사용법, 도구 상세 |
| [운영 가이드](docs/guide/ops-guide.md) | 배포, 환경변수, 모니터링 |
| [데모 패키지](docs/demo/README.md) | 시연 시나리오 3종 |
| [API 레퍼런스](docs/guide/ops-guide.md#api-엔드포인트-레퍼런스) | 전체 엔드포인트 |
| [Docs Portal](https://govon-org.github.io/GovOn/) | 통합 문서 사이트 |

---

## 리소스

| 자원 | 링크 |
|------|------|
| PyPI | [`pip install govon`](https://pypi.org/project/govon/) |
| npm | [`npm install -g govon`](https://www.npmjs.com/package/govon) |
| Homebrew | [`brew tap govon-org/govon`](https://github.com/GovOn-Org/homebrew-govon) |
| HF Space | [umyunsang/govon-runtime](https://huggingface.co/spaces/umyunsang/govon-runtime) |
| GitHub Releases | [v1.0.6](https://github.com/GovOn-Org/GovOn/releases) |
| Docker | `ghcr.io/govon-org/govon` |
| Civil Adapter | [umyunsang/govon-civil-adapter](https://huggingface.co/umyunsang/govon-civil-adapter) |
| Legal Adapter | [siwo/govon-legal-adapter](https://huggingface.co/siwo/govon-legal-adapter) |

---

## 기여

```bash
git clone https://github.com/GovOn-Org/GovOn.git
cd GovOn
pip install -e ".[dev]"
pytest
```

자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

---

## 프로젝트 정보

**동아대학교 컴퓨터공학과** 현장미러형 산학연계 프로젝트

지방자치단체 공무원의 민원 답변 업무를 AI 에이전트가 보조합니다.
AI가 공무원을 대체하는 것이 아니라, 반복 작업을 자동화하여 더 중요한 판단에 집중할 수 있도록 돕습니다.

[![Public Roadmap](https://img.shields.io/badge/Public_Roadmap-Workstreams-7C3AED)](https://github.com/GovOn-Org/GovOn/issues/402)
[![Discussion](https://img.shields.io/badge/Discussion-Community-green?logo=github)](https://github.com/GovOn-Org/GovOn/discussions/606)

---

## 라이선스

MIT License - [LICENSE](LICENSE)
