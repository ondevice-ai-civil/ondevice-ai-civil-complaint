# GovOn - 온프레미스 AI 기반 민원 처리 인프라

> 일선 공무원의 업무 부담 최소화 및 국가 정보 보안의 완벽한 보장

[![DORA Dashboard](https://img.shields.io/badge/DORA_Dashboard-Grafana-F46800?logo=grafana)](https://umyunsang.grafana.net/public-dashboards/a7672d6682fb498eb4578a8634262280)
[![W&B Projects](https://img.shields.io/badge/W%26B_Projects-All_Experiments-FFBE00?logo=weightsandbiases)](https://wandb.ai/umyun3/projects)
[![W&B Reports](https://img.shields.io/badge/W%26B_Reports-Analysis-EE6C4D?logo=weightsandbiases)](https://wandb.ai/umyun3/reports)
[![Docs Portal](https://img.shields.io/badge/Docs-Portal-blue?logo=readthedocs)](https://govon-org.github.io/GovOn/)

---

## 왜 GovOn인가

공공기관 민원 처리 현장에는 두 가지 핵심 문제가 있다.

**키워드 기반 오분류** -- 기존 시스템은 단순 키워드 매칭으로 민원을 분류한다. "도로 파손"이 "시설파손"으로, "배수" 민원이 산술 관련으로 오분류되는 일이 반복된다. 오분류된 민원은 담당 부서가 바뀌어 처리가 지연되고 이중 업무가 발생한다.

**비효율적 수작업 프로세스** -- 담당 공무원 1인이 민원 접수부터 분류, 유사 사례 검색, 답변 작성까지 전 과정을 수작업으로 처리한다. 243개 지자체 각각에서 이 비효율이 매일 반복된다.

GovOn은 한국어 특화 LLM(EXAONE-Deep-7.8B)을 QLoRA로 파인튜닝하고 AWQ로 양자화하여, 온프레미스 환경에서 민원 자동 분류, 유사 사례 검색(RAG), 답변 초안 생성을 수행하는 공공기관 내부 업무 시스템이다.

```
AS-IS: 키워드 매칭 → 오분류 반복 → 수작업 답변 → 처리 지연
TO-BE: LLM 문맥 분류 → RAG 기반 유사 사례 검색 → AI 답변 초안 생성 → 공무원 최종 확인
```

---

## 핵심 아키텍처

4단계 기술 파이프라인으로 구성된다.

```
┌─────────────────────────────────────────────────────────────┐
│  1. 온프레미스 LLM   EXAONE-Deep-7.8B (한국어 특화)         │
├─────────────────────────────────────────────────────────────┤
│  2. 파인튜닝         QLoRA (PEFT) + AI Hub 민원 데이터      │
├─────────────────────────────────────────────────────────────┤
│  3. 양자화           AWQ INT4 (14.56GB → 4.94GB, -66.1%)   │
├─────────────────────────────────────────────────────────────┤
│  4. RAG              FAISS + BM25 하이브리드 검색            │
└─────────────────────────────────────────────────────────────┘
```

### 하이브리드 전략

GovOn은 단순한 "반클라우드" 시스템이 아니다. **사건별 핵심부는 온프레미스**, **공개/공통 데이터는 정책에 맞춰 클라우드/범정부 AI 공통기반과 연계 가능한 하이브리드 구조**를 목표로 한다.

| 데이터 유형 | 배치 전략 |
|------------|----------|
| 민원 원문, 상담이력, 내부 처리 로그 | 온프레미스 고정 |
| 공개 법령, 지침, 표준 매뉴얼 | 클라우드/공통기반 연계 가능 |

> 정책 근거: 국가인공지능전략위원회, "대한민국 인공지능 행동계획(2026~2028)" 87쪽, 90쪽, 92쪽

---

## 빠른 시작

### Docker 배포 (권장)

```bash
# GHCR에서 이미지 Pull
docker pull ghcr.io/govon-org/govon:latest

# 환경변수 설정
export API_KEY=your-api-key
export MODEL_PATH=umyunsang/GovOn-EXAONE-LoRA-v2

# 볼륨 디렉토리 생성
mkdir -p models data agents configs

# 실행
docker compose -f docker-compose.offline.yml up -d

# 헬스체크
curl http://localhost:8000/health
```

### 개발 환경

```bash
git clone https://github.com/GovOn-org/GovOn.git
cd GovOn

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"

# 추론 서버 실행
uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000 --reload

# 테스트
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| **AI 모델** | EXAONE-Deep-7.8B (LG AI Research) |
| **파인튜닝** | QLoRA (PEFT, SFTTrainer, WandB) |
| **양자화** | AWQ INT4 (AutoAWQ) |
| **LLM 서빙** | vLLM (PagedAttention) |
| **임베딩** | multilingual-e5-large (1024차원) |
| **벡터 검색** | FAISS (IndexFlatIP) + BM25 하이브리드 |
| **백엔드** | FastAPI + Pydantic + SQLAlchemy |
| **프론트엔드** | React / Next.js + TypeScript |
| **컨테이너** | Docker Compose + NVIDIA Container Toolkit |
| **CI/CD** | GitHub Actions (CI, Docker Publish, Offline Package) |
| **모니터링** | DORA Metrics + Grafana Cloud |

---

## 프로젝트 구조

```
GovOn/
├── src/
│   ├── data_collection_preprocessing/   # AI Hub 수집 → PII 마스킹 → EXAONE 형식 변환
│   ├── training/                        # QLoRA 파인튜닝 (SFTTrainer, WandB 연동)
│   ├── quantization/                    # AWQ 양자화 (W4A16g128), LoRA 병합
│   ├── inference/                       # FastAPI 서빙 (핵심 모듈)
│   │   ├── api_server.py               # vLLMEngineManager, 엔드포인트, 보안 미들웨어
│   │   ├── retriever.py                # FAISS IndexFlatIP + multilingual-e5-large 임베딩
│   │   ├── index_manager.py            # MultiIndexManager (CASE/LAW/MANUAL/NOTICE)
│   │   ├── schemas.py                  # Pydantic 요청/응답 모델
│   │   ├── vllm_stabilizer.py          # EXAONE용 transformers 런타임 패치
│   │   └── db/                         # SQLAlchemy ORM, Alembic 마이그레이션
│   └── evaluation/                     # 모델 평가 스크립트
├── agents/                              # 에이전트 설정
├── configs/                             # 시스템 설정 파일
├── data/                                # 학습/검색 데이터
├── models/                              # 모델 파일, FAISS 인덱스
├── notebooks/                           # 실험 노트북
├── tests/                               # 테스트 코드
├── site/                                # 문서 포털 (MkDocs)
├── docs/                                # 프로젝트 문서 (PRD, WBS, 공식 문서)
├── scripts/                             # 배포 스크립트
├── Dockerfile                           # CUDA 12.1 + Python 3.10
├── docker-compose.yml                   # 온라인 빌드/실행
└── docker-compose.offline.yml           # 오프라인 GHCR 이미지 실행
```

---

## 성과 지표

### KPI 목표 및 현재 달성 현황

| 지표 | 목표 | 현재 결과 | 상태 |
|------|------|----------|------|
| 답변 생성 속도 (p95) | < 3초 | 2.43초 | 달성 |
| 분류 정확도 | 85% 이상 | 90% | 달성 |
| BERTScore F1 | >= 80% | 46.05% | 진행 중 |
| ROUGE-L F1 | >= 0.30 | - | 진행 중 |
| 벡터 검색 속도 (p95) | < 1초 | 39.76ms | 달성 |
| 모델 크기 감소 | - | 68.3% 감소 (AWQ) | 달성 |

---

## 마일스톤

| 마일스톤 | 기간 | 상태 | 핵심 산출물 |
|---------|------|------|-----------|
| **M1: 기획/정책정합 설계** | Week 1~4 | 100% 완료 | PRD, 데이터 수집, 환경 구축 |
| **M2: 온프레미스 MVP** | Week 5~8 | 100% 완료 | QLoRA SFT, vLLM 서빙, 기본 UI, 폐쇄망 배포 |
| **M3: 고도화/분리 설계** | Week 9~12 | 46% 진행 중 | RAG 통합, AWQ 양자화, 공개/공통 데이터 분리 |
| **M4: 발표/마이그레이션** | Week 13~16 | 예정 | 통합 테스트, UAT, 정책 정합형 전환, 최종 발표 |

---

## DORA Metrics 대시보드

프로젝트의 DevOps 성숙도를 DORA 4대 지표로 측정하고 Grafana Cloud에서 실시간 모니터링한다.

**[DORA Metrics Dashboard (공개 링크)](https://umyunsang.grafana.net/public-dashboards/a7672d6682fb498eb4578a8634262280)**

| 지표 | 설명 |
|------|------|
| 배포 빈도 | main 브랜치 머지 PR 수 / 주 |
| 리드 타임 | PR 생성 → 머지 평균 시간 |
| 변경 실패율 | hotfix/revert 커밋 비율 |
| MTTR | bug 이슈 open → close 평균 시간 |

---

## 문서 포털

프로젝트의 전체 기술 문서는 문서 포털에서 확인할 수 있다.

**[GovOn 문서 포털](https://govon-org.github.io/GovOn/)**

주요 문서:

- [아키텍처 개요](https://govon-org.github.io/GovOn/architecture/overview/)
- [배포 가이드](https://govon-org.github.io/GovOn/deployment/docker/)
- [CI/CD 파이프라인](https://govon-org.github.io/GovOn/cicd/overview/)
- [모델 연구](https://govon-org.github.io/GovOn/research/model-analysis/)

---

## 발표 자료

프로젝트 발표 자료(PDF)는 아래 링크에서 확인할 수 있다.

- [GovOn: Secure On-Premise AI (PDF)](docs/GovOn_Secure_On-Premise_AI.pdf)

---

## 팀

**동아대학교 AI학과** | 2026 현장미러형 연계 프로젝트

| 역할 | 이름 | GitHub |
|------|------|--------|
| 팀장 | 엄윤상 | [@umyunsang](https://github.com/umyunsang) |
| 팀원 | 장시우 | [@siuJang](https://github.com/siuJang) |
| 팀원 | 이유정 | [@yuujjjj](https://github.com/yuujjjj) |

**멘토**: 천세진 교수 (동아대학교)

---

## 기여하기

프로젝트에 기여하고 싶다면 아래 문서를 참고한다.

- [기여 가이드](CONTRIBUTING.md) - 기여 방법, 커밋 컨벤션, PR 규칙
- [행동 강령](CODE_OF_CONDUCT.md) - 커뮤니티 행동 강령
- [보안 정책](SECURITY.md) - 보안 취약점 신고 방법

### 브랜치 전략

- `main`: 프로덕션 브랜치 (직접 push 금지, PR을 통해서만 머지)
- `feat/*`: 기능 개발 브랜치
- `fix/*`: 버그 수정 브랜치
- `docs/*`: 문서 작업 브랜치

---

## 라이선스

이 프로젝트는 [MIT License](LICENSE)로 배포된다.

> **참고**: 이 프로젝트에서 사용하는 EXAONE 모델은 [LGAI EXAONE License](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B)의 적용을 받는다. 모델 사용 시 해당 라이선스를 확인한다.
