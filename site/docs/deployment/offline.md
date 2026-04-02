# 오프라인 배포

인터넷 접속이 불가능한 폐쇄망(에어갭) 환경에서 GovOn 시스템을 배포하는 방법을 설명한다. GitHub Release에서 오프라인 패키지를 다운로드하고, `docker load`로 이미지를 로드한 뒤 시스템을 실행한다.

---

## 오프라인 배포 개요

폐쇄망 배포는 다음 흐름으로 진행된다.

```
인터넷 환경 (사전 준비)              폐쇄망 환경 (배포 실행)
┌──────────────────────┐            ┌──────────────────────┐
│ GitHub Release에서    │            │ USB/외장 매체로      │
│ 오프라인 패키지       │  ──전달──→ │ 패키지 전달          │
│ 다운로드              │            │                      │
│                      │            │ tar 해제             │
│ 모델 파일 다운로드    │  ──전달──→ │ docker load          │
│ (HuggingFace)        │            │ docker compose up    │
└──────────────────────┘            └──────────────────────┘
```

---

## 사전 요구사항

### 폐쇄망 서버 요구사항

| 소프트웨어 | 최소 버전 | 확인 명령 |
|-----------|----------|----------|
| Docker Engine | 24.0+ | `docker --version` |
| Docker Compose | 2.20+ | `docker compose version` |
| NVIDIA Driver | 525.0+ | `nvidia-smi` |
| NVIDIA Container Toolkit | 1.13+ | `docker info \| grep nvidia` |

!!! warning "사전 설치 필수"
    Docker, NVIDIA Driver, NVIDIA Container Toolkit은 폐쇄망 전환 전에 미리 설치해야 한다. 오프라인 설치가 필요한 경우 각 공식 문서의 오프라인 설치 가이드를 참조한다.

### 하드웨어 요구사항

온라인 배포와 동일한 하드웨어 사양이 필요하다. 자세한 내용은 [온라인 배포 - 하드웨어 요구사항](online.md#하드웨어-요구사항)을 참조한다.

---

## 1단계: 오프라인 패키지 다운로드

인터넷이 가능한 환경에서 GitHub Release 페이지에 접속하여 오프라인 패키지를 다운로드한다.

### GitHub Release에서 다운로드

[https://github.com/GovOn-Org/GovOn/releases](https://github.com/GovOn-Org/GovOn/releases)에서 최신 릴리스의 `govon-offline-package.tar.gz` 파일을 다운로드한다.

```bash
# 또는 CLI로 다운로드
gh release download --pattern "govon-offline-package.tar.gz"
```

### 패키지 내용물

오프라인 패키지에는 다음 파일이 포함되어 있다.

| 파일 | 설명 |
|------|------|
| `govon-image.tar.gz` | Docker 이미지 아카이브 |
| `.env.airgap.example` | 폐쇄망용 런타임 환경변수 템플릿 |
| `scripts/offline-deploy.sh` | 자동 배포 스크립트 |
| `scripts/smoke-test.sh` | 배포 검증 스크립트 |
| `docker-compose.offline.yml` | 오프라인용 Compose 설정 |

### 모델 파일 다운로드

LLM 모델 파일은 별도로 다운로드해야 한다.

```bash
# HuggingFace에서 모델 다운로드
pip install huggingface_hub
huggingface-cli download umyunsang/GovOn-EXAONE-AWQ-v2 --local-dir ./models/GovOn-EXAONE-AWQ-v2
```

!!! tip "모델 파일 크기"
    AWQ 양자화 모델 기준 약 5GB이다. USB 또는 외장 하드에 충분한 공간을 확보한다.

---

## 2단계: 폐쇄망으로 파일 전달

다운로드한 파일을 USB, 외장 하드 등의 물리 매체로 폐쇄망 서버에 전달한다.

전달해야 하는 파일 목록:

- `govon-offline-package.tar.gz` (오프라인 패키지)
- `models/` 디렉토리 (모델 파일)
- FAISS 인덱스 파일 (있는 경우)
- 학습/검색 데이터 파일 (있는 경우)

---

## 3단계: 패키지 해제

폐쇄망 서버에서 오프라인 패키지를 해제한다.

```bash
# 작업 디렉토리 생성
mkdir -p /opt/govon && cd /opt/govon

# 오프라인 패키지 해제
tar xzf govon-offline-package.tar.gz
```

해제 후 디렉토리 구조:

```
/opt/govon/
├── .env.airgap.example
├── govon-image.tar.gz
├── scripts/
│   ├── offline-deploy.sh
│   └── smoke-test.sh
└── docker-compose.offline.yml
```

---

## 4단계: 자동 배포 (권장)

`offline-deploy.sh` 스크립트를 실행하면 다음 과정을 자동으로 수행한다.

```bash
chmod +x scripts/offline-deploy.sh
./scripts/offline-deploy.sh
```

스크립트 실행 과정:

1. Docker 및 Docker Compose 설치 확인
2. NVIDIA Container Toolkit 감지 (경고만 표시)
3. Docker 이미지 파일 로드 (`docker load`)
4. `.env.airgap.example` 기준 `.env` 생성
5. `API_KEY`, `BM25_INDEX_HMAC_KEY`가 placeholder인지 검사하고, 그대로면 fail-fast
6. 볼륨 디렉토리 생성 (`models/`, `data/`, `agents/`, `configs/`, `logs/`, `.cache/`)
7. 컨테이너 실행 (`docker compose --env-file .env -f docker-compose.offline.yml up -d`)
8. 헬스체크 대기 (최대 120초)

!!! warning "첫 실행 전 필수 수정"
    `offline-deploy.sh`는 `.env.airgap.example`을 복사한 직후 바로 기동하지 않는다.
    `API_KEY`, `BM25_INDEX_HMAC_KEY`가 예시 placeholder 그대로면 중단되므로, `.env`에서 안전한 임의 문자열로 교체한 뒤 다시 실행해야 한다.

정상 완료 시 출력:

```
==============================
[SUCCESS] GovOn 서버가 정상 시작되었습니다!
API 주소: http://localhost:8000
헬스체크: http://localhost:8000/health
==============================
```

---

## 4단계 (대안): 수동 배포

자동 스크립트 대신 수동으로 배포하려면 다음 단계를 따른다.

### Docker 이미지 로드

```bash
gunzip -c govon-image.tar.gz | docker load
```

성공 시 출력:

```
Loaded image: ghcr.io/govon-org/govon:latest
```

### 모델 파일 배치

```bash
# 볼륨 디렉토리 생성
mkdir -p models/faiss_index models/bm25_index data/processed agents configs logs .cache

# 모델 파일 복사 (USB 등에서)
cp -r /media/usb/models/GovOn-EXAONE-AWQ-v2 ./models/
cp -r /media/usb/models/faiss_index/* ./models/faiss_index/
```

### 환경변수 설정

오프라인 환경에서는 `.env.airgap.example`을 `.env`로 복사한 뒤 필요한 값을 수정한다. 경로는 호스트가 아니라 컨테이너 내부 경로(`/app/...`) 기준이다.

```bash
cp .env.airgap.example .env

# 필요 시 수정
# MODEL_PATH=/app/models/GovOn-EXAONE-AWQ-v2
# API_KEY=your-secure-api-key
# BM25_INDEX_HMAC_KEY=your-secure-hmac-key
```

### 컨테이너 실행

```bash
docker compose --env-file .env -f docker-compose.offline.yml up -d
```

### 상태 확인

```bash
# 컨테이너 상태
docker compose --env-file .env -f docker-compose.offline.yml ps

# 헬스체크
curl -f http://localhost:8000/health
```

---

## 5단계: Smoke Test

배포가 완료된 후 `smoke-test.sh` 스크립트로 시스템이 정상 동작하는지 검증한다.

```bash
chmod +x scripts/smoke-test.sh
./scripts/smoke-test.sh
```

또는 특정 호스트를 대상으로 테스트한다.

```bash
./scripts/smoke-test.sh http://localhost:8000
```

### 테스트 항목

| 테스트 | 설명 |
|--------|------|
| `GET /health` | 헬스 엔드포인트 연결 및 응답 확인 |
| 응답 구조 검증 | `/health` 응답에 `status` 필드 존재 여부 확인 |

정상 완료 시 출력:

```
=== GovOn Smoke Test ===
대상: http://localhost:8000

[TEST] GET /health ... PASS
[TEST] /health 응답 구조 ... PASS

==============================
결과: PASS=2, FAIL=0
상태: PASSED
```

---

## 오프라인 패키지 빌드

오프라인 패키지는 GitHub Actions 워크플로우(`offline-package.yml`)로 자동 생성된다.

### 자동 빌드 트리거

| 이벤트 | 동작 |
|--------|------|
| GitHub Release 발행 | 패키지 빌드 후 릴리스 에셋으로 업로드 |
| 수동 실행 (`workflow_dispatch`) | 패키지 빌드 후 아티팩트로 업로드 (7일 보관) |

### 빌드 과정

```
GHCR에서 이미지 Pull → docker save로 아카이브 → 배포 스크립트와 함께 tar.gz 패키징
```

### 수동으로 패키지 생성

인터넷이 가능한 환경에서 직접 패키지를 생성할 수도 있다.

```bash
# 1. 이미지 Pull
docker pull ghcr.io/govon-org/govon:latest

# 2. 이미지 아카이브
docker save ghcr.io/govon-org/govon:latest | gzip > govon-image.tar.gz

# 3. 패키지 생성
tar czf govon-offline-package.tar.gz \
  govon-image.tar.gz \
  .env.airgap.example \
  scripts/offline-deploy.sh \
  scripts/smoke-test.sh \
  docker-compose.offline.yml
```

---

## 문제 해결

### 이미지 로드 실패

```
Error: no such file: govon-image.tar.gz
```

오프라인 패키지가 올바르게 해제되었는지 확인한다. `govon-image.tar.gz` 파일이 현재 디렉토리 또는 프로젝트 루트에 있어야 한다.

### GPU 미감지

```
[WARNING] NVIDIA Container Toolkit이 감지되지 않았습니다.
```

NVIDIA Container Toolkit이 설치되어 있는지 확인한다.

```bash
# 확인
docker info 2>/dev/null | grep -i nvidia

# 재설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 서버 시작 실패 (120초 타임아웃)

모델 로딩에 시간이 더 필요한 경우이다. 컨테이너 로그를 확인한다.

```bash
docker compose --env-file .env -f docker-compose.offline.yml logs
```

일반적인 원인:

- GPU 메모리 부족: `GPU_UTILIZATION` 값을 낮춘다
- 모델 경로 오류: `MODEL_PATH` 환경변수를 확인한다
- FAISS 인덱스 파일 누락: `models/faiss_index/` 디렉토리를 확인한다
