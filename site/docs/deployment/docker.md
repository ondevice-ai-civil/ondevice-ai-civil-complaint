# Docker 가이드

GovOn 시스템의 Docker 이미지 구성, 빌드 방법, GHCR(GitHub Container Registry)를 통한 이미지 관리 방법을 설명한다.

---

## Dockerfile 구성

GovOn Docker 이미지는 NVIDIA CUDA 12.1 + Ubuntu 22.04 기반으로, Python 3.10 런타임과 모든 추론 의존성을 포함한다.

### 베이스 이미지

```
nvidia/cuda:12.1.1-devel-ubuntu22.04
```

| 항목 | 값 |
|------|-----|
| CUDA 버전 | 12.1.1 |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 |
| 작업 디렉토리 | `/app` |
| 노출 포트 | `8000` |

### 기본 환경변수

Dockerfile에 다음 기본값이 설정되어 있다. 실행 시 `docker-compose.yml` 또는 `-e` 옵션으로 재정의할 수 있다.

| 환경변수 | 기본값 | 설명 |
|---------|--------|------|
| `MODEL_PATH` | `umyunsang/GovOn-EXAONE-LoRA-v2` | HuggingFace 모델 경로 또는 로컬 경로 |
| `DATA_PATH` | `/app/data/processed/v2_train.jsonl` | 학습 데이터 경로 |
| `INDEX_PATH` | `/app/models/faiss_index/complaints.index` | FAISS 인덱스 파일 경로 |

### 빌드 레이어 구조

```
1. nvidia/cuda:12.1.1-devel-ubuntu22.04    ← 베이스 이미지
2. python3.10, pip, git, 시스템 패키지     ← 시스템 의존성
3. requirements.txt + pyproject.toml        ← Python 의존성 (캐시 레이어)
4. src/, agents/                            ← 소스 코드 복사
5. models/, data/ 디렉토리 생성             ← 볼륨 마운트 대상
```

!!! tip "레이어 캐싱"
    의존성 설치(`requirements.txt`)와 소스 코드 복사를 분리하여 소스 코드 변경 시 의존성 레이어를 재사용한다. 이로 인해 반복 빌드 속도가 크게 향상된다.

---

## GHCR 이미지

GovOn Docker 이미지는 GitHub Container Registry(GHCR)에 자동 배포된다.

### 이미지 Pull

```bash
docker pull ghcr.io/govon-org/govon:latest
```

### 이미지 태그 규칙

| 태그 형식 | 예시 | 설명 |
|----------|------|------|
| `latest` | `ghcr.io/govon-org/govon:latest` | `main` 브랜치 최신 빌드 |
| `v*` | `ghcr.io/govon-org/govon:v1.0.0` | 릴리스 태그 기반 버전 |
| `sha-*` | `ghcr.io/govon-org/govon:sha-abc1234` | 커밋 SHA 기반 (CI 추적용) |

### 자동 빌드 트리거

`main` 브랜치 push 또는 `v*` 태그 생성 시 GitHub Actions 워크플로우(`docker-publish.yml`)가 자동 실행된다.

```
main push / v* 태그 → Docker Build → GHCR Push → Trivy 보안 스캔 → Smoke Test
```

빌드 파이프라인에는 다음 단계가 포함된다.

1. **Docker Buildx**: 멀티플랫폼 빌드 지원
2. **GHCR Push**: `latest`, `sha-*`, `v*` 태그로 이미지 Push
3. **Trivy 스캔**: CRITICAL/HIGH 등급 취약점 검사
4. **Smoke Test**: 이미지 구조 검증(Python 설치, 소스 코드 존재, FastAPI import)

---

## 로컬 빌드

GHCR 이미지 대신 로컬에서 직접 빌드하려면 다음 명령을 실행한다.

```bash
# 프로젝트 루트 디렉토리에서 실행
docker build -t govon-backend:latest .
```

또는 Docker Compose를 사용하여 빌드와 실행을 동시에 수행한다.

```bash
docker compose build
docker compose up -d
```

---

## docker-compose 설정

### 온라인 환경 (docker-compose.yml)

인터넷 접속이 가능한 환경에서 로컬 빌드 후 실행하는 구성이다.

```yaml
services:
  govon-backend:
    build:
      context: .
      dockerfile: Dockerfile
    image: govon-backend:latest
    container_name: govon-backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./agents:/app/agents
      - ./configs:/app/configs
    environment:
      - API_KEY=${API_KEY}
      - MODEL_PATH=${MODEL_PATH:-umyunsang/GovOn-EXAONE-LoRA-v2}
      - GPU_UTILIZATION=0.8
      - MAX_MODEL_LEN=8192
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### 오프라인 환경 (docker-compose.offline.yml)

폐쇄망에서 GHCR에서 미리 pull한 이미지를 사용하는 구성이다.

```yaml
services:
  govon-backend:
    image: ghcr.io/govon-org/govon:latest
    container_name: govon-backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./agents:/app/agents
      - ./configs:/app/configs
    environment:
      - API_KEY=${API_KEY}
      - MODEL_PATH=${MODEL_PATH:-umyunsang/GovOn-EXAONE-LoRA-v2}
      - GPU_UTILIZATION=0.8
      - MAX_MODEL_LEN=8192
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### 볼륨 마운트

| 호스트 경로 | 컨테이너 경로 | 설명 |
|------------|-------------|------|
| `./models` | `/app/models` | FAISS 인덱스, BM25 인덱스, 모델 파일 |
| `./data` | `/app/data` | 학습/테스트 데이터 |
| `./agents` | `/app/agents` | 에이전트 설정 파일 |
| `./configs` | `/app/configs` | 시스템 설정 파일 |

### GPU 설정

Docker Compose의 `deploy.resources.reservations.devices` 설정으로 컨테이너에 GPU를 할당한다. 이 설정을 사용하려면 호스트에 **NVIDIA Container Toolkit**이 설치되어 있어야 한다.

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all          # 모든 GPU 사용, 특정 개수는 숫자로 지정
          capabilities: [gpu]
```

NVIDIA Container Toolkit 설치 방법은 [온라인 배포 가이드](online.md#nvidia-container-toolkit-설치)를 참조한다.

### 헬스체크

컨테이너는 `/health` 엔드포인트를 통해 자동 헬스체크를 수행한다.

| 항목 | 값 |
|------|-----|
| 체크 간격 | 30초 |
| 타임아웃 | 10초 |
| 재시도 횟수 | 3회 |
| 시작 대기 | 60초 (모델 로딩 시간 고려) |

```bash
# 헬스체크 수동 확인
curl http://localhost:8000/health
```

정상 응답 예시:

```json
{"status": "healthy"}
```
