# 온라인 배포

인터넷 접속이 가능한 환경에서 GHCR(GitHub Container Registry) 이미지를 pull하여 GovOn 시스템을 배포하는 방법을 설명한다.

---

## 사전 요구사항

배포 서버에 다음 소프트웨어가 설치되어 있어야 한다.

| 소프트웨어 | 최소 버전 | 확인 명령 |
|-----------|----------|----------|
| Docker Engine | 24.0+ | `docker --version` |
| Docker Compose | 2.20+ | `docker compose version` |
| NVIDIA Driver | 525.0+ | `nvidia-smi` |
| NVIDIA Container Toolkit | 1.13+ | `docker info \| grep nvidia` |
| curl | - | `curl --version` |

### 하드웨어 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| GPU | NVIDIA RTX 3060 (12GB VRAM) | NVIDIA RTX 4090 (24GB VRAM) |
| RAM | 16GB | 32GB 이상 |
| 디스크 | 50GB 여유 공간 | 100GB 이상 |
| CPU | 4코어 | 8코어 이상 |

!!! warning "GPU 필수"
    GovOn은 vLLM 기반 LLM 추론을 수행하므로 NVIDIA GPU가 반드시 필요하다. GPU 없이는 추론 서버가 시작되지 않는다.

---

## NVIDIA Container Toolkit 설치

Docker 컨테이너에서 GPU를 사용하려면 NVIDIA Container Toolkit이 필요하다.

### Ubuntu/Debian

```bash
# NVIDIA Container Toolkit 저장소 추가
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 런타임 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 설치 확인

```bash
# GPU가 Docker 컨테이너에서 인식되는지 확인
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

정상 설치 시 호스트와 동일한 `nvidia-smi` 출력이 표시된다.

---

## 배포 절차

### 1단계: 이미지 Pull

```bash
docker pull ghcr.io/govon-org/govon:latest
```

특정 버전을 배포하려면 태그를 지정한다.

```bash
docker pull ghcr.io/govon-org/govon:v1.0.0
```

### 2단계: 환경변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성한다.

```bash
# .env 파일 생성
cat > .env << 'EOF'
API_KEY=your-secure-api-key-here
MODEL_PATH=umyunsang/GovOn-EXAONE-LoRA-v2
GPU_UTILIZATION=0.8
MAX_MODEL_LEN=8192
CORS_ORIGINS=http://localhost:3000
EOF
```

### 환경변수 상세

| 환경변수 | 필수 | 기본값 | 설명 |
|---------|------|--------|------|
| `API_KEY` | 권장 | - | API 인증 키 (`X-API-Key` 헤더로 전달) |
| `MODEL_PATH` | 선택 | `umyunsang/GovOn-EXAONE-LoRA-v2` | HuggingFace 모델 ID 또는 로컬 모델 경로 |
| `DATA_PATH` | 선택 | `/app/data/processed/v2_train.jsonl` | 학습 데이터 파일 경로 |
| `INDEX_PATH` | 선택 | `/app/models/faiss_index/complaints.index` | FAISS 인덱스 파일 경로 |
| `GPU_UTILIZATION` | 선택 | `0.8` | vLLM GPU 메모리 사용 비율 (0.0~1.0) |
| `MAX_MODEL_LEN` | 선택 | `8192` | 최대 시퀀스 길이 (토큰 수) |
| `CORS_ORIGINS` | 선택 | - | 허용할 CORS 오리진 (쉼표 구분) |

!!! tip "GPU_UTILIZATION 조정"
    VRAM이 부족하면 `GPU_UTILIZATION` 값을 `0.6`~`0.7`로 낮춘다. VRAM이 넉넉하면 `0.9`까지 올려 처리량을 높일 수 있다.

### 3단계: 볼륨 디렉토리 준비

```bash
mkdir -p models/faiss_index data/processed agents configs
```

필요한 데이터 파일을 배치한다.

| 경로 | 내용 |
|------|------|
| `models/faiss_index/` | FAISS 인덱스 파일 (`.index`, `.metadata`) |
| `models/bm25_index/` | BM25 인덱스 파일 |
| `data/processed/` | 전처리된 학습/검색 데이터 |

### 4단계: 컨테이너 실행

```bash
docker compose up -d
```

오프라인 Compose 파일을 사용하여 GHCR 이미지로 실행할 수도 있다.

```bash
docker compose -f docker-compose.offline.yml up -d
```

### 5단계: 상태 확인

```bash
# 컨테이너 상태 확인
docker compose ps

# 헬스체크
curl http://localhost:8000/health
```

정상 응답:

```json
{"status": "healthy"}
```

---

## 헬스체크

### /health 엔드포인트

GovOn API 서버는 `/health` 엔드포인트를 제공한다.

```bash
curl -f http://localhost:8000/health
```

| HTTP 상태 | 의미 |
|-----------|------|
| `200 OK` | 서버 정상 동작 |
| 연결 실패 | 서버 미시작 또는 모델 로딩 중 |

!!! info "시작 대기 시간"
    최초 실행 시 모델 로딩에 60초 이상 소요될 수 있다. Docker Compose의 `start_period: 60s` 설정이 이를 반영한다.

### 컨테이너 로그 확인

```bash
# 전체 로그 확인
docker compose logs

# 실시간 로그 추적
docker compose logs -f

# 최근 100줄만 확인
docker compose logs --tail=100
```

---

## Blue/Green 배포

프로덕션 환경에서는 `scripts/deploy.sh` 스크립트를 사용하여 무중단 Blue/Green 배포를 수행할 수 있다.

### 새 버전 배포

```bash
./scripts/deploy.sh deploy latest
# 또는 특정 버전
./scripts/deploy.sh deploy v1.0.0
```

### 배포 상태 확인

```bash
./scripts/deploy.sh status
```

출력 예시:

```
========================================
       GovOn 배포 상태
========================================
 활성 슬롯  : blue
 Blue  (8001): Up 2 hours
 Green (8002): stopped
========================================
```

### 롤백

```bash
./scripts/deploy.sh rollback
```

### 동작 방식

1. 현재 비활성 슬롯에 새 버전을 배포한다
2. 헬스체크가 통과하면 활성 슬롯을 전환한다
3. 이전 슬롯을 중지한다
4. 헬스체크가 실패하면 자동으로 배포를 취소하고 이전 상태를 유지한다

---

## 컨테이너 관리

### 중지

```bash
docker compose down
```

### 재시작

```bash
docker compose restart
```

### 이미지 업데이트

```bash
docker pull ghcr.io/govon-org/govon:latest
docker compose down
docker compose up -d
```

### 디스크 정리

```bash
# 사용하지 않는 이미지 정리
docker image prune -f

# 전체 정리 (주의: 모든 미사용 리소스 삭제)
docker system prune -f
```
