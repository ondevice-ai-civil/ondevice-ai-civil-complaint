# 트러블슈팅

GovOn 추론 서버 및 데이터 파이프라인에서 자주 발생하는 문제와 해결 방법을 안내한다.
각 문제는 **증상 - 원인 - 해결 방법** 순서로 정리되어 있다.

---

## GPU OOM (Out of Memory)

### 증상

서버 기동 또는 추론 요청 시 다음과 유사한 오류가 발생한다.

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate XX MiB (GPU X; XX GiB total capacity; XX GiB already allocated)
```

### 원인

vLLM 엔진이 GPU 메모리를 초과하여 할당하려고 시도한다. AWQ INT4 모델(EXAONE-Deep-7.8B)은 약 5~8 GB VRAM을 사용하지만, vLLM의 KV 캐시와 multilingual-e5-large 임베딩 모델이 추가 메모리를 소비한다.

### 해결 방법

**1단계: `GPU_UTILIZATION` 낮추기**

기본값은 `0.8`이다. VRAM 16 GB 환경에서 OOM이 발생하면 `0.7` 이하로 조정한다.

```bash
export GPU_UTILIZATION=0.7
```

**2단계: `MAX_MODEL_LEN` 줄이기**

기본값은 `8192`이다. KV 캐시 메모리를 줄이려면 `4096` 또는 `2048`로 낮춘다.

```bash
export MAX_MODEL_LEN=4096
```

**3단계: 다른 GPU 프로세스 종료**

```bash
# GPU 사용 프로세스 확인
nvidia-smi

# 불필요한 프로세스 종료
kill -9 <PID>
```

**4단계: PyTorch 캐시 정리 후 재기동**

```bash
python -c "import torch; torch.cuda.empty_cache()"
uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 환경별 권장 설정

| VRAM | `GPU_UTILIZATION` | `MAX_MODEL_LEN` | 비고 |
|------|-------------------|------------------|------|
| 16 GB | `0.7` | `4096` | 최소 요구사항 |
| 24 GB | `0.8` | `8192` | 권장 환경 |
| 40 GB+ | `0.85` | `8192` | 여유로운 환경 |

!!! tip "GPU 메모리 모니터링"
    추론 서버 실행 중 GPU 메모리를 실시간으로 모니터링하려면 다음 명령을 사용한다.
    ```bash
    watch -n 1 nvidia-smi
    ```

---

## vLLM 서빙 오류

### EXAONE 런타임 패치 실패

#### 증상

서버 기동 시 다음과 유사한 오류가 발생한다.

```
AttributeError: module 'transformers.modeling_rope_utils' has no attribute 'RopeParameters'
AttributeError: module 'transformers.utils.generic' has no attribute 'check_model_inputs'
```

#### 원인

EXAONE-Deep-7.8B 모델은 최신 transformers API를 사용하는데, 설치된 transformers 버전에 해당 속성이 없는 경우 발생한다. GovOn은 `src/inference/vllm_stabilizer.py`에서 런타임 패치로 이 문제를 해결한다.

#### 해결 방법

1. `vllm_stabilizer.py`의 패치가 정상 적용되는지 확인한다.

```bash
python -c "from src.inference.vllm_stabilizer import apply_transformers_patch; apply_transformers_patch()"
```

정상이면 다음 메시지가 출력된다.

```
Applying runtime patches for EXAONE...
  [SUCCESS] Injected RopeParameters into transformers.modeling_rope_utils
  [SUCCESS] Injected check_model_inputs into transformers.utils.generic
  [SUCCESS] Injected ALL_ATTENTION_FUNCTIONS dummy
```

2. 패치가 이미 적용 완료된 경우(속성이 이미 존재) 성공 메시지 없이 종료된다. 이는 정상이다.

3. 여전히 오류가 발생하면 transformers 버전을 확인한다.

```bash
pip show transformers
```

`>= 4.40.0`인지 확인하고, 버전이 낮으면 업그레이드한다.

```bash
pip install --upgrade transformers>=4.40.0
```

### trust_remote_code 오류

#### 증상

```
ValueError: Loading ... requires you to execute the configuration file
in that repo on your local machine.
```

#### 원인

EXAONE 모델은 커스텀 코드를 사용하므로 `trust_remote_code=True` 설정이 필요하다.

#### 해결 방법

GovOn의 `api_server.py`는 이 설정을 기본으로 활성화한다 (`TRUST_REMOTE_CODE = True`). 직접 vLLM을 사용하는 경우 다음을 확인한다.

```python
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    trust_remote_code=True,  # 반드시 True
    # ...
)
```

### enforce_eager 관련 오류

#### 증상

CUDA graph 관련 오류 또는 불안정한 추론 결과가 발생한다.

#### 해결 방법

GovOn은 EXAONE 패치 모델의 안정성을 위해 `enforce_eager=True`를 기본 사용한다. 이 설정은 `api_server.py`의 `vLLMEngineManager.initialize()`에 적용되어 있다.

### vLLM import 오류

#### 증상

```
ImportError: No module named 'vllm'
```

또는

```
ImportError: libcudart.so.12: cannot open shared object file
```

#### 해결 방법

1. 추론 의존성을 설치한다.

```bash
pip install -e ".[inference]"
```

2. CUDA 라이브러리가 경로에 포함되어 있는지 확인한다.

```bash
# CUDA 라이브러리 경로 확인
echo $LD_LIBRARY_PATH

# 경로가 비어 있으면 설정
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

3. PyTorch CUDA 버전과 시스템 CUDA 버전이 일치하는지 확인한다.

```bash
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
nvcc --version
```

---

## AWQ 양자화 호환성 문제

### AutoAWQ 설치 오류

#### 증상

```
ImportError: No module named 'awq'
```

또는 CUDA 버전 불일치로 빌드가 실패한다.

#### 해결 방법

1. AutoAWQ를 설치한다.

```bash
pip install autoawq>=0.2.0
```

2. CUDA 버전과 PyTorch CUDA 버전이 일치하는지 확인한다.

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

두 버전이 다르면 PyTorch를 CUDA 버전에 맞게 재설치한다.

```bash
# CUDA 12.1 예시
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 캘리브레이션 데이터 형식 오류

#### 증상

양자화 시 캘리브레이션 샘플이 0개로 준비된다.

#### 해결 방법

캘리브레이션 데이터가 JSONL 형식이며 각 행에 `instruction`, `input`, `output` 필드가 포함되어 있는지 확인한다.

```json
{"instruction": "다음 민원을 분석하세요.", "input": "도로 파손 민원입니다.", "output": "해당 민원은 도로 관리 부서로 접수됩니다."}
```

---

## FAISS 인덱스 오류

### 인덱스 파일 미존재

#### 증상

```
ERROR - Data path not found: data/processed/v2_train.jsonl
```

또는 서버 기동 시 RAG 기능이 비활성화된다.

#### 해결 방법

1. `INDEX_PATH` 환경변수에 지정된 경로에 인덱스 파일이 있는지 확인한다.

```bash
ls -la models/faiss_index/complaints.index
ls -la models/faiss_index/complaints.index.meta.json
```

2. 인덱스가 없으면 `DATA_PATH`에 JSONL 데이터를 배치한다. 서버 기동 시 자동으로 인덱스를 빌드한다.

```bash
export DATA_PATH=data/processed/v2_train.jsonl
export INDEX_PATH=models/faiss_index/complaints.index
```

3. 인덱스를 수동으로 빌드하려면 Python에서 직접 실행한다.

```python
from src.inference.retriever import CivilComplaintRetriever

retriever = CivilComplaintRetriever(
    data_path="data/processed/v2_train.jsonl"
)
retriever.save_index("models/faiss_index/complaints.index")
```

### 메타데이터 파일 누락

#### 증상

```
FileNotFoundError: .../complaints.index.meta.json
```

#### 원인

FAISS 인덱스 파일(`complaints.index`)은 존재하지만, 함께 저장되는 메타데이터 파일(`complaints.index.meta.json`)이 누락되었다.

#### 해결 방법

인덱스와 메타데이터를 함께 재빌드한다. 위의 수동 빌드 방법을 사용한다.

### MultiIndexManager 인덱스 로드 실패

#### 증상

```
WARNING - 인덱스 자동 로드 실패 (case): ...
```

#### 해결 방법

`MultiIndexManager`는 `models/faiss_index/` 디렉토리 하위에 타입별 디렉토리 구조를 기대한다.

```
models/faiss_index/
├── case/
│   ├── index.faiss
│   └── metadata.json
├── law/
├── manual/
├── notice/
└── index_registry.json
```

해당 디렉토리와 파일이 존재하는지 확인한다. `index_registry.json`이 손상된 경우 삭제하면 자동으로 재생성된다.

### 인덱스 차원 불일치

#### 증상

```
RuntimeError: Sizes of tensors must match except in dimension 0.
```

#### 원인

FAISS 인덱스가 다른 임베딩 모델로 생성되었을 수 있다. GovOn은 `multilingual-e5-large` (차원=1024)를 사용한다.

#### 해결 방법

인덱스를 삭제하고 재빌드한다.

```bash
rm -rf models/faiss_index/complaints.index*
# 서버 재시작 시 자동 빌드
```

---

## Docker 네트워킹 문제

### GPU가 컨테이너에서 인식되지 않음

#### 증상

```
RuntimeError: No CUDA GPUs are available
```

#### 해결 방법

1. NVIDIA Container Toolkit이 설치되어 있는지 확인한다.

```bash
nvidia-container-cli --version
```

2. Docker에서 GPU 접근을 테스트한다.

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

3. `docker-compose.yml`에 GPU 설정이 포함되어 있는지 확인한다.

```yaml
services:
  govon-backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 컨테이너 간 통신 실패

#### 증상

프론트엔드 컨테이너에서 API 서버에 접속할 수 없다.

#### 해결 방법

1. 동일한 Docker 네트워크에 있는지 확인한다.

```bash
docker network ls
docker network inspect govon_default
```

2. 서비스 이름으로 접근한다. Docker Compose 환경에서는 서비스 이름이 호스트명으로 동작한다.

```bash
# 컨테이너 내부에서
curl http://govon-backend:8000/health
```

### 포트 충돌

#### 증상

```
Error starting userland proxy: listen tcp 0.0.0.0:8000: bind: address already in use
```

#### 해결 방법

해당 포트를 사용 중인 프로세스를 확인하고 종료한다.

```bash
# 포트 사용 프로세스 확인
lsof -i :8000

# 프로세스 종료
kill -9 <PID>
```

또는 Docker Compose에서 다른 포트를 매핑한다.

```yaml
ports:
  - "8001:8000"  # 호스트 8001 → 컨테이너 8000
```

---

## API 서버 기동 문제

### API Key 인증 오류

#### 증상

```json
{"detail": "유효하지 않은 API 키입니다."}
```

HTTP 상태 코드 `401 Unauthorized`가 반환된다.

#### 해결 방법

1. 환경변수와 요청 헤더의 API 키가 일치하는지 확인한다.

```bash
echo $API_KEY
```

2. 요청에 올바른 헤더를 포함한다.

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{"prompt": "테스트", "max_tokens": 10}'
```

3. 개발 환경에서 인증을 비활성화하려면 `API_KEY` 환경변수를 해제한다.

```bash
unset API_KEY
```

!!! warning "프로덕션에서는 반드시 API_KEY를 설정한다"
    `API_KEY` 미설정 시 모든 요청이 인증 없이 통과한다. 개발 환경에서만 사용한다.

### Rate Limiting 초과

#### 증상

```json
{"error": "Rate limit exceeded"}
```

HTTP 상태 코드 `429 Too Many Requests`가 반환된다.

#### 원인

`slowapi` 기반 Rate Limiting이 활성화되어 있으며, 기본 제한을 초과했다.

| 엔드포인트 | 제한 |
|------------|------|
| `/v1/generate` | 30회/분 |
| `/v1/stream` | 30회/분 |
| `/search`, `/v1/search` | 60회/분 |

#### 해결 방법

- 요청 빈도를 줄인다.
- `slowapi`가 설치되어 있지 않으면 Rate Limiting이 자동으로 비활성화된다. 개발 환경에서 제한을 해제하려면 `slowapi`를 제거한다.

```bash
pip uninstall slowapi
```

### CORS 오류

#### 증상

브라우저 콘솔에서 다음과 유사한 오류가 나타난다.

```
Access to fetch at 'http://localhost:8000/v1/generate' from origin
'http://localhost:3000' has been blocked by CORS policy
```

#### 해결 방법

`CORS_ORIGINS` 환경변수에 프론트엔드 출처를 추가한다. 여러 출처는 쉼표로 구분한다.

```bash
export CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

!!! info "`CORS_ORIGINS`가 비어 있으면 CORS 미들웨어 자체가 추가되지 않는다"
    서버 간 통신(Server-to-Server)에서는 CORS가 필요하지 않으므로 비워 두어도 된다.

### 임베딩 모델 다운로드 실패

#### 증상

```
OSError: Can't load tokenizer for 'intfloat/multilingual-e5-large'
```

#### 해결 방법

네트워크 연결을 확인하고 HuggingFace Hub에 접근 가능한지 확인한다. 오프라인 환경에서는 모델을 미리 다운로드하여 로컬 경로를 지정한다.

```python
# 온라인 환경에서 미리 다운로드
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large")
model.save("models/multilingual-e5-large")
```

---

## 로그 분석

GovOn은 `loguru`를 사용하여 구조화된 로그를 출력한다.

### 로그 레벨

| 레벨 | 용도 | 예시 |
|------|------|------|
| `DEBUG` | 상세 디버깅 정보 | 검색 결과 개수, 임베딩 차원 |
| `INFO` | 정상 동작 기록 | 서버 시작, 모델 로딩 완료 |
| `WARNING` | 주의가 필요한 상황 | PII 마스커 초기화 실패, 인덱스 로드 실패 |
| `ERROR` | 오류 발생 | 모델 로딩 실패, 추론 오류 |

### 로그에서 문제 진단하기

```bash
# 서버 로그에서 ERROR만 필터링
uvicorn src.inference.api_server:app 2>&1 | grep "ERROR"

# Docker 컨테이너 로그
docker logs govon-backend --since 10m
docker logs govon-backend -f  # 실시간 추적

# 특정 키워드로 필터링
docker logs govon-backend 2>&1 | grep "OOM\|OutOfMemory\|CUDA"
```

### 자주 나타나는 로그 메시지와 의미

| 로그 메시지 | 의미 | 조치 |
|-------------|------|------|
| `PIIMasker 초기화 실패` | PII 마스킹 비활성화 상태 | 검색 결과에 개인정보가 노출될 수 있음. NER 모델 확인 |
| `인덱스 자동 로드 실패` | 특정 타입의 인덱스 없음 | 해당 타입의 인덱스 파일 확인 |
| `RAG 비활성화` | 검색 기능 없이 동작 | `DATA_PATH`, `INDEX_PATH` 확인 |
| `EXAONE 런타임 패치 적용` | 정상 기동 과정 | 조치 불필요 |

---

## 빠른 진단 체크리스트

문제가 발생하면 다음 순서로 확인한다.

```mermaid
graph TD
    A[문제 발생] --> B{서버가 기동되는가?}
    B -->|아니오| C{GPU 인식되는가?}
    C -->|아니오| D[nvidia-smi 확인<br/>CUDA 드라이버 설치]
    C -->|예| E{모델 로딩 오류?}
    E -->|OOM| F[GPU_UTILIZATION 낮추기<br/>MAX_MODEL_LEN 줄이기]
    E -->|AttributeError| G[vllm_stabilizer 패치 확인<br/>transformers 버전 확인]
    E -->|ImportError| H[pip install -e .[inference]]
    B -->|예| I{API 응답이 정상인가?}
    I -->|401| J[API_KEY 환경변수 확인]
    I -->|429| K[Rate Limit 확인<br/>요청 빈도 줄이기]
    I -->|500| L[서버 로그 ERROR 확인]
    I -->|CORS 오류| M[CORS_ORIGINS 설정]
```

---

## 관련 문서

- [시작하기](getting-started.md) -- 사전 요구사항, 설치, 서버 기동
- [개발 규칙](development.md) -- 브랜치 전략, 커밋 컨벤션
- [보안 정책](security.md) -- API 인증, Rate Limiting, 프롬프트 인젝션 방어
- [인프라 아키텍처](../deployment/architecture.md) -- Docker 구성, 네트워크 아키텍처
