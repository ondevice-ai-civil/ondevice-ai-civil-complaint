# EXAONE-3.5-7.8B-Instruct 실험 계획서
## QLoRA 파인튜닝 및 AWQ 양자화

**문서 버전**: 1.0
**작성일**: 2026-03-05
**프로젝트**: On-Device AI 민원 분석 및 처리 시스템
**실행 환경**: Google Colab Pro A100 런타임
**베이스 모델**: [LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)

---

## 목차

1. [실험 개요 (Experiment Overview)](#1-실험-개요-experiment-overview)
2. [모델 및 데이터셋 구성 (Model & Dataset Configuration)](#2-모델-및-데이터셋-구성-model--dataset-configuration)
3. [실험 환경 설정 (Environment Setup)](#3-실험-환경-설정-environment-setup)
4. [데이터 준비 파이프라인 (Data Preparation)](#4-데이터-준비-파이프라인-data-preparation)
5. [QLoRA 파인튜닝 실험 설계 (QLoRA Fine-tuning Design)](#5-qlora-파인튜닝-실험-설계-qlora-fine-tuning-design)
6. [AWQ 양자화 실험 설계 (AWQ Quantization Design)](#6-awq-양자화-실험-설계-awq-quantization-design)
7. [평가 메트릭 및 벤치마크 (Evaluation Metrics)](#7-평가-메트릭-및-벤치마크-evaluation-metrics)
8. [실험 추적 및 로깅 (Experiment Tracking)](#8-실험-추적-및-로깅-experiment-tracking)
9. [컴퓨팅 자원 계획 (Computing Resources)](#9-컴퓨팅-자원-계획-computing-resources)
10. [실험 일정 및 마일스톤 (Timeline)](#10-실험-일정-및-마일스톤-timeline)
11. [코드 구조 및 재현성 (Reproducibility)](#11-코드-구조-및-재현성-reproducibility)

---

## 1. 실험 개요 (Experiment Overview)

### 1.1 연구 목적 및 가설

#### 연구 목적
본 실험은 한국어 특화 LLM인 EXAONE-3.5-7.8B-Instruct 모델을 민원 도메인에 특화하여 파인튜닝하고, 온프레미스 환경 배포를 위한 최적의 양자화 기법을 검증하는 것을 목표로 합니다.

#### 핵심 가설
1. **H1 (QLoRA 효과성)**: QLoRA 기법을 사용하여 4-bit 양자화 상태에서 파인튜닝해도 민원 분류 및 답변 생성 성능이 85% 이상의 정확도를 달성할 수 있다.
2. **H2 (도메인 적응)**: AI Hub 공공 민원 데이터셋으로 파인튜닝하면 일반 도메인 대비 민원 처리 태스크에서 30%p 이상의 성능 향상이 발생한다.
3. **H3 (AWQ 양자화 효율성)**: AWQ 4-bit 양자화 적용 시 모델 크기 50% 이상 감소, 추론 속도 2배 이상 향상을 달성하면서도 성능 저하가 5% 미만으로 유지된다.
4. **H4 (Chat Template 최적화)**: EXAONE 표준 Chat Template (`[|user|]`, `[|assistant|]`) 적용 시 일반 프롬프트 대비 답변 품질이 향상된다.

### 1.2 기대 효과 및 성과 지표

#### 기대 효과
- **업무 효율성**: 민원 처리 시간 60% 이상 단축 (평균 15분 → 3분 이하)
- **모델 경량화**: VRAM 사용량 50% 감소 (bfloat16 15GB → AWQ 8GB 미만)
- **추론 속도**: p50 응답 시간 2초 이하, p95 응답 시간 5초 이하 달성
- **비용 절감**: 클라우드 API 대비 연간 90% 비용 절감

#### 성과 지표 (KPI)
| 지표 | 베이스라인 | 목표값 | 측정 방법 |
|------|-----------|--------|----------|
| 민원 분류 정확도 | 55% (키워드 기반) | ≥85% | Test set accuracy |
| 답변 생성 BLEU | N/A | ≥30 | BLEU-4 score |
| 답변 생성 ROUGE-L | N/A | ≥40 | ROUGE-L F1 |
| 추론 속도 (p50) | N/A | <2초 | vLLM 벤치마크 |
| GPU VRAM 사용량 | 15GB (bf16) | <8GB (AWQ) | nvidia-smi |
| 모델 파일 크기 | 15.6GB (bf16) | <8GB (AWQ) | 파일 크기 측정 |

### 1.3 실험 범위 및 제한사항

#### 실험 범위
- **모델**: EXAONE-3.5-7.8B-Instruct (단일 모델 집중 연구)
- **파인튜닝 기법**: QLoRA (4-bit NF4, LoRA rank 실험)
- **양자화 기법**: AWQ (4-bit, activation-aware)
- **데이터셋**: AI Hub 공공 민원 상담 LLM 데이터 (71852) + 민간 민원 데이터 (71844)
- **태스크**: 민원 분류, 답변 생성 (multi-task learning)

#### 제한사항
1. **데이터 규모**: 최대 100,000건 (Colab 디스크 용량 제약)
2. **학습 시간**: Colab Pro A100 최대 24시간 런타임 제약
3. **평가 기준**: 실제 공무원 만족도 평가는 제외 (자동 평가 지표만 사용)
4. **언어**: 한국어 민원 데이터만 다룸 (다국어 미지원)
5. **도메인**: 지자체 민원으로 한정 (중앙행정 제외)

---

## 2. 모델 및 데이터셋 구성 (Model & Dataset Configuration)

### 2.1 베이스 모델: EXAONE-3.5-7.8B-Instruct

#### 모델 선정 근거
| 평가 기준 | 상세 내용 |
|----------|----------|
| **한국어 성능** | 한국 표준 테스트 (CSAT Math 2025) 89.9% 달성 |
| **모델 크기** | 7.8B 파라미터 - A100 40GB 환경에 최적 |
| **컨텍스트 길이** | 32,768 토큰 - 긴 민원 및 유사 사례 참조 가능 |
| **아키텍처** | Grouped Query Attention (GQA) - 추론 효율성 우수 |
| **라이선스** | EXAONE AI Model License 1.1-NC (연구/비상업적 사용 허용) |
| **공식 지원** | Hugging Face Transformers, PEFT, vLLM 공식 지원 |

#### 모델 상세 정보
```python
MODEL_CONFIG = {
    "model_id": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "model_type": "exaone",
    "architecture": "ExaoneForCausalLM",
    "num_parameters": "7.8B",
    "vocab_size": 102400,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,  # GQA
    "max_position_embeddings": 32768,
    "rope_theta": 10000.0,
    "torch_dtype": "bfloat16",
    "use_cache": True
}
```

#### Chat Template 구조
```python
EXAONE_CHAT_TEMPLATE = """[|system|]
{system_message}
[|user|]
{user_message}
[|assistant|]
{assistant_message}[|endofturn|]"""

# 민원 응답 생성용 시스템 프롬프트
CIVIL_COMPLAINT_SYSTEM_PROMPT = """당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다.
민원 내용을 단계적으로 분석하고, 공손하고 명확한 표준 답변을 작성해야 합니다.

답변 작성 원칙:
1. 민원인의 요청사항을 정확히 파악합니다.
2. 관련 법규 및 정책을 근거로 답변합니다.
3. 공손하고 존중하는 어투를 사용합니다.
4. 담당부서와 연락처를 명시합니다."""
```

### 2.2 데이터셋 구성

#### 2.2.1 AI Hub 데이터셋 (우선순위 기반 선정)

| 데이터셋 번호 | 명칭 | 예상 규모 | 사용 목적 | 우선순위 |
|--------------|------|----------|----------|---------|
| **71852** | **공공 민원 상담 LLM 데이터** | 150,000건+ | **주 학습 데이터** (Instruction Tuning) | 1 |
| **71844** | **민간 민원 상담 LLM 데이터** | 200,000건+ | 보조 학습 데이터 (도메인 확장) | 2 |
| 98 | 민원(콜센터) 질의-응답 데이터 | 100,000건+ | 단답형 QA 보강 | 3 |
| 619 | 민원 업무 자동화 언어 데이터 | 100,000건+ | 전문 용어 학습 | 4 |

#### 2.2.2 데이터셋 수집 전략
```bash
# AI Hub 데이터 다운로드 (aihubshell 사용)
# 우선순위 1: 공공 민원 상담 LLM 데이터
./aihubshell -mode d -datasetkey 71852

# 우선순위 2: 민간 민원 상담 LLM 데이터
./aihubshell -mode d -datasetkey 71844

# 목표: 최소 50,000건, 최대 100,000건 확보
```

#### 2.2.3 데이터 포맷 예시
```json
{
  "id": "CIVIL_00001",
  "instruction": "다음 민원에 대해 단계적으로 분석하고, 표준 서식에 맞춰 공손하고 명확한 답변을 작성하세요.",
  "input": "[카테고리: 도로/교통]\n민원 내용: 우리 동네 횡단보도에 신호등이 없어서 어르신들이 건너기 위험합니다. 신호등 설치를 요청드립니다.",
  "output": "안녕하십니까. 민원을 제출해 주셔서 감사합니다.\n\n말씀하신 횡단보도 신호등 설치 건에 대해 답변드리겠습니다. 해당 구간은 교통량과 보행량을 고려하여 신호등 설치 필요성을 검토하고 있습니다.\n\n현장 조사 후 교통안전시설 심의를 거쳐 2026년 상반기 내 신호등 설치를 추진할 예정입니다.\n\n담당: 도로교통과 (☎ 02-1234-5678)\n감사합니다.",
  "category": "도로/교통",
  "source": "aihub_71852",
  "original_question_length": 67,
  "original_answer_length": 184
}
```

#### 2.2.4 데이터 분할 비율
```python
SPLIT_RATIOS = {
    "train": 0.80,      # 80% 학습용
    "validation": 0.10, # 10% 검증용
    "test": 0.10        # 10% 평가용
}

# 예상 데이터 규모 (목표: 50,000건)
EXPECTED_DATA_SIZE = {
    "train": 40000,
    "validation": 5000,
    "test": 5000
}

# AWQ 캘리브레이션 데이터셋
CALIBRATION_CONFIG = {
    "num_samples": 512,
    "seq_length": 2048,
    "random_seed": 42
}
```

### 2.3 토크나이저 설정

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    trust_remote_code=True
)

# 토크나이저 특수 설정
TOKENIZER_CONFIG = {
    "padding_side": "left",          # Causal LM에 적합
    "truncation_side": "right",      # 긴 텍스트는 오른쪽 절단
    "max_length": 2048,              # 학습 시 최대 길이
    "add_eos_token": True,           # EOS 토큰 자동 추가
    "special_tokens": {
        "bos_token": "[|startoftext|]",
        "eos_token": "[|endoftext|]",
        "pad_token": "[PAD]"
    }
}
```

### 2.4 데이터 전처리 파이프라인

#### 전처리 단계
1. **PII 마스킹**: 이름, 주민번호, 전화번호, 주소 등 개인정보 자동 탐지 및 마스킹
2. **데이터 정제**: 중복 제거, 길이 필터링 (최소 20자 이상)
3. **품질 검증**: 민원-답변 쌍 매칭 확인, 특수문자 정규화
4. **포맷 변환**: EXAONE Chat Template 형식으로 변환
5. **토큰화**: 최대 2048 토큰으로 제한 (컨텍스트 윈도우 고려)

#### 실행 코드 예시
```python
# 프로젝트 내 전처리 파이프라인 사용
from src.data_collection_preprocessing.pipeline import DataPipeline

pipeline = DataPipeline()

# Mock 데이터로 테스트 (Colab 환경 확인용)
result = pipeline.run_full_pipeline(
    use_mock=True,
    mock_samples=1000,
    output_prefix="civil_complaint"
)

# 실제 데이터로 실행 (AI Hub 다운로드 후)
result = pipeline.run_full_pipeline(
    use_mock=False,
    max_seoul_records=10000,
    output_prefix="civil_complaint"
)
```

---

## 3. 실험 환경 설정 (Environment Setup)

### 3.1 Google Colab Pro A100 환경 명세

#### 하드웨어 사양
| 구성요소 | 사양 | 비고 |
|----------|------|------|
| **GPU** | NVIDIA A100 40GB | Tensor Core 활용 |
| **System RAM** | 83GB | High-RAM 런타임 |
| **Disk Space** | 225GB | 모델 + 데이터 + 체크포인트 |
| **CPU** | Intel Xeon 2.2GHz (12 cores) | 데이터 전처리용 |

#### 런타임 제약사항
- **최대 실행 시간**: 24시간 (Colab Pro)
- **세션 타임아웃**: 90분 비활성 시 종료
- **GPU 할당 제한**: 연속 사용 시 제한 가능성 있음

### 3.2 필수 라이브러리 설치

#### 설치 스크립트 (Colab 노트북 첫 셀)
```bash
%%bash

# 시스템 업데이트
apt-get update -qq

# Git LFS (대용량 모델 다운로드용)
apt-get install -y git-lfs
git lfs install

# Python 라이브러리 설치
pip install -q --upgrade pip

# Core ML libraries
pip install -q torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face ecosystem
pip install -q transformers==4.40.0
pip install -q datasets==2.18.0
pip install -q accelerate==0.28.0
pip install -q peft==0.10.0

# Quantization
pip install -q bitsandbytes==0.43.0  # QLoRA 4-bit
pip install -q autoawq==0.2.0        # AWQ quantization
pip install -q optimum==1.17.0

# Training utilities
pip install -q trl==0.8.1            # SFTTrainer
pip install -q einops==0.7.0
pip install -q sentencepiece==0.2.0
pip install -q protobuf==4.25.0

# Evaluation
pip install -q evaluate==0.4.1
pip install -q rouge-score==0.1.2
pip install -q sacrebleu==2.4.0

# Monitoring
pip install -q wandb==0.16.4
pip install -q tensorboard==2.16.2

# Utilities
pip install -q python-dotenv==1.0.1
pip install -q tqdm==4.66.2
```

#### 버전 명세서 (requirements.txt)
```txt
# Core
torch==2.1.2
transformers==4.40.0
datasets==2.18.0
accelerate==0.28.0

# Fine-tuning
peft==0.10.0
bitsandbytes==0.43.0
trl==0.8.1

# Quantization
autoawq==0.2.0
optimum==1.17.0

# Evaluation
evaluate==0.4.1
rouge-score==0.1.2
sacrebleu==2.4.0

# Monitoring
wandb==0.16.4
tensorboard==2.16.2

# Utilities
einops==0.7.0
sentencepiece==0.2.0
protobuf==4.25.0
python-dotenv==1.0.1
tqdm==4.66.2
```

### 3.3 프로젝트 저장소 클론 및 설정

#### Colab에서 프로젝트 클론
```python
%%bash

# 프로젝트 클론
cd /content
git clone https://github.com/YOUR_USERNAME/ondevice-ai-civil-complaint.git
cd ondevice-ai-civil-complaint

# 환경 변수 설정
cat > .env << 'EOF'
# AI Hub API Key (AI Hub 가입 후 발급)
AIHUB_API_KEY=YOUR_AIHUB_API_KEY

# Seoul Open Data API Key (선택사항)
SEOUL_API_KEY=YOUR_SEOUL_API_KEY

# Weights & Biases (실험 추적)
WANDB_API_KEY=YOUR_WANDB_API_KEY
WANDB_PROJECT=exaone-civil-complaint
WANDB_ENTITY=YOUR_USERNAME

# Logging
LOG_LEVEL=INFO
EOF

# 디렉토리 구조 확인
tree -L 2 -d
```

#### 디렉토리 구조
```
/content/ondevice-ai-civil-complaint/
├── data/
│   ├── raw/              # AI Hub 다운로드 데이터
│   ├── processed/        # 전처리 완료 데이터
│   └── calibration/      # AWQ 캘리브레이션 데이터
├── models/
│   ├── base/             # 베이스 모델 (EXAONE)
│   ├── checkpoints/      # 학습 체크포인트
│   ├── merged/           # LoRA 병합 모델
│   └── quantized/        # AWQ 양자화 모델
├── src/
│   ├── data_collection_preprocessing/
│   ├── training/         # 학습 스크립트 (생성 예정)
│   └── evaluation/       # 평가 스크립트 (생성 예정)
├── notebooks/            # Colab 노트북 (생성 예정)
└── logs/                 # 실험 로그
```

### 3.4 GPU 설정 확인

```python
import torch

# GPU 가용성 확인
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# bfloat16 지원 확인 (A100 필수 기능)
print(f"bfloat16 Supported: {torch.cuda.is_bf16_supported()}")

# Expected Output:
# CUDA Available: True
# CUDA Version: 12.1
# GPU Count: 1
# GPU Name: NVIDIA A100-SXM4-40GB
# GPU Memory: 42.48 GB
# bfloat16 Supported: True
```

---

## 4. 데이터 준비 파이프라인 (Data Preparation)

### 4.1 데이터 수집 스크립트 실행

#### Step 1: AI Hub 데이터 다운로드
```bash
# Colab 터미널에서 실행

# aihubshell 다운로드 (Linux 64-bit)
cd /content
wget https://api.aihub.or.kr/down/aihubshell_linux.tar.gz
tar -xzf aihubshell_linux.tar.gz
chmod +x aihubshell

# API 키 설정
export AIHUB_API_KEY="YOUR_API_KEY"

# 데이터셋 다운로드 (71852: 공공 민원 상담 LLM 데이터)
./aihubshell -mode d -datasetkey 71852

# 다운로드 확인
ls -lh ~/aihub/
```

#### Step 2: 프로젝트 데이터 파이프라인 실행
```python
# Colab 노트북에서 실행
import sys
sys.path.append('/content/ondevice-ai-civil-complaint')

from src.data_collection_preprocessing.pipeline import DataPipeline
from src.data_collection_preprocessing.config import get_config

# 설정 로드
config = get_config()

# 파이프라인 초기화
pipeline = DataPipeline(config)

# Option 1: Mock 데이터로 테스트 (빠른 검증)
print("Testing with mock data...")
result = pipeline.run_full_pipeline(
    use_mock=True,
    mock_samples=1000,
    output_prefix="test_civil_complaint"
)

# Option 2: 실제 데이터로 실행
print("Running with real data...")
result = pipeline.run_full_pipeline(
    use_mock=False,
    output_prefix="civil_complaint"
)

# 결과 확인
print(f"\nPipeline Success: {result.success}")
print(f"Total Raw Records: {result.total_raw_records}")
print(f"Total Processed Records: {result.total_processed_records}")
print(f"Duration: {result.duration_seconds:.2f} seconds")

# 품질 리포트 출력
print("\n" + pipeline.get_quality_report())
```

### 4.2 전처리 파이프라인 상세

#### PII 마스킹 규칙
```python
# src/data_collection_preprocessing/pii_masking.py 활용
PII_PATTERNS = {
    "이름": r"[가-힣]{2,4}(?=\s*님|\s*씨|\s*선생|\s*의원)",
    "주민번호": r"\d{6}[-\s]?\d{7}",
    "전화번호": r"0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}",
    "휴대폰": r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}",
    "이메일": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "주소": r"(서울|경기|인천|부산|대구|광주|대전|울산|세종|강원|충북|충남|전북|전남|경북|경남|제주)[^\s]{2,}(시|군|구)[^\s]{2,}(동|읍|면)"
}

# 마스킹 예시
# 원본: "김철수 님께서 02-1234-5678로 문의하셨습니다."
# 마스킹: "[이름] 님께서 [전화번호]로 문의하셨습니다."
```

#### 데이터 품질 필터링
```python
QUALITY_FILTERS = {
    "min_complaint_length": 20,    # 최소 민원 길이 (글자수)
    "min_answer_length": 10,       # 최소 답변 길이
    "max_text_length": 4096,       # 최대 텍스트 길이
    "remove_duplicates": True,     # 중복 제거
    "remove_empty": True,          # 빈 값 제거
    "normalize_whitespace": True   # 공백 정규화
}

# 중복 제거 로직: MD5 해시 기반
# 예상 중복률: 5-10%
```

### 4.3 캘리브레이션 데이터셋 생성

```python
# AWQ 양자화를 위한 캘리브레이션 데이터셋 생성
from src.data_collection_preprocessing.calibration_dataset import (
    CalibrationDatasetGenerator
)

# 생성기 초기화
calibration_gen = CalibrationDatasetGenerator(config.calibration)

# 캘리브레이션 데이터셋 생성
calibration_paths = calibration_gen.generate_and_save(
    processed_records=pipeline.processed_records,
    filename="exaone_civil_calibration"
)

print(f"Calibration dataset saved:")
for key, path in calibration_paths.items():
    print(f"  {key}: {path}")

# Expected output:
# /content/ondevice-ai-civil-complaint/data/calibration/exaone_civil_calibration.json
# /content/ondevice-ai-civil-complaint/data/calibration/exaone_civil_calibration.txt
# /content/ondevice-ai-civil-complaint/data/calibration/exaone_civil_calibration_stats.json
```

### 4.4 예상 데이터 크기 및 형식

| 항목 | 예상 크기 | 포맷 | 비고 |
|------|----------|------|------|
| Raw Data (AI Hub) | 50,000건 | JSON | 다운로드 원본 |
| Processed Train | 40,000건 (~500MB) | JSONL | 학습용 |
| Processed Validation | 5,000건 (~60MB) | JSONL | 검증용 |
| Processed Test | 5,000건 (~60MB) | JSONL | 평가용 |
| Calibration Dataset | 512건 (~6MB) | TXT/JSON | AWQ 양자화용 |

---

## 5. QLoRA 파인튜닝 실험 설계 (QLoRA Fine-tuning Design)

### 5.1 QLoRA 개요

**QLoRA (Quantized Low-Rank Adaptation)**는 대형 언어 모델을 효율적으로 파인튜닝하기 위한 기법입니다:
- **4-bit NormalFloat (NF4)**: 사전학습 가중치를 4-bit로 양자화하여 메모리 사용량 75% 감소
- **LoRA 어댑터**: 전체 모델 대신 저차원 행렬만 학습하여 학습 파라미터 99% 감소
- **Double Quantization**: 양자화 상수도 양자화하여 추가 메모리 절약
- **Paged Optimizers**: CPU-GPU 메모리 스왑으로 OOM 방지

### 5.2 QLoRA 하이퍼파라미터 설정

#### 기본 설정 (Baseline)
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit 로딩
    bnb_4bit_quant_type="nf4",            # NormalFloat4 데이터 타입
    bnb_4bit_compute_dtype=torch.bfloat16, # 계산은 bfloat16
    bnb_4bit_use_double_quant=True,       # Double quantization
)

# LoRA 설정 (Baseline)
lora_config = LoraConfig(
    r=16,                          # LoRA rank (저차원 행렬 차원)
    lora_alpha=32,                 # LoRA scaling factor (alpha/r = 2.0)
    target_modules=[               # 적용할 레이어
        "q_proj",   # Query projection
        "k_proj",   # Key projection
        "v_proj",   # Value projection
        "o_proj",   # Output projection
        "gate_proj", # MLP gate
        "up_proj",   # MLP up
        "down_proj"  # MLP down
    ],
    lora_dropout=0.05,             # LoRA 레이어 dropout
    bias="none",                   # Bias 학습 안함
    task_type="CAUSAL_LM",         # Causal Language Modeling
    inference_mode=False,          # 학습 모드
)
```

#### 실험 변수 (Ablation Study)
```python
# 실험 1: LoRA Rank 탐색
LORA_RANK_EXPERIMENTS = [
    {"r": 8,  "lora_alpha": 16, "name": "rank8"},
    {"r": 16, "lora_alpha": 32, "name": "rank16_baseline"},
    {"r": 32, "lora_alpha": 64, "name": "rank32"},
    {"r": 64, "lora_alpha": 128, "name": "rank64"},
]

# 실험 2: Target Modules 탐색
TARGET_MODULE_EXPERIMENTS = [
    {
        "modules": ["q_proj", "v_proj"],
        "name": "qv_only"
    },
    {
        "modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "name": "attention_only_baseline"
    },
    {
        "modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
        "name": "attention_mlp_full"
    },
]

# 실험 3: LoRA Dropout 탐색
DROPOUT_EXPERIMENTS = [
    {"lora_dropout": 0.0, "name": "dropout0"},
    {"lora_dropout": 0.05, "name": "dropout005_baseline"},
    {"lora_dropout": 0.1, "name": "dropout01"},
]
```

### 5.3 학습 하이퍼파라미터

#### Training Arguments (Baseline)
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # 출력 디렉토리
    output_dir="/content/ondevice-ai-civil-complaint/models/checkpoints/exaone-qlora-baseline",

    # 학습 설정
    num_train_epochs=3,                    # 에폭 수
    per_device_train_batch_size=4,         # 배치 크기 (A100 40GB 기준)
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,         # 실제 배치 = 4 * 4 = 16

    # 최적화
    learning_rate=2e-4,                    # 학습률
    lr_scheduler_type="cosine",            # Cosine annealing
    warmup_ratio=0.03,                     # Warmup 3%
    weight_decay=0.01,                     # Weight decay
    max_grad_norm=1.0,                     # Gradient clipping

    # 정밀도
    bf16=True,                             # bfloat16 학습
    tf32=True,                             # TF32 연산 (A100)

    # 메모리 최적화
    gradient_checkpointing=True,           # Gradient checkpointing
    optim="paged_adamw_8bit",              # 8-bit Adam optimizer

    # 로깅
    logging_steps=10,
    logging_dir="/content/logs",
    report_to="wandb",                     # Weights & Biases

    # 평가
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,                    # 최대 3개 체크포인트 유지

    # 기타
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42,
    data_seed=42,
    remove_unused_columns=False,
)
```

#### 학습률 스케줄 실험
```python
LEARNING_RATE_EXPERIMENTS = [
    {
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "name": "lr1e4_cosine"
    },
    {
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "name": "lr2e4_cosine_baseline"
    },
    {
        "learning_rate": 3e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "name": "lr3e4_cosine"
    },
    {
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.05,
        "name": "lr2e4_linear"
    },
]
```

### 5.4 학습 스크립트 (Full Implementation)

```python
# train_qlora.py
import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

# Weights & Biases 초기화
wandb.init(
    project="exaone-civil-complaint",
    name="qlora-baseline-run1",
    config={
        "model": "EXAONE-3.5-7.8B-Instruct",
        "method": "QLoRA",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 16,  # effective
        "epochs": 3
    }
)

# 1. 모델 및 토크나이저 로드
print("Loading model and tokenizer...")
model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # SFT 시 right padding

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# LoRA 준비
model = prepare_model_for_kbit_training(model)

# 2. LoRA 어댑터 적용
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params: ~42M / total params: ~7.8B (0.54%)

# 3. 데이터셋 로드
print("Loading dataset...")
data_files = {
    "train": "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_train.jsonl",
    "validation": "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_val.jsonl"
}
dataset = load_dataset("json", data_files=data_files)

# 4. Chat Template 포맷팅
def format_chat_template(example):
    """EXAONE Chat Template 포맷"""
    messages = [
        {"role": "system", "content": "당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다."},
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example['output']}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

formatted_train = dataset["train"].map(format_chat_template)
formatted_val = dataset["validation"].map(format_chat_template)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="/content/models/checkpoints/exaone-qlora-baseline",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    logging_dir="/content/logs",
    report_to="wandb",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=42,
)

# 6. Trainer 초기화
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train,
    eval_dataset=formatted_val,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,  # Packing 비활성화 (긴 민원 처리)
)

# 7. 학습 시작
print("Starting training...")
trainer.train()

# 8. 최종 모델 저장
print("Saving final model...")
trainer.save_model("/content/models/checkpoints/exaone-qlora-baseline/final")

# 9. LoRA 어댑터만 저장 (경량)
model.save_pretrained("/content/models/checkpoints/exaone-qlora-baseline/lora_adapter")
tokenizer.save_pretrained("/content/models/checkpoints/exaone-qlora-baseline/lora_adapter")

print("Training complete!")
wandb.finish()
```

### 5.5 Ablation Study 실험 계획

| 실험 ID | 변경 변수 | 설정값 | 목적 | 예상 소요 시간 |
|---------|----------|--------|------|---------------|
| EXP-001 | Baseline | r=16, lr=2e-4 | 기준 성능 측정 | 6시간 |
| EXP-002 | LoRA Rank | r=8 | 경량화 효과 검증 | 5시간 |
| EXP-003 | LoRA Rank | r=32 | 성능 향상 검증 | 7시간 |
| EXP-004 | LoRA Rank | r=64 | 성능 포화점 확인 | 8시간 |
| EXP-005 | Learning Rate | lr=1e-4 | 안정성 검증 | 6시간 |
| EXP-006 | Learning Rate | lr=3e-4 | 빠른 수렴 검증 | 6시간 |
| EXP-007 | Target Modules | qv_only | 최소 설정 검증 | 5시간 |
| EXP-008 | Dropout | 0.1 | Overfitting 방지 | 6시간 |

**총 실험 소요 시간**: 약 49시간 (Colab Pro 2-3일 분산 실행)

### 5.6 예상 학습 메트릭

#### Loss Curve 예상값
| Epoch | Train Loss | Validation Loss | Perplexity |
|-------|-----------|----------------|------------|
| 0 (Initial) | 2.50 | 2.48 | 11.97 |
| 1 | 1.20 | 1.35 | 3.86 |
| 2 | 0.85 | 1.15 | 3.16 |
| 3 | 0.65 | 1.10 | 3.00 |

---

## 6. AWQ 양자화 실험 설계 (AWQ Quantization Design)

### 6.1 AWQ 개요

**AWQ (Activation-aware Weight Quantization)**는 활성화 값의 분포를 고려하여 가중치를 양자화하는 기법입니다:
- **Activation-aware**: 중요한 가중치는 보호하고 덜 중요한 가중치만 양자화
- **Per-channel Scaling**: 채널별 스케일링으로 정확도 유지
- **INT4 양자화**: 4-bit 정수 양자화로 모델 크기 75% 감소
- **vLLM 호환**: vLLM 서빙 프레임워크와 완벽 호환

### 6.2 AWQ 양자화 설정

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# AWQ 양자화 설정
awq_config = {
    "zero_point": True,        # Zero-point 양자화 활성화
    "q_group_size": 128,       # 그룹 크기 (128 권장)
    "w_bit": 4,                # Weight bit (4-bit)
    "version": "GEMM",         # GEMM 커널 사용
}

# 캘리브레이션 설정
calibration_config = {
    "calib_data": "/content/ondevice-ai-civil-complaint/data/calibration/exaone_civil_calibration.txt",
    "n_samples": 512,          # 캘리브레이션 샘플 수
    "seq_len": 2048,           # 시퀀스 길이
}
```

### 6.3 양자화 실험 계획

#### 실험 1: Group Size 탐색
```python
GROUP_SIZE_EXPERIMENTS = [
    {"q_group_size": 64, "name": "group64"},
    {"q_group_size": 128, "name": "group128_baseline"},
    {"q_group_size": 256, "name": "group256"},
]
```

#### 실험 2: Zero-Point 효과 검증
```python
ZERO_POINT_EXPERIMENTS = [
    {"zero_point": False, "name": "no_zeropoint"},
    {"zero_point": True, "name": "with_zeropoint_baseline"},
]
```

### 6.4 AWQ 양자화 스크립트

```python
# quantize_awq.py
import os
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

print("=" * 60)
print("AWQ Quantization for EXAONE-3.5-7.8B-Instruct")
print("=" * 60)

# 1. 파인튜닝된 모델 로드 (LoRA 병합 완료 상태)
model_path = "/content/models/merged/exaone-qlora-baseline-merged"
quant_path = "/content/models/quantized/exaone-qlora-awq-4bit"

print(f"\n[1/5] Loading fine-tuned model from: {model_path}")
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 캘리브레이션 데이터 로드
print("\n[2/5] Loading calibration dataset...")
calib_file = "/content/ondevice-ai-civil-complaint/data/calibration/exaone_civil_calibration.txt"

with open(calib_file, "r", encoding="utf-8") as f:
    calib_data = [line.strip() for line in f if line.strip()]

# 토큰화
print(f"Tokenizing {len(calib_data)} calibration samples...")
calib_samples = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).input_ids
    for text in calib_data[:512]
]

# 3. AWQ 양자화 수행
print("\n[3/5] Performing AWQ quantization...")
print("Configuration:")
print(f"  - Weight bits: 4")
print(f"  - Group size: 128")
print(f"  - Zero-point: True")
print(f"  - Calibration samples: 512")

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_samples
)

# 4. 양자화 모델 저장
print(f"\n[4/5] Saving quantized model to: {quant_path}")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# 5. 통계 출력
print("\n[5/5] Quantization Statistics:")
original_size = sum(
    p.numel() * p.element_size() for p in model.model.parameters()
) / (1024 ** 3)  # GB

quantized_size = os.path.getsize(
    os.path.join(quant_path, "model.safetensors")
) / (1024 ** 3)  # GB

print(f"  Original model size: {original_size:.2f} GB")
print(f"  Quantized model size: {quantized_size:.2f} GB")
print(f"  Compression ratio: {original_size / quantized_size:.2f}x")
print(f"  Size reduction: {(1 - quantized_size / original_size) * 100:.1f}%")

print("\n" + "=" * 60)
print("AWQ Quantization Complete!")
print("=" * 60)
```

### 6.5 양자화 전후 성능 비교 계획

| 모델 버전 | 크기 | VRAM | 추론 속도 (p50) | 정확도 예상 |
|----------|------|------|----------------|------------|
| Base (bf16) | 15.6 GB | 15 GB | 2.5초 | 100% (기준) |
| QLoRA Finetuned (bf16) | 15.6 GB | 15 GB | 2.5초 | 100% |
| AWQ 4-bit | 4.2 GB | 6 GB | 1.2초 | ≥95% |

---

## 7. 평가 메트릭 및 벤치마크 (Evaluation Metrics)

### 7.1 태스크별 평가 지표

#### Task 1: 민원 분류 (Classification)
```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

CLASSIFICATION_METRICS = {
    "accuracy": "전체 정확도",
    "macro_f1": "클래스 균형 F1 점수",
    "weighted_f1": "샘플 가중 F1 점수",
    "per_class_precision": "클래스별 정밀도",
    "per_class_recall": "클래스별 재현율",
}

# 예상 성능 목표
TARGET_CLASSIFICATION = {
    "accuracy": 0.85,      # 85% 이상
    "macro_f1": 0.80,      # 80% 이상
    "weighted_f1": 0.85,   # 85% 이상
}
```

#### Task 2: 답변 생성 품질 (Generation Quality)
```python
from evaluate import load

# BLEU Score (답변 일치도)
bleu = load("sacrebleu")

# ROUGE Score (요약 품질)
rouge = load("rouge")

GENERATION_METRICS = {
    "bleu_4": "BLEU-4 점수 (n-gram overlap)",
    "rouge_l": "ROUGE-L F1 (longest common subsequence)",
    "rouge_1": "ROUGE-1 (unigram overlap)",
    "rouge_2": "ROUGE-2 (bigram overlap)",
}

# 예상 성능 목표
TARGET_GENERATION = {
    "bleu_4": 30.0,        # BLEU ≥ 30
    "rouge_l": 40.0,       # ROUGE-L ≥ 40
    "rouge_1": 45.0,       # ROUGE-1 ≥ 45
    "rouge_2": 25.0,       # ROUGE-2 ≥ 25
}
```

#### Task 3: 언어 모델 Perplexity
```python
import torch
import numpy as np

def compute_perplexity(model, tokenizer, texts):
    """Perplexity 계산"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity

# 예상 Perplexity 목표
TARGET_PERPLEXITY = {
    "validation": 3.0,     # PPL ≤ 3.0
    "test": 3.2,           # PPL ≤ 3.2
}
```

### 7.2 효율성 지표

#### 추론 속도 벤치마크
```python
import time

def benchmark_inference(model, tokenizer, prompts, num_runs=100):
    """추론 속도 벤치마킹"""
    latencies = []

    for prompt in prompts[:num_runs]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.95,
                do_sample=True
            )
        end = time.time()

        latencies.append((end - start) * 1000)  # ms

    return {
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "mean": np.mean(latencies),
        "std": np.std(latencies),
    }

# 예상 추론 속도 목표
TARGET_LATENCY = {
    "bf16": {"p50": 2500, "p95": 5000},    # ms
    "awq_4bit": {"p50": 1200, "p95": 2500}, # ms
}
```

#### 메모리 사용량 측정
```python
def measure_memory_usage(model):
    """GPU 메모리 사용량 측정"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        }
    return None

# 예상 메모리 사용량
TARGET_MEMORY = {
    "bf16": {"max_allocated_gb": 15.0},
    "awq_4bit": {"max_allocated_gb": 6.0},
}
```

### 7.3 평가 스크립트

```python
# evaluate_model.py
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from tqdm import tqdm

print("=" * 60)
print("Model Evaluation")
print("=" * 60)

# 1. 모델 및 데이터 로드
model_path = "/content/models/quantized/exaone-qlora-awq-4bit"
test_data_path = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"

print(f"\nLoading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f"Loading test dataset from: {test_data_path}")
test_dataset = load_dataset("json", data_files=test_data_path)["train"]

# 2. 평가 메트릭 초기화
bleu = load("sacrebleu")
rouge = load("rouge")

# 3. 추론 및 평가
print("\nRunning inference on test set...")
predictions = []
references = []
categories_true = []
categories_pred = []

for example in tqdm(test_dataset):
    # 프롬프트 생성
    prompt = f"[|user|]\n{example['instruction']}\n\n{example['input']}\n[|assistant|]\n"

    # 추론
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.95,
            do_sample=False  # Greedy decoding for evaluation
        )

    # 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_answer = generated_text.split("[|assistant|]")[-1].strip()

    predictions.append(predicted_answer)
    references.append(example['output'])

    # 카테고리 추출 (간단 구현)
    categories_true.append(example['category'])
    # TODO: 카테고리 예측 로직 추가

# 4. BLEU/ROUGE 계산
print("\nComputing BLEU scores...")
bleu_score = bleu.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

print("Computing ROUGE scores...")
rouge_scores = rouge.compute(
    predictions=predictions,
    references=references
)

# 5. 결과 출력
results = {
    "bleu": bleu_score["score"],
    "rouge_1": rouge_scores["rouge1"] * 100,
    "rouge_2": rouge_scores["rouge2"] * 100,
    "rouge_l": rouge_scores["rougeL"] * 100,
}

print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)
print(json.dumps(results, indent=2, ensure_ascii=False))

# 6. 결과 저장
output_path = "/content/evaluation_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_path}")
```

---

## 8. 실험 추적 및 로깅 (Experiment Tracking)

### 8.1 Weights & Biases 설정

```python
import wandb

# W&B 초기화
wandb.login(key=os.getenv("WANDB_API_KEY"))

wandb.init(
    project="exaone-civil-complaint",
    entity="YOUR_USERNAME",
    name="qlora-baseline-exp001",
    config={
        # 모델 설정
        "model_id": "EXAONE-3.5-7.8B-Instruct",
        "model_size": "7.8B",

        # QLoRA 설정
        "method": "QLoRA",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],

        # 학습 설정
        "learning_rate": 2e-4,
        "batch_size": 16,  # effective
        "num_epochs": 3,
        "warmup_ratio": 0.03,
        "lr_scheduler": "cosine",

        # 데이터셋
        "dataset": "AI Hub 71852 + 71844",
        "train_samples": 40000,
        "val_samples": 5000,

        # 환경
        "gpu": "A100 40GB",
        "precision": "bfloat16",
    },
    tags=["qlora", "baseline", "exaone", "civil-complaint"]
)
```

### 8.2 체크포인트 저장 전략

```python
CHECKPOINT_STRATEGY = {
    # 저장 간격
    "save_steps": 500,                # 500 스텝마다 저장
    "save_total_limit": 3,            # 최대 3개 체크포인트 유지
    "load_best_model_at_end": True,   # 최고 성능 모델 로드

    # 저장 경로 구조
    "checkpoint_dir": "/content/models/checkpoints/{experiment_name}/",

    # 메타데이터 저장
    "save_metadata": True,
    "metadata_fields": [
        "epoch", "step", "eval_loss", "eval_accuracy",
        "learning_rate", "timestamp"
    ]
}

# 체크포인트 디렉토리 예시
# /content/models/checkpoints/exaone-qlora-baseline/
# ├── checkpoint-500/
# ├── checkpoint-1000/
# ├── checkpoint-1500/
# └── final/
```

### 8.3 실험 결과 기록 양식

```python
# experiment_log.json (템플릿)
EXPERIMENT_LOG_TEMPLATE = {
    "experiment_id": "EXP-001",
    "experiment_name": "qlora-baseline",
    "timestamp": "2026-03-05T10:00:00",
    "status": "completed",  # running, completed, failed

    "config": {
        "model": "EXAONE-3.5-7.8B-Instruct",
        "method": "QLoRA",
        "hyperparameters": {
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "batch_size": 16,
            "epochs": 3
        }
    },

    "results": {
        "train_loss": 0.65,
        "eval_loss": 1.10,
        "perplexity": 3.00,
        "accuracy": 0.87,
        "bleu_4": 32.5,
        "rouge_l": 42.1
    },

    "metrics": {
        "training_time_hours": 6.5,
        "gpu_memory_peak_gb": 38.2,
        "total_steps": 7500,
        "samples_per_second": 4.8
    },

    "artifacts": {
        "checkpoint_path": "/content/models/checkpoints/exaone-qlora-baseline/final",
        "wandb_run_url": "https://wandb.ai/...",
        "tensorboard_log": "/content/logs/..."
    },

    "notes": "Baseline 실험. 안정적 수렴, Overfitting 없음."
}
```

---

## 9. 컴퓨팅 자원 계획 (Computing Resources)

### 9.1 GPU 메모리 요구사항

| 단계 | VRAM 사용량 | System RAM | 비고 |
|------|-----------|-----------|------|
| 데이터 전처리 | 0 GB | 8 GB | CPU only |
| 모델 로딩 (bf16) | 15.6 GB | 16 GB | Base model |
| 모델 로딩 (4-bit) | 4.5 GB | 12 GB | QLoRA loading |
| QLoRA 학습 | 32-38 GB | 24 GB | Optimizer states 포함 |
| 추론 (bf16) | 15 GB | 8 GB | Generation |
| 추론 (AWQ 4-bit) | 6 GB | 6 GB | Quantized |
| AWQ 양자화 | 12 GB | 16 GB | Calibration |

### 9.2 예상 학습 시간

```python
# 학습 시간 계산
TRAINING_TIME_ESTIMATION = {
    "dataset_size": 40000,
    "batch_size": 16,  # effective
    "num_epochs": 3,
    "steps_per_epoch": 40000 / 16,  # 2500
    "total_steps": 2500 * 3,        # 7500
    "time_per_step": 2.8,           # seconds (A100 기준)
    "total_time": 7500 * 2.8 / 3600, # hours
}

# 예상 학습 시간: 약 5.8 시간 (1 에폭당 2시간)
```

| 실험 | 에폭 | 예상 시간 | Colab 세션 |
|------|------|----------|-----------|
| Baseline (r=16) | 3 | 6시간 | 1 세션 |
| Rank=8 | 3 | 5시간 | 1 세션 |
| Rank=32 | 3 | 7시간 | 1 세션 |
| Rank=64 | 3 | 8시간 | 1 세션 |
| AWQ 양자화 | N/A | 1시간 | 1 세션 |

### 9.3 디스크 용량 계획

| 항목 | 용량 | 경로 |
|------|------|------|
| 베이스 모델 (bf16) | 15.6 GB | `/content/models/base/` |
| 학습 데이터 | 1.5 GB | `/content/data/processed/` |
| 체크포인트 (3개) | 48 GB | `/content/models/checkpoints/` |
| LoRA 어댑터 | 150 MB | `/content/models/checkpoints/.../lora_adapter/` |
| 병합 모델 | 15.6 GB | `/content/models/merged/` |
| AWQ 모델 | 4.2 GB | `/content/models/quantized/` |
| 로그 및 캐시 | 2 GB | `/content/logs/`, `~/.cache/` |
| **총 예상 용량** | **87 GB** | Colab 225GB 이내 |

### 9.4 잠재적 문제점 및 대응 방안

| 문제점 | 대응 방안 |
|--------|----------|
| **OOM (Out of Memory)** | - Gradient checkpointing 활성화<br>- Batch size 감소 (4→2)<br>- Gradient accumulation 증가 |
| **세션 타임아웃** | - 주기적 체크포인트 저장<br>- Google Drive 마운트하여 자동 백업<br>- Auto-reconnect 스크립트 사용 |
| **디스크 부족** | - 중간 체크포인트 삭제<br>- LoRA 어댑터만 저장 (150MB)<br>- 불필요한 캐시 정리 |
| **GPU 할당 제한** | - Colab Pro+ 업그레이드 고려<br>- 실험 스케줄링 (시간대 분산) |
| **학습 불안정** | - Learning rate 감소<br>- Warmup ratio 증가 (0.03→0.1)<br>- Gradient clipping 강화 |

---

## 10. 실험 일정 및 마일스톤 (Timeline)

### 10.1 Week 5: 모델 준비 및 데이터 수집

| Day | 작업 | 예상 시간 | 담당 | 산출물 |
|-----|------|----------|------|--------|
| D1 | Colab 환경 설정 및 라이브러리 설치 | 2h | 팀 전체 | `setup_colab.ipynb` |
| D1-2 | AI Hub 데이터 다운로드 (71852, 71844) | 4h | Data | Raw datasets |
| D2 | 데이터 전처리 파이프라인 실행 | 4h | Data | Processed JSONL |
| D3 | 캘리브레이션 데이터셋 생성 | 2h | Data | `calibration.txt` |
| D3-4 | 베이스 모델 다운로드 및 검증 | 2h | ML | EXAONE model |
| D4-5 | 학습 스크립트 작성 및 테스트 | 6h | ML | `train_qlora.py` |

**마일스톤 M5-1**: 데이터 및 모델 준비 완료 (D5 종료 시)

### 10.2 Week 6: QLoRA 파인튜닝 실험

| Day | 실험 | 설정 | 예상 시간 | 산출물 |
|-----|------|------|----------|--------|
| D6 | EXP-001: Baseline | r=16, lr=2e-4 | 6h | Checkpoint, WandB log |
| D7 | EXP-002: Rank=8 | r=8 | 5h | Checkpoint, WandB log |
| D8 | EXP-003: Rank=32 | r=32 | 7h | Checkpoint, WandB log |
| D9 | EXP-005: LR=1e-4 | lr=1e-4 | 6h | Checkpoint, WandB log |
| D10 | EXP-006: LR=3e-4 | lr=3e-4 | 6h | Checkpoint, WandB log |
| D10 | 중간 결과 분석 및 보고 | - | 2h | `week6_report.md` |

**마일스톤 M6-1**: QLoRA 실험 완료 및 Best 모델 선정 (D10 종료 시)

### 10.3 Week 7: AWQ 양자화 및 평가

| Day | 작업 | 예상 시간 | 산출물 |
|-----|------|----------|--------|
| D11 | Best 모델 LoRA 병합 | 1h | Merged model |
| D11 | AWQ 양자화 실행 (Baseline) | 1h | AWQ 4-bit model |
| D12 | AWQ 그룹 크기 실험 (64, 128, 256) | 3h | 3 quantized models |
| D13 | 양자화 전후 성능 비교 평가 | 4h | `evaluation_results.json` |
| D14 | 추론 속도 벤치마크 | 2h | `benchmark_results.json` |
| D14 | 메모리 사용량 측정 | 1h | `memory_profile.json` |

**마일스톤 M7-1**: AWQ 양자화 완료 및 성능 검증 (D14 종료 시)

### 10.4 Week 8: 통합 및 문서화

| Day | 작업 | 예상 시간 | 산출물 |
|-----|------|----------|--------|
| D15 | 최종 모델 선정 및 저장 | 2h | Final model artifacts |
| D15-16 | vLLM 서빙 테스트 (선택) | 4h | `vllm_serving_test.ipynb` |
| D16-17 | 평가 리포트 작성 | 6h | `evaluation_report.md` |
| D17-18 | 실험 계획서 업데이트 | 4h | `experiment_plan.md` (final) |
| D18 | 멘토 중간 점검 준비 | 2h | Presentation slides |

**마일스톤 M8-1**: MVP 완료 및 멘토 점검 통과 (D18 종료 시)

### 10.5 전체 일정 요약 (Gantt Chart)

```
Week 5: 데이터 및 모델 준비
[==============================] D1-D5

Week 6: QLoRA 파인튜닝 실험
[==============================] D6-D10
  ├─ Baseline (D6) [======]
  ├─ Rank=8 (D7)   [=====]
  ├─ Rank=32 (D8)  [=======]
  ├─ LR=1e-4 (D9)  [======]
  └─ LR=3e-4 (D10) [======]

Week 7: AWQ 양자화 및 평가
[==============================] D11-D14
  ├─ 병합 & 양자화 (D11-D12) [========]
  └─ 평가 & 벤치마크 (D13-D14) [=========]

Week 8: 통합 및 문서화
[==============================] D15-D18
  ├─ 모델 배포 준비 (D15-D16) [=========]
  └─ 문서화 & 발표 (D17-D18) [==========]
```

---

## 11. 코드 구조 및 재현성 (Reproducibility)

### 11.1 프로젝트 디렉토리 구조

```
/content/ondevice-ai-civil-complaint/
├── data/
│   ├── raw/                          # 원본 데이터
│   │   ├── aihub/
│   │   │   ├── 71852/               # AI Hub 공공 민원 데이터
│   │   │   └── 71844/               # AI Hub 민간 민원 데이터
│   │   └── seoul_api/               # 서울 열린데이터 광장
│   ├── processed/                    # 전처리 완료 데이터
│   │   ├── civil_complaint_train.jsonl
│   │   ├── civil_complaint_val.jsonl
│   │   └── civil_complaint_test.jsonl
│   └── calibration/                  # AWQ 캘리브레이션 데이터
│       ├── exaone_civil_calibration.json
│       ├── exaone_civil_calibration.txt
│       └── exaone_civil_calibration_stats.json
│
├── models/
│   ├── base/                         # 베이스 모델 (다운로드)
│   │   └── EXAONE-3.5-7.8B-Instruct/
│   ├── checkpoints/                  # 학습 체크포인트
│   │   ├── exaone-qlora-baseline/
│   │   │   ├── checkpoint-500/
│   │   │   ├── checkpoint-1000/
│   │   │   ├── final/
│   │   │   └── lora_adapter/        # LoRA 어댑터만 (150MB)
│   │   ├── exaone-qlora-rank8/
│   │   └── exaone-qlora-rank32/
│   ├── merged/                       # LoRA 병합 모델
│   │   └── exaone-qlora-baseline-merged/
│   └── quantized/                    # AWQ 양자화 모델
│       ├── exaone-qlora-awq-4bit/
│       └── exaone-qlora-awq-4bit-group128/
│
├── src/
│   ├── data_collection_preprocessing/  # 데이터 파이프라인 (기존)
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   ├── aihub_collector.py
│   │   ├── seoul_api_collector.py
│   │   ├── pii_masking.py
│   │   ├── data_preprocessor.py
│   │   └── calibration_dataset.py
│   ├── training/                     # 학습 스크립트 (신규 작성)
│   │   ├── __init__.py
│   │   ├── train_qlora.py           # QLoRA 학습 메인 스크립트
│   │   ├── trainer_config.py         # TrainingArguments 설정
│   │   └── data_collator.py          # Custom data collator
│   ├── quantization/                 # 양자화 스크립트 (신규 작성)
│   │   ├── __init__.py
│   │   ├── quantize_awq.py          # AWQ 양자화 메인 스크립트
│   │   └── merge_lora.py             # LoRA 병합 스크립트
│   └── evaluation/                   # 평가 스크립트 (신규 작성)
│       ├── __init__.py
│       ├── evaluate_model.py         # 종합 평가 스크립트
│       ├── metrics.py                # 평가 메트릭 정의
│       └── benchmark.py              # 추론 속도 벤치마크
│
├── notebooks/                        # Jupyter/Colab 노트북
│   ├── 01_setup_environment.ipynb   # 환경 설정 가이드
│   ├── 02_data_preparation.ipynb    # 데이터 준비 워크플로우
│   ├── 03_qlora_training.ipynb      # QLoRA 학습 노트북
│   ├── 04_awq_quantization.ipynb    # AWQ 양자화 노트북
│   └── 05_evaluation.ipynb          # 평가 노트북
│
├── logs/                             # 실험 로그
│   ├── tensorboard/
│   │   └── exaone-qlora-baseline/
│   └── wandb/
│
├── docs/                             # 문서
│   └── outputs/
│       └── M2_MVP/
│           ├── experiment_plan.md   # 본 문서
│           ├── evaluation_report.md # 평가 리포트 (작성 예정)
│           └── training_logs/       # 학습 로그 정리
│
├── .env                              # 환경 변수 (API 키 등)
├── requirements.txt                  # Python 의존성
└── README.md                         # 프로젝트 README
```

### 11.2 시드 고정 전략

```python
# reproducibility.py
import os
import random
import numpy as np
import torch
from transformers import set_seed

def set_random_seed(seed: int = 42):
    """
    모든 난수 생성기의 시드를 고정하여 재현성 확보

    Args:
        seed: 난수 시드 (기본값: 42)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # PyTorch backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hugging Face transformers
    set_seed(seed)

    # Environment variable (for some libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ Random seed set to {seed} for reproducibility")

# 모든 실험 스크립트 시작 시 호출
set_random_seed(42)
```

### 11.3 필수 라이브러리 버전 명세

```txt
# requirements.txt (Freeze된 버전)
torch==2.1.2+cu121
torchvision==0.16.2+cu121
torchaudio==2.1.2+cu121

transformers==4.40.0
datasets==2.18.0
accelerate==0.28.0
peft==0.10.0
trl==0.8.1

bitsandbytes==0.43.0
autoawq==0.2.0
optimum==1.17.0

evaluate==0.4.1
rouge-score==0.1.2
sacrebleu==2.4.0

wandb==0.16.4
tensorboard==2.16.2

einops==0.7.0
sentencepiece==0.2.0
protobuf==4.25.0
python-dotenv==1.0.1
tqdm==4.66.2

# Python 버전: 3.10.12 (Colab 기본)
# CUDA 버전: 12.1
# cuDNN 버전: 8.9.7
```

### 11.4 실험 재현을 위한 체크리스트

#### 환경 재현
- [ ] Google Colab Pro A100 런타임 선택
- [ ] Python 3.10.12 확인
- [ ] CUDA 12.1 확인
- [ ] `requirements.txt` 설치 (`pip install -r requirements.txt`)
- [ ] 프로젝트 저장소 클론

#### 데이터 재현
- [ ] AI Hub API 키 발급 및 `.env` 설정
- [ ] 데이터셋 71852, 71844 다운로드
- [ ] 데이터 전처리 파이프라인 실행 (`pipeline.py --mode full`)
- [ ] 캘리브레이션 데이터셋 생성 확인

#### 모델 재현
- [ ] EXAONE-3.5-7.8B-Instruct 모델 다운로드
- [ ] 시드 고정 (`set_random_seed(42)`)
- [ ] QLoRA 학습 실행 (`train_qlora.py`)
- [ ] 체크포인트 저장 확인

#### 평가 재현
- [ ] Test set 고정 (`test_seed=42`)
- [ ] 평가 스크립트 실행 (`evaluate_model.py`)
- [ ] 벤치마크 스크립트 실행 (`benchmark.py`)

---

## 부록 A: Colab 노트북 퀵스타트 가이드

### A.1 환경 설정 노트북 (01_setup_environment.ipynb)

```python
# Cell 1: GPU 확인
!nvidia-smi

# Cell 2: 라이브러리 설치
%%bash
pip install -q torch==2.1.2 transformers==4.40.0 datasets==2.18.0 \
    accelerate==0.28.0 peft==0.10.0 bitsandbytes==0.43.0 \
    autoawq==0.2.0 trl==0.8.1 wandb==0.16.4

# Cell 3: 프로젝트 클론
!git clone https://github.com/YOUR_USERNAME/ondevice-ai-civil-complaint.git
%cd ondevice-ai-civil-complaint

# Cell 4: 환경 변수 설정
import os
from google.colab import userdata

os.environ["AIHUB_API_KEY"] = userdata.get('AIHUB_API_KEY')
os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')

# Cell 5: 디렉토리 생성
!mkdir -p data/raw data/processed data/calibration models logs

print("✓ Setup complete!")
```

### A.2 학습 노트북 (03_qlora_training.ipynb)

```python
# Cell 1: 데이터 준비 확인
!ls -lh data/processed/

# Cell 2: 학습 스크립트 실행
%run src/training/train_qlora.py \
    --model_id LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
    --train_data data/processed/civil_complaint_train.jsonl \
    --val_data data/processed/civil_complaint_val.jsonl \
    --output_dir models/checkpoints/exaone-qlora-baseline \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --wandb_project exaone-civil-complaint

# Cell 3: 학습 모니터링 (WandB 링크 확인)
```

---

## 부록 B: 참고 문헌 및 리소스

### B.1 논문 및 문서
1. **QLoRA**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
2. **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
3. **EXAONE Model Card**: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct

### B.2 공식 리포지토리
- Hugging Face Transformers: https://github.com/huggingface/transformers
- PEFT (LoRA): https://github.com/huggingface/peft
- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- vLLM: https://github.com/vllm-project/vllm

### B.3 데이터셋
- AI Hub 공공 민원 상담 LLM 데이터 (71852): https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71852
- AI Hub 민간 민원 상담 LLM 데이터 (71844): https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71844

---

## 문서 개정 이력

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|----------|
| 1.0 | 2026-03-05 | Claude Code | 초안 작성 |

---

**문서 끝**
