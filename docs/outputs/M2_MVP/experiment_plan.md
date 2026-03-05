# EXAONE-Deep-7.8B 실험 계획서
## QLoRA 파인튜닝 및 AWQ 양자화

**문서 버전**: 2.0 (Updated for EXAONE-Deep-7.8B)
**작성일**: 2026-03-05
**프로젝트**: On-Device AI 민원 분석 및 처리 시스템
**실행 환경**: Google Colab L4 (24GB VRAM) / A100 (40GB VRAM)
**베이스 모델**: [LGAI-EXAONE/EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B)

---

## 목차

1. [실험 개요 (Experiment Overview)](#1-실험-개요-experiment-overview)
2. [모델 및 데이터셋 구성 (Model & Dataset Configuration)](#2-모델-및-데이터셋-구성-model--dataset-configuration)
3. [실험 환경 및 호환성 (Environment & Compatibility)](#3-실험-환경-및-호환성-environment--compatibility)
4. [데이터 준비 및 전처리 (Data Preparation)](#4-데이터-준비-및-전처리)
5. [QLoRA 파인튜닝 실험 설계 (QLoRA Fine-tuning Design)](#5-qlora-파인튜닝-실험-설계-qlora-fine-tuning-design)
6. [AWQ 양자화 실험 설계 (AWQ Quantization Design)](#6-awq-양자화-실험-설계-awq-quantization-design)
7. [평가 메트릭 및 벤치마크 (Evaluation Metrics)](#7-평가-메트릭-및-벤치마크-evaluation-metrics)
8. [향후 일정 및 목표 (Timeline)](#8-향후-일정-및-목표)

---

## 1. 실험 개요 (Experiment Overview)

### 1.1 연구 목적 및 가설

#### 연구 목적
본 실험은 한국어 특화 LLM인 EXAONE-Deep-7.8B 모델을 민원 도메인에 특화하여 파인튜닝하고, 온프레미스 환경 배포를 위한 최적의 양자화 기법을 검증하는 것을 목표로 합니다.

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

---

## 2. 모델 및 데이터셋 구성 (Model & Dataset Configuration)

### 2.1 베이스 모델: EXAONE-Deep-7.8B

#### 모델 선정 근거
| 평가 기준 | 상세 내용 |
|----------|----------|
| **한국어 성능** | 한국 표준 테스트 (CSAT Math 2025) 89.9% 달성 |
| **모델 크기** | 7.8B 파라미터 - Colab L4 (24GB VRAM) 환경에 최적 |
| **컨텍스트 길이** | 32,768 토큰 - 긴 민원 및 유사 사례 참조 가능 |
| **아키텍처** | Grouped Query Attention (GQA) - 추론 효율성 우수 |
| **라이선스** | EXAONE AI Model License 1.1-NC (연구/비상업적 사용 허용) |
| **공식 지원** | vLLM, AutoAWQ 공식 지원 |

#### QLoRA 학습 상세 설정
```python
QLORA_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "task_type": "CAUSAL_LM"
}
```

#### Training Hyperparameters (EXP-001 Baseline)
- **Batch Size**: 2
- **Gradient Accumulation**: 8 (Effective Batch Size: 16)
- **Learning Rate**: 2e-4
- **Epochs**: 1 (Initial Test) / 3 (Full Training)
- **Max Seq Length**: 2048
- **Optimizer**: `paged_adamw_8bit`
- **Scheduler**: `cosine`

### 2.2 데이터셋 구성

#### 2.2.1 AI Hub 데이터셋 (우선순위 기반 선정)

| 데이터셋 번호 | 명칭 | 예상 규모 | 사용 목적 | 우선순위 |
|--------------|------|----------|----------|---------|
| **71852** | **공공 민원 상담 LLM 데이터** | 150,000건+ | **주 학습 데이터** (Instruction Tuning) | 1 |
| **71844** | **민간 민원 상담 LLM 데이터** | 200,000건+ | 보조 학습 데이터 (도메인 확장) | 2 |

#### 2.2.2 데이터 분할 비율
```python
SPLIT_RATIOS = {
    "train": 0.80,      # 80% 학습용
    "validation": 0.10, # 10% 검증용
    "test": 0.10        # 10% 평가용
}
```

---

## 3. 실험 환경 및 호환성 (Environment & Compatibility)

### 3.1 필수 라이브러리 및 버전
- `transformers==5.3.0` (EXAONE 모델 지원 버전)
- `trl==0.12.0` (DataCollator 호환 버전)
- `peft>=0.14.0`
- `bitsandbytes>=0.45.0`

### 3.2 모델 호환성 패치 (Monkey-patching)
EXAONE 모델의 원활한 학습을 위해 아래와 같은 패치가 적용되었습니다.
1. `transformers.utils.generic.check_model_inputs` 삭제 대응 (수동 정의)
2. `get_input_embeddings` 및 `get_output_embeddings` 몽키 패치 적용

---

## 4. 데이터 준비 및 전처리

### 4.1 데이터 포맷 (EXAONE Chat Template)
```text
[|system|]
당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다.
[|user|]
{instruction}\n\n{input}
[|assistant|]
{output}
```

### 4.2 전처리 파이프라인
1. **PII 마스킹**: 이름, 주민번호, 전화번호 등 개인정보 자동 탐지 및 마스킹
2. **데이터 정제**: 중복 제거, 길이 필터링 (최소 20자 이상)
3. **포맷 변환**: EXAONE Chat Template 형식으로 변환

---

## 5. QLoRA 파인튜닝 실험 설계 (QLoRA Fine-tuning Design)

| 실험 ID | 변경 변수 | 설정값 | 목적 |
|---------|----------|--------|------|
| EXP-001 | Baseline | r=16, lr=2e-4 | 기준 성능 측정 |
| EXP-002 | LoRA Rank | r=8, r=32 | 경량화 및 성능 변화 검증 |
| EXP-003 | Learning Rate | lr=1e-4 | 수렴 안정성 검증 |

---

## 6. AWQ 양자화 실험 설계 (AWQ Quantization Design)

- **방법**: AWQ (Activation-aware Weight Quantization) 4-bit
- **설정**: group_size=128, zero_point=True
- **목표**: 모델 크기 < 8GB, 추론 속도 향상

---

## 7. 평가 메트릭 및 벤치마크 (Evaluation Metrics)

- **분류 정확도**: Accuracy, F1-Score
- **생성 품질**: BLEU-4, ROUGE-L
- **추론 성능**: p50/p95 Latency, VRAM Usage

---

## 8. 향후 일정 및 목표

### 8.1 Week 6 목표
- **EXP-001 Baseline QLoRA** 완료 (현재 진행 중)
- **EXP-002 Rank 변화 실험** (r=8, r=32)
- **EXP-003 Learning Rate 변화 실험** (lr=1e-4)

### 8.2 Week 7 목표
- **LoRA Merge & AWQ Quantization**
- **민원 분류 정확도 및 답변 생성 품질 평가**
