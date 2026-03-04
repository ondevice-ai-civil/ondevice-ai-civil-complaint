# EXAONE-Deep-7.8B AWQ 모델 분석

## 문서 정보
- **작성일**: 2026-03-05
- **작성 목적**: 민원 처리 시스템을 위한 EXAONE-Deep-7.8B AWQ 모델 분석
- **프로젝트**: On-Device AI 기반 민원 처리 시스템
- **대상 모델**: [LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ)

---

## 1. EXAONE-Deep-7.8B 모델 개요

### 1.1 기본 정보
- **개발사**: LG AI Research
- **모델 계열**: EXAONE Deep (Reasoning Enhanced Language Models)
- **모델 크기**: 7.8B 파라미터 (임베딩 제외 시 6.98B)
- **출시일**: 2025년 3월 18일
- **기반 모델**: [EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)
- **라이선스**: EXAONE AI Model License Agreement 1.1 - NC (Non-Commercial)

### 1.2 모델 특징
EXAONE Deep은 수학 및 코딩 추론 작업에서 뛰어난 성능을 보이는 추론 강화 언어 모델입니다. 동급 오픈 소스 모델뿐만 아니라 OpenAI의 o1-mini를 능가하는 성능을 보여줍니다.

### 1.3 모델 변형
LG AI Research는 다양한 사이즈와 양자화 버전을 제공합니다:
- **모델 크기**: 2.4B, 7.8B, 32B
- **정밀도**: BF16 (원본)
- **양자화 버전**: AWQ, GGUF (Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS)

---

## 2. 모델 아키텍처 분석

### 2.1 핵심 아키텍처 스펙

| 구성 요소 | 값 | 설명 |
|---------|-----|------|
| **총 파라미터** | 7.8B | 임베딩 포함 |
| **유효 파라미터** | 6.98B | 임베딩 제외 |
| **레이어 수** | 32 | Transformer 레이어 |
| **히든 차원** | 4,096 | Hidden size |
| **중간 차원** | 14,336 | FFN intermediate size |
| **어텐션 헤드** | 32 | Query heads (GQA) |
| **KV 헤드** | 8 | Key-Value heads |
| **헤드 차원** | 128 | Head dimension |
| **어휘 크기** | 102,400 | Vocabulary size |
| **컨텍스트 길이** | 32,768 tokens | 최대 시퀀스 길이 |

### 2.2 어텐션 메커니즘: GQA (Grouped-Query Attention)

EXAONE Deep은 **GQA (Grouped-Query Attention)** 를 채택했습니다:
- **Query Heads**: 32개
- **Key-Value Heads**: 8개
- **그룹화 비율**: 4:1 (32 Q-heads / 8 KV-heads)

#### GQA의 장점
1. **메모리 효율성**: KV 캐시 크기를 4배 감소시켜 긴 컨텍스트 처리에 유리
2. **추론 속도**: 캐시 크기 감소로 인한 메모리 대역폭 요구사항 감소
3. **품질 유지**: MQA(Multi-Query Attention) 대비 더 나은 품질 유지

### 2.3 위치 인코딩: RoPE Scaling

```json
{
  "rope_type": "llama3",
  "rope_theta": 1000000.0,
  "factor": 8.0,
  "original_max_position_embeddings": 8192,
  "low_freq_factor": 1.0,
  "high_freq_factor": 4.0
}
```

- **기본 컨텍스트**: 8,192 토큰
- **확장 컨텍스트**: 32,768 토큰 (4배 확장)
- **RoPE 타입**: LLaMA 3 방식
- **스케일링 팩터**: 8.0

### 2.4 활성화 함수 및 정규화

- **활성화 함수**: SiLU (Swish Linear Unit)
- **정규화**: Layer Normalization (epsilon=1e-05)
- **드롭아웃**: 0.0 (attention, embed)
- **가중치 타이핑**: False (임베딩과 출력 레이어 분리)

### 2.5 토큰 구성

- **BOS Token ID**: 1
- **EOS Token ID**: 361
- **PAD Token ID**: 0

---

## 3. AWQ 양자화 특징 및 장점

### 3.1 AWQ (Activation-aware Weight Quantization) 개요

AWQ는 MIT Han Lab에서 개발한 4비트 가중치 전용 양자화 기법으로, MLSys 2024 Best Paper Award를 수상했습니다.

#### 핵심 원리
1. **Activation-aware**: 활성화 분포를 기반으로 중요한 가중치 채널 식별
2. **선택적 보호**: 상위 1%의 중요 가중치만 보호해도 양자화 오류 크게 감소
3. **등가 변환**: 하드웨어 비효율적인 혼합 정밀도 대신, 중요 채널을 스케일 업하는 수학적 등가 변환 사용
4. **하드웨어 친화적**: 전체 가중치를 동일한 비트 폭으로 양자화하여 하드웨어 가속 최적화

### 3.2 AWQ의 장점

#### 메모리 효율성
- **압축률**: FP16 대비 약 4배 메모리 절감
- **원본 크기**: ~15.6GB (7.8B × 2 bytes)
- **AWQ 크기**: ~4.5GB (추정, 4비트 + 오버헤드)

#### 추론 성능
- **속도 향상**: Hugging Face FP16 구현 대비 3배 이상 속도 향상 (TinyChat 사용 시)
- **Marlin 커널**: AWQ에 최적화된 커널 사용 시 10.9배 속도 향상
- **처리량**: Marlin-AWQ 조합으로 741 tok/s 달성 (벤치마크 기준)

#### 품질 보존
- **정확도 손실**: 1% 미만의 정확도 손실
- **추론 품질**: Q4_K_M 같은 다른 4비트 양자화 방식 대비 우수한 품질

### 3.3 EXAONE-Deep-7.8B AWQ 양자화 설정

```json
{
  "quantization_config": {
    "bits": 4,
    "group_size": 128,
    "modules_to_not_convert": ["lm_head"],
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": true
  }
}
```

#### 주요 파라미터 설명

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **bits** | 4 | 4비트 양자화 |
| **group_size** | 128 | 그룹 단위 양자화 (128개 가중치마다) |
| **quant_method** | awq | AWQ 양자화 방식 |
| **version** | gemm | GEMM (General Matrix Multiply) 최적화 커널 |
| **zero_point** | true | Zero-point 양자화 사용 |
| **modules_to_not_convert** | lm_head | 언어 모델 헤드는 전체 정밀도 유지 |

#### 양자화 사양
- **표기**: W4A16g128
  - **W4**: 4비트 가중치 (Weights)
  - **A16**: 16비트 활성화 (Activations)
  - **g128**: 그룹 크기 128

### 3.4 다른 양자화 방식과의 비교

| 양자화 방식 | 비트 폭 | 메모리 | 속도 | 품질 | 하드웨어 지원 |
|-----------|--------|-------|------|------|------------|
| **AWQ** | 4-bit | 매우 좋음 | 매우 빠름 | 우수 | 광범위 (CUDA, ROCm 등) |
| **GPTQ** | 4-bit | 매우 좋음 | 빠름 | 좋음 | 광범위 |
| **GGUF Q4_K_M** | 4-bit | 매우 좋음 | 보통 | 보통 | CPU/GPU |
| **bitsandbytes** | 4/8-bit | 좋음 | 보통 | 좋음 | NVIDIA 전용 |

---

## 4. 사전 양자화된 AWQ 버전 분석

### 4.1 공식 AWQ 모델
- **모델 경로**: `LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ`
- **제공 형태**: 사전 양자화된 즉시 사용 가능한 모델
- **변환 품질**: LG AI Research에서 공식적으로 변환 및 검증

### 4.2 모델 파일 구성
- **텐서 타입**: F16 (활성화), I32 (양자화된 가중치)
- **설정 파일**: config.json (양자화 설정 포함)
- **토크나이저**: 원본 모델과 동일 (vocab_size=102,400)

### 4.3 로딩 및 사용 요구사항

#### 필수 라이브러리
```bash
pip install transformers>=4.43.1
pip install autoawq>=0.2.8
```

#### 기본 로딩 코드
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ"

# 모델 로딩 (AWQ 설정 자동 적용)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 활성화는 BF16
    trust_remote_code=True,
    device_map="auto"  # 자동 GPU 배치
)

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 4.4 추론 최적화 가이드

#### 1. 추론 프롬프트 구조
EXAONE Deep 모델은 추론 성능을 위해 특별한 프롬프트 구조가 필요합니다:

```python
# 올바른 방법: apply_chat_template 사용
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # <thought>\n 자동 추가
    return_tensors="pt"
)
```

- **필수**: `add_generation_prompt=True` 설정
- **효과**: `<thought>\n` 태그로 추론 단계 시작
- **성능**: 이 프롬프트 없이는 출력 품질이 크게 저하됨

#### 2. 권장 생성 파라미터
```python
output = model.generate(
    input_ids.to("cuda"),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=32768,  # 긴 추론 과정 허용
    do_sample=True,
    temperature=0.6,       # 권장 온도
    top_p=0.95            # 권장 top-p
)
```

#### 3. 프롬프트 작성 가이드
```python
# 수학 문제 예시
prompt = r"""문제: x, y, z가 양의 실수일 때...
단계별로 추론하고, 최종 답을 \boxed{} 안에 작성하세요."""

# 민원 처리 예시
prompt = """민원 내용: [민원 텍스트]
민원 카테고리를 분류하고 단계별로 근거를 제시한 후,
최종 분류 결과를 제공하세요."""
```

#### 4. 멀티턴 대화 처리
- **자동 처리**: 토크나이저가 이전 추론 단계(`<thought>...\n</thought>`) 자동 제거
- **이유**: 추론 단계가 많은 토큰을 생성하므로 컨텍스트 관리 필요
- **장점**: 사용자가 수동으로 처리할 필요 없음

#### 5. 시스템 프롬프트 회피
- **비권장**: 시스템 프롬프트 사용
- **권장**: 지시사항을 사용자 프롬프트에 포함

---

## 5. vLLM 호환성 및 서빙 최적화

### 5.1 지원 추론 프레임워크

EXAONE-Deep-7.8B AWQ는 다음 프레임워크를 공식 지원합니다:

| 프레임워크 | 지원 여부 | 특징 | 추천 용도 |
|-----------|----------|------|---------|
| **vLLM** | ✅ 완벽 지원 | 높은 처리량, 배치 처리 최적화 | 프로덕션 서빙 |
| **TensorRT-LLM** | ✅ 지원 | NVIDIA GPU 최적화 | NVIDIA 환경 |
| **SGLang** | ✅ 지원 | 구조화된 생성 | 제약 조건이 있는 생성 |
| **llama.cpp** | ✅ 지원 | CPU/Metal 지원 | CPU 추론, Apple Silicon |
| **Ollama** | ✅ 지원 | 간편한 로컬 배포 | 개발 및 테스트 |
| **LM-Studio** | ✅ 지원 | GUI 기반 | 비개발자 사용 |

### 5.2 vLLM 통합

#### vLLM 설치
```bash
pip install vllm>=0.3.0
pip install autoawq>=0.2.8
```

#### vLLM 서버 실행
```bash
# 기본 실행
vllm serve LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9

# 텐서 병렬화 (멀티 GPU)
vllm serve LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 8192
```

#### 주의사항: AWQ 변환 오류
- **초기 문제**: EXAONE-Deep-32B-AWQ에서 vLLM 텐서 병렬화 시 KeyError 발생
- **원인**: LG AI Research의 AWQ 변환 과정에 오류
- **해결**: LG AI Research가 변환 오류 수정 후 재배포
- **현재 상태**: 7.8B 및 32B AWQ 모델 모두 vLLM에서 정상 동작

### 5.3 vLLM 성능 최적화

#### Marlin 커널 활용
vLLM은 AWQ 모델을 위한 고성능 Marlin 커널을 지원합니다:

- **Marlin-AWQ 성능**:
  - GPTQ 대비 2.6배 속도 향상
  - AWQ 대비 10.9배 속도 향상 (Marlin 커널 사용 시)
  - 처리량: 741 tok/s (벤치마크 기준)

- **품질-속도 균형**: Marlin-AWQ가 최적의 sweet spot
  - AWQ의 우수한 품질 보존
  - Marlin 커널의 최고 처리량

#### 배치 처리 최적화
```python
from vllm import LLM, SamplingParams

# vLLM 모델 로딩
llm = LLM(
    model="LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.9
)

# 배치 추론
prompts = [prompt1, prompt2, prompt3, ...]
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=4096
)

outputs = llm.generate(prompts, sampling_params)
```

### 5.4 프로덕션 배포 권장사항

#### 1. OpenAI 호환 API 서버
```bash
vllm serve LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ \
    --dtype bfloat16 \
    --api-key your-api-key \
    --host 0.0.0.0 \
    --port 8000
```

#### 2. 클라이언트 사용
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ",
    messages=[
        {"role": "user", "content": "민원을 분류해주세요: ..."}
    ],
    temperature=0.6,
    max_tokens=2048
)
```

---

## 6. 한국어 성능 및 민원 처리 적합성

### 6.1 한국어 지원 능력

#### 공식 한국어 벤치마크
EXAONE Deep은 한국어 표준화 시험에서 우수한 성능을 입증했습니다:

| 벤치마크 | 점수 | 설명 |
|---------|------|------|
| **CSAT Math 2025** | **89.9%** (pass@1) | 2025년 대학수학능력시험 수학 |
| - | 7.8B 모델 중 최고 성능 | 동급 모델 대비 탁월 |

#### 한국어 토큰 효율성
- **어휘 크기**: 102,400 토큰
- **한국어 최적화**: EXAONE 시리즈는 한국어 처리를 위해 설계됨
- **토큰화 효율**: 한국어 텍스트를 효율적으로 인코딩

### 6.2 수학 및 추론 성능

EXAONE Deep의 강력한 추론 능력은 민원 처리의 복잡한 판단에 유용합니다:

| 벤치마크 | EXAONE-Deep-7.8B | 비교 대상 | 비고 |
|---------|-----------------|---------|------|
| **MATH-500** | 94.8% | OpenAI o1-mini: 90.0% | 상회 |
| **AIME 2024** | 70.0% / 83.3% | DeepSeek-R1-Distill | 경쟁력 있음 |
| **AIME 2025** | 59.6% / 76.7% | - | 고난도 추론 |
| **GPQA Diamond** | 62.6% | - | 도메인 전문 질문 |
| **Live Code Bench** | 55.2% | - | 코딩 작업 |

### 6.3 민원 처리 적합성 분석

#### 강점
1. **복잡한 추론 능력**
   - 수학 문제 해결에서 입증된 단계별 추론
   - 민원의 다층적 맥락 이해 및 분석에 적용 가능
   - `<thought>` 태그로 추론 과정 명시적 추적

2. **한국어 이해력**
   - 한국어 표준화 시험에서 검증된 언어 이해 능력
   - 민원 텍스트의 미묘한 뉘앙스 파악
   - 한국 문화적 맥락 이해

3. **긴 컨텍스트 처리**
   - 32,768 토큰 컨텍스트 지원
   - 장문의 민원 문서 전체 분석 가능
   - 관련 규정 및 참고 자료 함께 제공 가능

4. **추론 투명성**
   - `<thought>` 블록으로 추론 과정 가시화
   - 민원 처리 근거 설명 가능
   - 결정의 투명성 및 신뢰성 향상

#### 민원 처리 작업에 대한 활용 시나리오

##### 1. 민원 분류
```python
prompt = """다음 민원을 분석하고 적절한 카테고리로 분류하세요.
단계별로 근거를 제시하고, 최종 분류 결과를 제공하세요.

민원 내용:
[민원 텍스트]

가능한 카테고리:
- 환경오염 관련
- 교통/주차 불편
- 시설물 관리
- 민원 서비스
- 기타
"""
```

##### 2. 민원 우선순위 판단
```python
prompt = """다음 민원들의 긴급도를 평가하고 우선순위를 매기세요.
각 민원의 긴급도 판단 근거를 단계별로 설명하세요.

민원 1: [내용]
민원 2: [내용]
민원 3: [내용]
"""
```

##### 3. 답변 초안 생성
```python
prompt = """다음 민원에 대한 답변 초안을 작성하세요.
관련 규정과 절차를 고려하여 단계별로 답변을 구성하세요.

민원 내용: [내용]
관련 규정: [규정]
"""
```

#### 고려사항
1. **도메인 특화 필요성**
   - 일반 모델이므로 민원 처리 도메인에 특화된 파인튜닝 권장
   - Few-shot learning으로 민원 처리 스타일 학습 가능

2. **법률/규정 지식**
   - RAG (Retrieval-Augmented Generation)로 규정 데이터베이스 연동 필요
   - 긴 컨텍스트 활용하여 관련 규정 직접 제공

3. **품질 검증**
   - 실제 민원 데이터로 철저한 평가 필요
   - 인간 검토자와의 협업 체계 구축

---

## 7. 하드웨어 요구사항

### 7.1 GPU 메모리 요구사항

#### AWQ 4-bit (W4A16g128)
```
기본 모델 메모리 = 파라미터 × 비트 / 8

가중치 메모리 (4-bit):
- 6.98B × 4 bits / 8 = ~3.49GB
- 오버헤드 (스케일링, zero-point 등) ~20% = ~0.7GB
- 총 가중치: ~4.2GB

활성화 메모리 (BF16, 배치=1, 시퀀스=2048):
- 레이어당 활성화: 2 × 배치 × 시퀀스 × 히든
- 32 layers × 2 × 1 × 2048 × 4096 × 2 bytes
- ~1GB (추정)

KV 캐시 (BF16, 배치=1, 시퀀스=2048):
- 2 × 레이어 × 배치 × KV헤드 × 시퀀스 × 헤드차원 × 2 bytes
- 2 × 32 × 1 × 8 × 2048 × 128 × 2
- ~268MB

총 메모리: ~5.5-6GB (배치=1, 시퀀스=2048)
```

#### GPU별 권장 설정

| GPU 모델 | VRAM | 배치 크기 | 최대 시퀀스 | vLLM 추천 |
|---------|------|----------|-----------|-----------|
| **RTX 4060 Ti 16GB** | 16GB | 1-2 | 8,192 | ✅ 적합 |
| **RTX 4070** | 12GB | 1 | 4,096 | ✅ 적합 |
| **RTX 4080** | 16GB | 2-4 | 8,192 | ✅✅ 권장 |
| **RTX 4090** | 24GB | 4-8 | 16,384 | ✅✅ 이상적 |
| **A100 40GB** | 40GB | 8-16 | 32,768 | ✅✅ 프로덕션 |
| **A100 80GB** | 80GB | 16-32 | 32,768 | ✅✅ 대규모 |

### 7.2 시스템 요구사항

#### 최소 요구사항
- **GPU**: 8GB VRAM (제한적 사용)
- **RAM**: 16GB 시스템 메모리
- **저장공간**: 10GB (모델 + 라이브러리)
- **CUDA**: 11.8 이상

#### 권장 요구사항 (프로덕션)
- **GPU**: 16GB+ VRAM (RTX 4060 Ti 16GB, RTX 4080 이상)
- **RAM**: 32GB 시스템 메모리
- **저장공간**: 20GB SSD
- **CUDA**: 12.1 이상
- **vLLM**: 0.3.0 이상

#### 멀티 GPU 설정
```bash
# 텐서 병렬화 (2개 GPU)
vllm serve LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 16384

# 장점:
# - 더 큰 배치 크기 처리
# - 더 긴 시퀀스 지원
# - 높은 처리량
```

### 7.3 CPU 추론 (llama.cpp)

AWQ는 주로 GPU 최적화되어 있지만, GGUF 변환 후 CPU 추론도 가능합니다:

```bash
# GGUF 버전 사용
# LGAI-EXAONE/EXAONE-Deep-7.8B-GGUF

# llama.cpp 실행
./main -m EXAONE-Deep-7.8B-Q4_K_M.gguf \
       -n 512 \
       -t 8 \
       --temp 0.6 \
       --top-p 0.95
```

- **CPU**: 16코어 이상 권장
- **RAM**: 8-16GB (양자화 레벨에 따라)
- **속도**: GPU 대비 10-50배 느림 (CPU 성능에 따라)

### 7.4 Apple Silicon (M1/M2/M3)

```bash
# Metal 가속 사용 (llama.cpp)
./main -m EXAONE-Deep-7.8B-Q4_K_M.gguf \
       -n 512 \
       -t 8 \
       --gpu-layers 32 \
       --temp 0.6
```

| Mac 모델 | 메모리 | 추론 속도 (예상) | 적합성 |
|---------|-------|---------------|-------|
| **M1 Pro 16GB** | 16GB | 15-25 tok/s | ✅ 개발 용도 |
| **M2 Max 32GB** | 32GB | 25-40 tok/s | ✅ 프로덕션 가능 |
| **M3 Max 64GB** | 64GB | 40-60 tok/s | ✅✅ 이상적 |

---

## 8. 성능 벤치마크 요약

### 8.1 추론 성능 (공식 벤치마크)

| 작업 유형 | 벤치마크 | 점수 | 순위 |
|---------|---------|------|------|
| **수학 추론** | MATH-500 | 94.8% | 7.8B급 1위, o1-mini 상회 |
| **고급 수학** | AIME 2024 | 70.0% (pass@1) | 경쟁력 있음 |
| **고급 수학** | AIME 2025 | 59.6% (pass@1) | 최신 벤치마크 |
| **한국어 수학** | CSAT Math 2025 | 89.9% | 7.8B급 최고 |
| **도메인 추론** | GPQA Diamond | 62.6% | 전문 영역 강점 |
| **코딩** | Live Code Bench | 55.2% | 실용적 수준 |

### 8.2 양자화 효율성

| 메트릭 | FP16 원본 | AWQ 4-bit | 개선율 |
|-------|----------|-----------|-------|
| **모델 크기** | ~15.6GB | ~4.5GB | 71% 감소 |
| **메모리 사용** | ~20GB | ~6GB | 70% 감소 |
| **추론 속도** | 1x (기준) | 3-10x | Marlin 커널 사용 시 |
| **품질 손실** | 0% | <1% | 거의 무손실 |

### 8.3 vLLM 처리량 (예상)

기준: RTX 4090, 배치=8, 시퀀스=2048

| 구성 | 처리량 (tok/s) | 지연시간 (ms/tok) |
|-----|---------------|-----------------|
| **Transformers (FP16)** | ~80 tok/s | ~12ms |
| **vLLM + AWQ** | ~240 tok/s | ~4ms |
| **vLLM + Marlin-AWQ** | ~700+ tok/s | ~1.4ms |

*실제 성능은 하드웨어, 시퀀스 길이, 배치 크기에 따라 달라질 수 있습니다.*

---

## 9. 프로젝트 적용 권장사항

### 9.1 즉시 시작 가능한 설정

```python
# requirements.txt
transformers>=4.43.1
autoawq>=0.2.8
torch>=2.0.0
vllm>=0.3.0

# 기본 민원 분류 예제
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def classify_complaint(complaint_text, categories):
    prompt = f"""다음 민원을 분석하고 적절한 카테고리로 분류하세요.
단계별로 근거를 제시하고, 최종 분류 결과를 제공하세요.

민원 내용:
{complaint_text}

가능한 카테고리:
{', '.join(categories)}

분류 결과를 다음 형식으로 제공하세요:
카테고리: [선택된 카테고리]
신뢰도: [0-100]
근거: [분류 이유]
"""

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    output = model.generate(
        input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.95
    )

    return tokenizer.decode(output[0], skip_special_tokens=False)
```

### 9.2 단계별 프로젝트 통합 계획

#### Phase 1: 프로토타입 (1-2주)
1. **환경 설정**
   - GPU 서버 준비 (최소 16GB VRAM)
   - 필수 라이브러리 설치
   - EXAONE-Deep-7.8B-AWQ 다운로드

2. **기본 평가**
   - 샘플 민원 데이터로 분류 테스트
   - 추론 속도 및 품질 평가
   - 프롬프트 엔지니어링 실험

3. **베이스라인 구축**
   - 기본 민원 분류 파이프라인 구현
   - 성능 메트릭 수립

#### Phase 2: MVP 개발 (2-4주)
1. **vLLM 통합**
   - vLLM 서버 설정
   - OpenAI 호환 API 구축
   - 배치 처리 최적화

2. **프롬프트 최적화**
   - 민원 유형별 프롬프트 템플릿 개발
   - Few-shot 예제 선정
   - 출력 파싱 로직 구현

3. **평가 시스템**
   - 실제 민원 데이터로 평가
   - 인간 평가자와 결과 비교
   - 정확도, 일관성, 속도 측정

#### Phase 3: 파인튜닝 (4-6주)
1. **데이터 준비**
   - 민원 분류 학습 데이터 수집
   - 프롬프트-응답 쌍 구축
   - 데이터 품질 검증

2. **파인튜닝**
   - LoRA/QLoRA를 사용한 효율적 파인튜닝
   - 민원 도메인 특화 학습
   - 검증 세트로 성능 모니터링

3. **평가 및 반복**
   - 파인튜닝된 모델 vs 베이스 모델 비교
   - 오류 분석 및 개선
   - A/B 테스트

#### Phase 4: 프로덕션 배포 (2-3주)
1. **인프라 구축**
   - 고가용성 vLLM 서버 구성
   - 로드 밸런싱 및 오토 스케일링
   - 모니터링 및 로깅 시스템

2. **통합 테스트**
   - 엔드투엔드 테스트
   - 부하 테스트
   - 장애 복구 테스트

3. **점진적 롤아웃**
   - 파일럿 사용자 그룹
   - 피드백 수집 및 개선
   - 전체 배포

### 9.3 비용 효율성 분석

#### 클라우드 배포 (AWS 예시)
| 인스턴스 | GPU | VRAM | 시간당 비용 | 월 비용 (24/7) |
|---------|-----|------|-----------|-------------|
| **g5.xlarge** | A10G | 24GB | $1.006 | ~$730 |
| **g5.2xlarge** | A10G | 24GB | $1.212 | ~$880 |
| **p3.2xlarge** | V100 | 16GB | $3.06 | ~$2,220 |

#### 온프레미스 (초기 투자)
| 구성 | 비용 (예상) | 월 전력비 | TCO (3년) |
|-----|-----------|---------|----------|
| **워크스테이션 (RTX 4090)** | ~$3,000-4,000 | ~$30 | ~$5,000 |
| **서버 (A100 40GB)** | ~$15,000-20,000 | ~$100 | ~$23,000 |

**AWQ 양자화 비용 절감**:
- FP16 대비 GPU 메모리 요구사항 70% 감소
- 더 작은/저렴한 GPU 사용 가능
- 동일 GPU에서 더 큰 배치 처리 (처리량 증가)

### 9.4 대안 모델 비교

민원 처리 시스템을 위한 다른 옵션과의 비교:

| 모델 | 크기 | 한국어 | 추론 | 비용 | 추천도 |
|-----|------|-------|------|------|-------|
| **EXAONE-Deep-7.8B-AWQ** | 7.8B | ✅✅ 우수 | ✅✅ 탁월 | 💰 낮음 | ⭐⭐⭐⭐⭐ |
| **GPT-4** | - | ✅✅ 우수 | ✅✅ 최고 | 💰💰💰 매우 높음 | ⭐⭐⭐ (비용) |
| **HyperCLOVA X** | - | ✅✅ 최고 | ✅ 좋음 | 💰💰 높음 | ⭐⭐⭐⭐ |
| **Llama-3-8B** | 8B | ✅ 보통 | ✅ 좋음 | 💰 낮음 | ⭐⭐⭐ |
| **EEVE-Korean-10.8B** | 10.8B | ✅✅ 우수 | ✅ 보통 | 💰 낮음 | ⭐⭐⭐⭐ |

**EXAONE-Deep-7.8B-AWQ의 차별점**:
1. 한국어 + 추론 능력의 최적 조합
2. 온프레미스 배포 가능 (데이터 프라이버시)
3. 추론 과정 가시화 (`<thought>` 태그)
4. 비용 효율적 (오픈소스 + 양자화)

---

## 10. 결론 및 다음 단계

### 10.1 핵심 요약

**EXAONE-Deep-7.8B AWQ 모델의 강점**:
1. ✅ **한국어 이해력**: CSAT Math 2025에서 89.9% (7.8B급 최고)
2. ✅ **추론 능력**: MATH-500에서 94.8% (OpenAI o1-mini 상회)
3. ✅ **효율성**: AWQ 4-bit 양자화로 메모리 70% 절감
4. ✅ **배포 용이성**: vLLM, TensorRT-LLM 등 다양한 프레임워크 지원
5. ✅ **투명성**: `<thought>` 태그로 추론 과정 추적 가능
6. ✅ **비용 효율**: 16GB GPU에서 실행 가능, 온프레미스 배포 가능

**민원 처리 시스템 적합성**:
- 복잡한 민원 분석 및 분류에 필요한 추론 능력 보유
- 한국어 텍스트 이해 및 생성 능력 검증됨
- 긴 민원 문서 처리 가능 (32K 컨텍스트)
- 처리 근거 제시 가능 (투명성)

### 10.2 즉시 실행 가능한 액션 아이템

#### 1주차: 환경 설정 및 초기 테스트
```bash
# 1. 의존성 설치
pip install transformers>=4.43.1 autoawq>=0.2.8 vllm>=0.3.0

# 2. 모델 다운로드 및 테스트
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    'LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ',
    torch_dtype='auto',
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ')
print('모델 로딩 성공!')
"

# 3. vLLM 서버 실행
vllm serve LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ \
    --dtype bfloat16 \
    --max-model-len 8192
```

#### 2-3주차: 민원 분류 프로토타입
1. 샘플 민원 데이터 수집 (각 카테고리별 10-20건)
2. 프롬프트 템플릿 작성 및 테스트
3. 분류 정확도 평가 (인간 레이블과 비교)
4. Few-shot 예제 최적화

#### 4-6주차: 시스템 통합 및 최적화
1. vLLM API와 백엔드 시스템 통합
2. 배치 처리 파이프라인 구현
3. 성능 모니터링 대시보드 구축
4. 사용자 피드백 수집 메커니즘

### 10.3 추가 고려사항

#### 데이터 프라이버시
- **온프레미스 배포**: 민감한 민원 데이터를 외부에 전송하지 않음
- **모델 격리**: 전용 서버에서 실행, 네트워크 격리
- **로깅 관리**: 개인정보 마스킹, 로그 보존 기간 설정

#### 모델 업데이트 전략
- **새 버전 모니터링**: LG AI Research의 EXAONE 업데이트 추적
- **A/B 테스트**: 새 모델과 기존 모델 성능 비교
- **점진적 마이그레이션**: 카나리 배포 방식 사용

#### 백업 및 폴백
- **API 폴백**: 모델 오류 시 규칙 기반 시스템으로 폴백
- **인간 검토**: 낮은 신뢰도 예측은 인간 검토자에게 전달
- **다중 모델**: 중요한 케이스는 여러 모델로 교차 검증

### 10.4 참고 자료

#### 공식 문서
- [EXAONE Deep 논문](https://arxiv.org/abs/2503.12524)
- [Hugging Face 모델 카드](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ)
- [GitHub 리포지토리](https://github.com/LG-AI-EXAONE/EXAONE-Deep)
- [LG AI Research 블로그](https://www.lgresearch.ai/news/view?seq=543)

#### AWQ 관련 자료
- [AWQ 논문](https://arxiv.org/abs/2306.00978) - [MLSys 2024 Best Paper]
- [AWQ GitHub](https://github.com/mit-han-lab/llm-awq)
- [AutoAWQ 라이브러리](https://github.com/casper-hansen/AutoAWQ)

#### vLLM 관련 자료
- [vLLM 공식 문서](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM 양자화 가이드](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)

#### 커뮤니티 리소스
- [Hugging Face 디스커션](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ/discussions)
- [EXAONE vLLM 이슈](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-32B-AWQ/discussions/1)

---

## 부록: 기술 용어 정리

### A. 양자화 관련 용어

- **AWQ (Activation-aware Weight Quantization)**: 활성화 분포를 고려한 가중치 전용 양자화 기법
- **W4A16**: 4비트 가중치, 16비트 활성화
- **Group Size**: 양자화 그룹 크기 (예: 128개 가중치마다 스케일 팩터 적용)
- **Zero-point**: 양자화 시 0을 표현하기 위한 오프셋 값

### B. 아키텍처 용어

- **GQA (Grouped-Query Attention)**: 쿼리 헤드를 그룹화하여 KV 캐시 크기 감소
- **RoPE (Rotary Position Embedding)**: 회전 위치 임베딩, 긴 컨텍스트 처리에 유리
- **SiLU (Sigmoid Linear Unit)**: Swish 활성화 함수의 다른 이름
- **BF16 (BFloat16)**: 16비트 부동소수점 형식, FP32와 범위 동일, 정밀도는 낮음

### C. 추론 프레임워크 용어

- **vLLM**: 고처리량 LLM 추론 엔진
- **TensorRT-LLM**: NVIDIA의 LLM 최적화 라이브러리
- **Marlin 커널**: AWQ/GPTQ 모델을 위한 고성능 GEMM 커널
- **Tensor Parallelism**: 모델을 여러 GPU에 분산하여 처리

### D. 성능 메트릭 용어

- **tok/s (tokens per second)**: 초당 생성 토큰 수
- **pass@1**: 첫 시도에서 정답을 맞힐 확률
- **cons@64**: 64번의 시도 중 일관된 정답 비율
- **VRAM (Video RAM)**: GPU 메모리

---

## 문서 변경 이력

| 버전 | 날짜 | 변경 내용 | 작성자 |
|-----|------|---------|-------|
| 1.0 | 2026-03-05 | 초기 작성 | Claude |

---

**Sources:**
- [LGAI-EXAONE/EXAONE-Deep-7.8B · Hugging Face](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B)
- [LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ · Hugging Face](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [GitHub - mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
- [EXAONE-Deep-32B-AWQ vLLM Discussion](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-32B-AWQ/discussions/1)
- [The Complete Guide to LLM Quantization with vLLM](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)
