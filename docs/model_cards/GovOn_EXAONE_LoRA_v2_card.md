---
language:
- ko
license: apache-2.0
library_name: peft
base_model: LGAI-EXAONE/EXAONE-Deep-7.8B
tags:
- lora
- qlora
- exaone
- civil-complaint
- govon
- korean
- government
datasets:
- custom
pipeline_tag: text-generation
---

# GovOn-EXAONE-LoRA-v2

지자체 민원 AI 어시스턴트를 위한 EXAONE-Deep-7.8B 기반 QLoRA 어댑터

## 모델 설명

**GovOn-EXAONE-LoRA-v2**는 LG AI Research의 [EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B) 모델을 베이스로, 한국 지방자치단체 민원 도메인에 특화된 QLoRA 파인튜닝 어댑터이다.

주민의 다양한 민원 질의(행정, 교통, 환경, 복지, 문화, 경제, 안전 등)에 대해 정확하고 신뢰성 있는 답변을 생성하는 것을 목표로 한다.

### v2 주요 개선사항

v1 대비 다음 사항을 개선했다:

- **EOS 토큰 학습 정상화**: `pad_token`을 `unk_token`으로 분리하여 EOS 학습 차단 문제 해결 (EOS 생성률 0% -> 20%)
- **데이터 균형화**: 카테고리별 30% 샘플링 제한으로 편향(행정 89.6%) 해소
- **PII 마스킹 강화**: 개인정보 마스킹 로직 v2 적용
- **학습 파이프라인 안정화**: SFTConfig + DataCollatorForCompletionOnlyLM 적용

## 학습 상세

### 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| 베이스 모델 | LGAI-EXAONE/EXAONE-Deep-7.8B |
| 파인튜닝 방식 | QLoRA (4-bit NF4) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 양자화 | 4-bit NF4, double quantization, bfloat16 compute |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Epochs | 3 |
| Batch size (per device) | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Max sequence length | 2048 |
| Max grad norm | 1.0 |
| Precision | bf16 |
| Gradient checkpointing | True |

### 학습 곡선 요약

| 지표 | 값 |
|------|-----|
| 초기 train loss | 3.3224 |
| 최종 train loss | 1.5320 |
| 최종 eval loss | 1.7872 |
| 최종 train token accuracy | 0.6444 |
| 최종 eval token accuracy | 0.6046 |
| Train-Eval gap | 0.2552 |
| Total steps | 1,902 |
| 학습 시간 | 약 167분 |

학습은 3 epoch 동안 안정적으로 수렴했으며, train-eval gap이 0.25 수준으로 과적합이 심하지 않다.

### v1 대비 개선

| 지표 | v1 | v2 | 변화 |
|------|-----|-----|------|
| eval_loss | 1.7909 | 1.7872 | -0.0037 (-0.21%) |
| eval token accuracy | 0.6044 | 0.6046 | +0.0002 |
| train_loss (avg) | 1.7535 | 1.7492 | -0.0043 |

## 학습 데이터

한국 지방자치단체 민원 데이터를 기반으로 구성했다.

| 분할 | 샘플 수 |
|------|---------|
| Train | 10,148 |
| Validation | 1,265 |
| Test | 1,265 |
| **합계** | **12,678** |

### 카테고리 분포 (8개)

행정, 교통, 환경, 복지, 문화, 경제, 안전, 기타

v1의 카테고리 편향(행정 89.6%)을 해소하기 위해 카테고리별 30% 샘플링 제한을 적용했다.

### 데이터 전처리

- 71,847건의 원본 데이터에서 카테고리 세분화 및 품질 필터링
- PII 마스킹 v2 적용 (전화번호, 주민등록번호, 이메일 등)
- Chat template 형식으로 변환 (system / user / assistant)
- DataCollatorForCompletionOnlyLM으로 assistant 응답 부분에만 loss 적용

## 평가 결과

### Sanity Check (5개 샘플)

| 지표 | v1 | v2 |
|------|-----|-----|
| EOS 생성률 | 0% (0/5) | 20% (1/5) |
| 평균 생성 길이 | 866자 | 838자 |

### 참고: v1 자동 평가 지표

| 지표 | v1 값 |
|------|-------|
| BLEU | 0.53 |
| ROUGE-L | 4.20 |
| length_ratio | 0.63 |

v2에 대한 본격적인 자동 평가(BLEU, ROUGE-L, BERTScore)는 추후 진행 예정이다.

## 사용 방법

### 필수 패키지

```bash
pip install transformers peft bitsandbytes accelerate torch
```

### 추론 코드

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 베이스 모델 로드 (4-bit 양자화)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-Deep-7.8B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "LGAI-EXAONE/EXAONE-Deep-7.8B",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.unk_token

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "umyunsang/GovOn-EXAONE-LoRA-v2")
model.eval()

# 민원 질의 생성
messages = [
    {"role": "system", "content": "당신은 지자체 민원 상담 AI 어시스턴트입니다. 주민의 질문에 정확하고 친절하게 답변해주세요."},
    {"role": "user", "content": "주민등록증 재발급 절차가 어떻게 되나요?"},
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

# <thought>...</thought> 태그 제거 (EXAONE-Deep CoT)
import re
response = re.sub(r"<thought>.*?</thought>", "", response, flags=re.DOTALL).strip()

print(response)
```

## 제한사항

1. **EOS 생성 불안정**: EOS 생성률이 20%로, 대부분의 응답이 `max_new_tokens`에 도달할 때까지 생성을 계속한다. `max_new_tokens`를 적절히 설정하고, 후처리로 응답을 정리할 필요가 있다.

2. **Thought 태그 포함**: EXAONE-Deep 모델의 특성상 `<thought>...</thought>` 태그가 응답에 포함될 수 있다. 사용자에게 보여주기 전에 반드시 제거해야 한다.

3. **응답 길이**: 응답이 참조 답변 대비 짧은 경향이 있다(v1 기준 length_ratio 0.63). 중요한 정보가 누락될 수 있으므로 응답 품질 검수가 필요하다.

4. **카테고리 범위**: 8개 카테고리(행정, 교통, 환경, 복지, 문화, 경제, 안전, 기타)에 대해 학습되었으며, 이 범위를 벗어나는 질의에 대해서는 답변 품질이 보장되지 않는다.

5. **법적/규정 정확성**: AI가 생성한 답변은 참고용이며, 법적 효력이 있는 공식 답변으로 사용할 수 없다. 실제 업무에서는 반드시 담당 공무원의 검토가 필요하다.

6. **최대 시퀀스 길이**: `max_seq_length=2048`로 학습되었으므로, 이를 초과하는 긴 입력은 잘릴 수 있다.

## 학습 인프라

| 항목 | 내용 |
|------|------|
| GPU | NVIDIA A100 40GB (Google Colab) |
| 학습 시간 | 약 167분 (2시간 47분) |
| 학습 프레임워크 | TRL 0.18.x + PEFT 0.18.1 + Transformers 4.49.0 |
| 양자화 라이브러리 | BitsAndBytes |
| 실험 추적 | Weights & Biases |
| W&B Run | [umyun3/GovOn-retrain-v2/uggxvc3s](https://wandb.ai/umyun3/GovOn-retrain-v2/runs/uggxvc3s) |

## 인용

```bibtex
@misc{govon-exaone-lora-v2,
  title={GovOn-EXAONE-LoRA-v2: QLoRA Fine-tuned EXAONE-Deep-7.8B for Korean Civil Complaint Assistance},
  author={GovOn Team},
  year={2026},
  url={https://huggingface.co/umyunsang/GovOn-EXAONE-LoRA-v2}
}
```
