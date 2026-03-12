# Models Directory

This directory contains information about the fine-tuned models and adapters developed for the On-Device AI Civil Complaint Analysis System.

All model weights are hosted on the [Hugging Face Model Hub](https://huggingface.co/umyunsang) due to file size limits.

| Model | Type | Size | Description |
|-------|------|------|-------------|
| [civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora) | LoRA Adapter | - | QLoRA 파인튜닝 어댑터 |
| [civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged) | Full Model (BF16) | 14.56 GB | LoRA 병합 풀 모델 |
| [civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq) | Quantized (4-bit) | 4.94 GB | AWQ 양자화 모델 (온디바이스 배포용) |

---

## 1. Fine-tuned LoRA Adapter (QLoRA)

- **Model Repository**: [umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora)
- **Base Model**: [LGAI-EXAONE/EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B)

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Fine-tuning Method | QLoRA (4-bit NF4) |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Best Eval Loss | 1.0179 |

### How to Load

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "LGAI-EXAONE/EXAONE-Deep-7.8B"
adapter_id = "umyunsang/civil-complaint-exaone-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter_id)
```

---

## 2. LoRA Merged Full Model (BF16)

LoRA 어댑터를 베이스 모델에 `merge_and_unload()`로 병합한 풀 모델입니다.

- **Model Repository**: [umyunsang/civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged)
- **Base Model**: [LGAI-EXAONE/EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B)
- **Model Size**: 14.56 GB (BF16)
- **Parameters**: 7.8B

### How to Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "umyunsang/civil-complaint-exaone-merged"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

---

## 3. AWQ Quantized Model (4-bit)

LoRA 병합 모델을 AWQ로 4-bit 양자화하여 온디바이스/엣지 배포에 최적화한 모델입니다.

- **Model Repository**: [umyunsang/civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq)
- **Base Model**: [umyunsang/civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged)
- **Quantized Size**: 4.94 GB (66.1% 압축, 2.95x 압축률)
- **GPU VRAM**: 5-7 GB (추론 시)

### AWQ Quantization Config

| Setting | Value |
|---------|-------|
| Weight Bits | 4-bit |
| Activation Bits | 16-bit (FP16) |
| Group Size | 128 |
| Zero-point | True |
| Version | GEMM |
| Calibration Data | 512 samples (민원 도메인) |

### How to Load (vLLM - Recommended)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="umyunsang/civil-complaint-exaone-awq",
    quantization="awq",
    trust_remote_code=True,
)

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=512)
outputs = llm.generate([prompt], sampling_params)
```

### How to Load (AutoAWQ)

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "umyunsang/civil-complaint-exaone-awq"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoAWQForCausalLM.from_quantized(
    model_id,
    fuse_layers=True,
    trust_remote_code=True,
    safetensors=True,
)
```

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Perplexity | 3.20 |
| BLEU Score | 17.32 |
| ROUGE-L Score | 18.28 |

### AWQ Inference Performance

| Metric | Value |
|--------|-------|
| Inference Latency | 9.29s |
| Throughput | 13.8 tokens/s |

---

## Model Pipeline

```
EXAONE-Deep-7.8B (Base)
  └─ QLoRA Fine-tuning ──→ civil-complaint-exaone-lora (Adapter)
       └─ merge_and_unload() ──→ civil-complaint-exaone-merged (BF16, 14.56GB)
            └─ AWQ Quantization ──→ civil-complaint-exaone-awq (4-bit, 4.94GB)
```
