# Models Directory

This directory contains information about the fine-tuned models and adapters developed for the On-Device AI Civil Complaint Analysis System.

## Fine-tuned LoRA Adapter (QLoRA)

Due to file size limits, the trained model weights (LoRA adapters) are hosted on the Hugging Face Model Hub instead of this repository.

- **Model Repository**: [umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora)
- **Base Model**: [LGAI-EXAONE/EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B)

### Model Features & Architecture
- **Architecture**: EXAONE-Deep-7.8B (32 layers, GQA, 32k context)
- **Fine-tuning Method**: QLoRA (4-bit NF4 quantization)
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Rank (r)**: 16
- **Alpha**: 32
- **Best Eval Loss**: 1.0179 (AI Hub Civil Complaint Dataset)

### How to Load
You can easily load the fine-tuned adapter using the `peft` and `transformers` libraries.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
base_model_id = "LGAI-EXAONE/EXAONE-Deep-7.8B"
adapter_id = "umyunsang/civil-complaint-exaone-lora"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# 2. Load Base Model (4-bit or bfloat16 recommended)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 3. Load LoRA Adapter
model = PeftModel.from_pretrained(model, adapter_id)

print("Civil Complaint Analysis Model Loaded Successfully!")
```

## Upcoming Models
- [ ] **AWQ Quantized Model**: A 4-bit quantized version optimized for vLLM and mobile deployment (Work in Progress).
