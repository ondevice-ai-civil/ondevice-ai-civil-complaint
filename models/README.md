# Models Directory

This directory contains information about the fine-tuned models and adapters developed for the On-Device AI Civil Complaint Analysis System.

All model weights are hosted on the [Hugging Face Model Hub](https://huggingface.co/umyunsang) due to file size limits.

> **⚠ 폐기 안내 (2026-03-19)**: 아래 v1 모델(LoRA, Merged, AWQ)은 **잘못 학습된 LoRA 어댑터**를 기반으로 생성되었기 때문에 전량 폐기 대상입니다. 해당 모델을 사용하지 마세요. 재학습된 v2 모델로 교체 예정입니다.

| Model | Type | Size | 상태 |
|-------|------|------|------|
| ~~[civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora)~~ | LoRA Adapter | - | ❌ 폐기 (잘못된 학습) |
| ~~[civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged)~~ | Full Model (BF16) | 14.56 GB | ❌ 폐기 (잘못된 LoRA 기반 병합) |
| ~~[civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq)~~ | Quantized (4-bit) | 4.94 GB | ❌ 폐기 (잘못된 병합 모델 기반 양자화) |
| [GovOn-EXAONE-LoRA-v2](https://huggingface.co/umyunsang/GovOn-EXAONE-LoRA-v2) | LoRA Adapter v2 | - | ✅ 최신 (재학습 완료) |

---

## 폐기 모델 상세 (v1 — 사용 금지)

아래 3개 모델은 잘못 학습된 LoRA 어댑터(v1)로부터 파생되어 모두 무효입니다.

### ~~1. Fine-tuned LoRA Adapter v1 (폐기)~~

- **Model Repository**: ~~[umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora)~~
- **폐기 사유**: 학습 데이터 또는 학습 설정 오류로 인해 정상적인 추론 결과를 생성하지 못함

### ~~2. LoRA Merged Full Model v1 (폐기)~~

- **Model Repository**: ~~[umyunsang/civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged)~~
- **폐기 사유**: 폐기된 LoRA v1을 병합하여 생성 → 원본 오류 그대로 상속

### ~~3. AWQ Quantized Model v1 (폐기)~~

- **Model Repository**: ~~[umyunsang/civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq)~~
- **폐기 사유**: 폐기된 병합 모델을 양자화하여 생성 → 원본 오류 그대로 상속

---

## 현재 유효 모델

### GovOn-EXAONE-LoRA-v2 (최신)

- **Model Repository**: [umyunsang/GovOn-EXAONE-LoRA-v2](https://huggingface.co/umyunsang/GovOn-EXAONE-LoRA-v2)
- **Base Model**: [LGAI-EXAONE/EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B)
- **상태**: 재학습 완료, AWQ 양자화 모델은 v2 기반으로 재생성 예정

---

## Model Pipeline (계획)

```
EXAONE-Deep-7.8B (Base)
  └─ QLoRA Fine-tuning ──→ GovOn-EXAONE-LoRA-v2 (Adapter) ✅ 완료
       └─ merge_and_unload() ──→ Merged Model v2 (BF16) 🔜 예정
            └─ AWQ Quantization ──→ AWQ Model v2 (4-bit) 🔜 예정
```
