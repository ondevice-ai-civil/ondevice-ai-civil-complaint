# Models Directory

GovOn에서 사용하는 파인튜닝 모델 및 LoRA 어댑터 목록.
모든 모델 가중치는 파일 크기 제한으로 [HuggingFace Hub](https://huggingface.co)에 호스팅됩니다.

---

## 현재 아키텍처 (EXAONE 4.0-32B Multi-LoRA)

```
EXAONE 4.0-32B-AWQ (단일 베이스, ~20GB VRAM)
  ├─ LoRA 없음      → planner (tool calling 네이티브)
  ├─ LoRA #1 civil  → draft_civil_response
  └─ LoRA #2 legal  → append_evidence
```

vLLM 서빙: `--enable-lora --enable-auto-tool-choice --tool-call-parser hermes`

---

## 현재 유효 어댑터

| 어댑터 | Repository | 용도 | 상태 |
|--------|-----------|------|------|
| civil-adapter | [umyunsang/GovOn-EXAONE-LoRA-v2](https://huggingface.co/umyunsang/GovOn-EXAONE-LoRA-v2) | `draft_civil_response` | ✅ 운영 중 |
| legal-adapter | [siwo/govon-legal-adapter](https://huggingface.co/siwo/govon-legal-adapter) | `append_evidence` | ✅ 학습 완료 |

### civil-adapter

- **Base**: LGAI-EXAONE/EXAONE-Deep-7.8B → EXAONE 4.0-32B-AWQ로 마이그레이션 예정
- **학습 데이터**: `umyunsang/govon-civil-response-data` (74K건)
- **방법**: QLoRA (4-bit NF4), rank=64, alpha=128

### legal-adapter (신규)

- **Repository**: [siwo/govon-legal-adapter](https://huggingface.co/siwo/govon-legal-adapter)
- **Base**: `LGAI-EXAONE/EXAONE-4.0-32B` (Unsloth 4-bit QLoRA)
- **학습 데이터**: `umyunsang/govon-legal-response-data` (100K건, 1차 / 총 270K)
- **방법**: Unsloth QLoRA, rank=16, alpha=32, target: 7 projection modules
- **학습 환경**: HuggingFace Spaces A100 80GB
- **Final loss**: 0.889 (365 steps)
- **학습 스크립트**: `training/legal_adapter/train_qlora.py`
- **평가 결과**: `training/legal_adapter/experiment_results.md`

---

## 폐기 모델 (v1 — 사용 금지)

> **⚠ 폐기 안내 (2026-03-19)**: 아래 v1 모델은 **잘못 학습된 LoRA 어댑터**를 기반으로 생성되어 전량 폐기 대상입니다.

| Model | Type | 폐기 사유 |
|-------|------|-----------|
| ~~[civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora)~~ | LoRA v1 | 잘못된 학습 설정 (pad_token 오류) |
| ~~[civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged)~~ | Merged BF16 | 폐기된 LoRA v1 기반 |
| ~~[civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq)~~ | AWQ 4-bit | 폐기된 병합 모델 기반 |
