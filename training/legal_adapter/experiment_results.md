# Legal Adapter 학습 실험 결과

## 개요

EXAONE 4.0-32B 기반 법률해석·근거인용 전용 LoRA 어댑터 (1차 학습).
evidence augmentation(근거 보강) follow-up에서 법령 조항 인용 품질 향상을 목표로 함.

- **HF Hub**: [siwo/govon-legal-adapter](https://huggingface.co/siwo/govon-legal-adapter)
- **학습 일시**: 2026-04-06
- **관련 이슈**: #471

---

## 학습 환경

| 항목 | 값 |
|------|-----|
| 하드웨어 | HuggingFace Spaces A100 SXM4 80GB |
| 프레임워크 | Unsloth + TRL SFTTrainer |
| 베이스 모델 | `LGAI-EXAONE/EXAONE-4.0-32B` |
| 양자화 | 4-bit NF4 (Unsloth) |

---

## 학습 설정

| 하이퍼파라미터 | 값 |
|--------------|-----|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down_proj (7개) |
| LoRA dropout | 0 |
| Gradient checkpointing | unsloth |
| Dataset | `umyunsang/govon-legal-response-data` |
| Train samples | 100,000 (1차 / 총 270K) |
| Val samples | 26,983 (전체 validation split) |
| Max seq length | 1024 |
| Packing | True |
| Batch size | 8 |
| Gradient accumulation | 8 (eff. batch 64) |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Optimizer | adamw_8bit |
| Warmup ratio | 0.03 |
| Epochs | 1 |
| Precision | BF16 + TF32 |

---

## 학습 결과

| 지표 | 값 |
|------|-----|
| 총 스텝 | 365 |
| 초기 loss (step 1) | 2.334 |
| 최종 loss (step 365) | 0.889 |
| 스텝당 소요 시간 | ~1.2분 |
| 총 학습 시간 | ~7시간 (A100 80GB) |

### Loss 곡선 (주요 체크포인트)

| Step | Loss |
|------|------|
| 1 | 2.334 |
| 60 | 0.978 |
| 320 | 0.889 |
| 365 | 0.889 |

---

## 평가 결과

> 평가는 `evaluate.py`를 사용하여 vLLM 엔드포인트 대상으로 실행.
> `python evaluate.py --sample-size 100`

평가 결과는 vLLM Multi-LoRA 서빙 통합(#468 확장) 후 업데이트 예정.

---

## 산출물

| 파일 | 위치 |
|------|------|
| 학습 스크립트 | `training/legal_adapter/train_qlora.py` |
| HF Space 스크립트 | [siwo/govon-legal-adapter-train](https://huggingface.co/spaces/siwo/govon-legal-adapter-train) |
| 어댑터 가중치 | [siwo/govon-legal-adapter](https://huggingface.co/siwo/govon-legal-adapter) |
| 학습 데이터셋 | [umyunsang/govon-legal-response-data](https://huggingface.co/datasets/umyunsang/govon-legal-response-data) |

---

## 2차 학습 계획

- 샘플 범위: `select(range(100_000, 242_854))` (나머지 ~143K)
- 베이스: `siwo/govon-legal-adapter` (1차 어댑터 로드 후 이어서 학습)
- 예상 시간: ~9시간 (A100 80GB)
