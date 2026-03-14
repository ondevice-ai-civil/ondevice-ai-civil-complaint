# 학술 자료 정리 (Research Documents)

> 학술대회/학술지 작성을 위한 기술 문서 모음
> 모델 분석, 파인튜닝, 경량화, 평가, 트러블슈팅 관련 문서를 카테고리별로 정리

---

## 01. 모델 구조 및 특징 분석 (`01_model_analysis/`)

| 문서 | 내용 | 원본 위치 |
|------|------|-----------|
| [exaone_analysis.md](01_model_analysis/exaone_analysis.md) | EXAONE-Deep-7.8B 아키텍처, GQA, RoPE, AWQ 양자화 특징, 한국어 성능, 하드웨어 요구사항, vLLM 호환성 종합 분석 | `docs/outputs/M1_Planning/01_Kickoff/` |

**주요 내용**: 7.8B 파라미터, 32 레이어, GQA(32Q/8KV), RoPE(llama3), 32K 컨텍스트, 102,400 어휘, CSAT Math 89.9%, MATH-500 94.8%

---

## 02. QLoRA 파인튜닝 (`02_finetuning/`)

| 문서 | 내용 | 원본 위치 |
|------|------|-----------|
| [experiment_plan.md](02_finetuning/experiment_plan.md) | 실험 설계서 — 연구 가설, KPI, QLoRA 설정, 데이터셋 구성, 평가 메트릭 | `docs/outputs/M2_MVP/` |
| [experiment_results.md](02_finetuning/experiment_results.md) | 실험 결과 기록 — EXP-001 Baseline (r=16, lr=2e-4, Eval Loss 1.0179) | `docs/outputs/M2_MVP/` |
| [COLAB_FINETUNING_GUIDE.md](02_finetuning/COLAB_FINETUNING_GUIDE.md) | Colab 실행 가이드 — 라이브러리 설치, 환경 설정, 실행 명령어 | `docs/outputs/M2_MVP/` |
| [ENVIRONMENT_NOTES.md](02_finetuning/ENVIRONMENT_NOTES.md) | 학습 환경 설정 — transformers 패치, 몽키패칭, TrainingArguments 변경 | `src/training/` |

**주요 내용**: QLoRA(NF4, r=16, alpha=32), 7개 타겟 모듈, AI Hub 71852+71844 데이터, 1 epoch 학습, Colab A100

---

## 03. AWQ 경량화 (`03_quantization/`)

| 문서 | 내용 | 원본 위치 |
|------|------|-----------|
| [awq_quantization_analysis.md](03_quantization/awq_quantization_analysis.md) | AWQ 양자화 분석 — W4A16g128 설정, 파이프라인, 결과, merged 모델 손상 문제, 재양자화 계획 | 신규 작성 |

**주요 내용**: W4A16g128, 15.6GB→4.94GB(68.3% 감소), VRAM 4.95GB, merged 모델 손상 발견 및 대응

---

## 04. 평가 결과 (`04_evaluation/`)

| 문서 | 내용 | 원본 위치 |
|------|------|-----------|
| [evaluation_report.md](04_evaluation/evaluation_report.md) | M2 MVP 성능 평가 — Perplexity 3.20, BLEU 17.32, ROUGE-L 18.28, 분류 정확도 이슈 분석 | `docs/outputs/M2_MVP/` |
| [MVP_FINAL_SUMMARY.md](04_evaluation/MVP_FINAL_SUMMARY.md) | M2 MVP 최종 보고서 — KPI 달성 현황, 기술적 이슈 해결 요약 | `docs/outputs/M2_MVP/` |
| [FINAL_M3_COMPLETION_REPORT.md](04_evaluation/FINAL_M3_COMPLETION_REPORT.md) | M3 최종 보고서 — vLLM 적용, 분류 정확도 90%, 추론 속도 2.43s, BERTScore 46.05 | `docs/outputs/M3_Optimization/` |

**주요 내용**: M2→M3 성능 개선 추적, 분류 정확도 2%→90%, 추론 속도 9.29s→2.43s

---

## 05. 환경 문제 및 트러블슈팅 (`05_troubleshooting/`)

| 문서 | 내용 | 원본 위치 |
|------|------|-----------|
| [colab-version-compatibility.md](05_troubleshooting/colab-version-compatibility.md) | Colab 버전 호환성 문제 — EXAONE 코드 재작성(2026-02), transformers 버전 충돌, 4가지 오류 상세, 해결 방법 | `docs/` |

**주요 내용**: trust_remote_code 동적 코드 다운로드 위험, transformers 4.44~4.49 고정, EXAONE revision `17b70148e344` 고정, 몽키패치 불필요 원칙

---

## 학술 활용 시 핵심 포인트

### 연구 기여점
1. **한국어 LLM 도메인 적응**: EXAONE-Deep-7.8B를 민원 도메인에 QLoRA로 특화
2. **경량화 파이프라인**: QLoRA → LoRA Merge → AWQ 4-bit 양자화 전 과정 실증
3. **온디바이스 배포 가능성**: 4.94GB 모델로 소비자급 GPU에서 실행 가능 검증
4. **버전 호환성 문제 분석**: trust_remote_code 기반 모델의 재현성 위험 체계적 분석

### 실험 환경 요약
- **학습**: Google Colab A100 (40/80GB), transformers ~4.44, QLoRA 4-bit
- **평가**: Google Colab L4 (24GB), transformers 4.44~4.49, 920 테스트 샘플
- **배포**: vLLM 0.17.0, AWQ W4A16g128, Marlin 커널

### 데이터셋
- AI Hub 71852 (공공 민원 상담 LLM 데이터)
- AI Hub 71844 (민간 민원 상담 LLM 데이터)
- PII 마스킹 적용 (개인정보 보호)
