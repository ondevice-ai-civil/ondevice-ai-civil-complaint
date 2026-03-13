# Notebooks 가이드

Colab A100 런타임 기준으로 작성된 실험 노트북 모음입니다.
모든 노트북은 W&B 로깅을 포함합니다.

## 폴더 구조

```
notebooks/
├── README.md
│
├── M3_issue70_data/              # [최우선] 학습 데이터 전면 재구성
│   └── 01_data_reconstruction.ipynb    ← Issue #70
│
├── M3_issue82_evaluation/        # [최우선] 평가 파이프라인 표준화
│   └── 02_evaluate_standard.ipynb      ← Issue #82
│
├── M3_issue67_hparam/            # [Phase 2] QLoRA 하이퍼파라미터 최적화
│   └── 03_qlora_hparam_optimization.ipynb  ← Issue #67
│
├── M3_issue68_generation/        # [Phase 2] 답변 생성 품질 고도화
│   └── (디코딩 파라미터 그리드서치 노트북 예정)  ← Issue #68
│
├── M3_issue69_inference/         # [Phase 3] 추론 속도 최적화
│   └── (AWQ 양자화 + 벤치마크 노트북 예정)     ← Issue #69
│
├── M2_MVP/                       # M2 마일스톤 (완료)
│   └── qlora_training.ipynb            ← EXP-001 Baseline 학습
│
└── tools/                        # 유틸리티 / 기타 도구
    ├── 00_COLAB_QUICKSTART_GUIDE.md
    ├── ollama_claude_code_colab.ipynb
    └── wb_skills_experiment_colab.ipynb
```

## 실행 순서 (우선순위)

| 순서 | 노트북 | 이슈 | W&B 프로젝트 | 설명 |
|------|--------|------|-------------|------|
| 1 | `M3_issue70_data/01_data_reconstruction.ipynb` | #70 | `govon-data-reconstruction` | 71852 데이터 재처리, 콜센터 제거, PII 개선 |
| 2 | `M3_issue82_evaluation/02_evaluate_standard.ipynb` | #82 | `govon-evaluation` | SacreBLEU/ROUGE-L/BERTScore 표준 평가 |
| 3 | `M3_issue67_hparam/03_qlora_hparam_optimization.ipynb` | #67 | `govon-qlora-hparam-search` | Rank/LR/Epoch 체계적 탐색 |
| 4 | `M3_issue68_generation/` | #68 | `govon-generation-quality` | 디코딩 파라미터 최적화 |
| 5 | `M3_issue69_inference/` | #69 | `govon-inference-speed` | AWQ 재양자화 + 속도 벤치마크 |

## 환경 요구사항

- **런타임**: Google Colab A100 (40GB VRAM)
- **W&B**: `wandb login` 필요 (API 키)
- **모델**: LGAI-EXAONE/EXAONE-Deep-7.8B (HuggingFace)
- **데이터**: Google Drive에 프로젝트 클론 필요

## 의존성 체인

```
#70 데이터 재구성 ──→ #67 HP 최적화 ──→ #68 생성 품질 ──→ #69 추론 속도
         ↘                                    ↗
          #82 평가 파이프라인 ─────────────────
```
