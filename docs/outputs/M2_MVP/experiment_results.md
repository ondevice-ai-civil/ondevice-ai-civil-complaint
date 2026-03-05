# EXAONE-Deep-7.8B 실험 결과 기록 및 추적 가이드

본 문서는 `M2: MVP` 단계에서 진행되는 QLoRA 파인튜닝 실험 결과를 체계적으로 기록하고 추적하기 위해 작성되었습니다.

## 1. 실험 기록 방법
각 실험은 아래 템플릿에 따라 독립된 섹션으로 작성합니다. 실험 ID는 `EXP-XXX` 형식을 따릅니다.

### 📋 실험 기록 템플릿
```markdown
### [EXP-XXX] 실험 제목 (예: Baseline r=16, lr=2e-4)
- **일시**: 2026-03-05
- **설정 요약**: r=16, alpha=32, batch_size=16(eff), epochs=1
- **환경**: Google Colab L4 (24GB VRAM)
- **주요 결과**:
  - Final Loss: 0.XXXX
  - Eval Loss: 0.XXXX
  - Training Time: Xh XXm
- **WandB Run**: [링크]
- **비고**: 수렴 속도가 예상보다 빠름 / OOM 발생으로 배치 사이즈 조정 등
```

## 2. 실험 로그 (Experiment Logs)

### [EXP-001] Baseline QLoRA 학습
- **일시**: 2026-03-05
- **설정 요약**: 
  - 모델: LGAI-EXAONE/EXAONE-Deep-7.8B
  - Rank (r): 16
  - Alpha: 32
  - Learning Rate: 2e-4
  - Batch Size: 2 (Grad Accum: 8) -> Effective Batch Size: 16
  - Max Seq Length: 2048
  - Epochs: 1
- **환경**: Google Colab A100 (80GB VRAM) - (L4에서 이동)
- **주요 결과**:
  - Step 100: Eval Loss 1.1938
  - Step 400: Eval Loss 1.0443
  - **Final (Step 781)**: Training Loss ~1.01 / **Best Eval Loss 1.0179** (at Step 700)
  - **Status**: **Completed**
- **WandB Run**: [offline-run-kmx8rlvv](https://wandb.ai/offline) (로컬 기록 완료)
- **비고**: 
  - `transformers 5.3.0` 및 `trl 0.12.0` 호환성 패치 적용 후 실행됨.
  - L4 런타임(24GB)에서 OOM 발생 후 A100으로 이전하여 성공적으로 완주.
  - 1 epoch 학습만으로도 Eval Loss 1.01 수준의 우수한 수렴도를 보임. 
  - 최종 모델은 `models/checkpoints/exaone-civil-qlora/final`에 저장됨.

---

## 3. 실험 결과 요약 및 비교 테이블

| 실험 ID | 설명 | r | lr | Epochs | Final Loss | Eval Loss | 비고 |
|---------|------|---|---|--------|------------|-----------|------|
| EXP-001 | Baseline | 16 | 2e-4 | 1 | 1.01 | 1.0179 | 학습 완료 (A100) |

## 4. 추적 가이드
1. **정기적 기록**: 매 실험 완료 후 Final Loss 및 Eval Loss를 테이블에 업데이트합니다.
2. **로그 보존**: 체크포인트 폴더의 `trainer_state.json`을 참고하여 상세 메트릭을 기록합니다.
3. **오류 기록**: 학습 도중 발생한 이슈(OOM, 하이퍼파라미터 발산 등)를 비고란에 기록하여 재현 시 참고합니다.
