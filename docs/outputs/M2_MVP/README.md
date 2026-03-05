# M2: MVP (Week 5-8)

## 주요 산출물

### 실험 계획 및 가이드 문서
- [x] **experiment_plan.md** - EXAONE-3.5-7.8B QLoRA 파인튜닝 및 AWQ 양자화 실험 계획서
  - 실험 개요 및 가설
  - 모델 및 데이터셋 구성
  - QLoRA 파인튜닝 설계
  - AWQ 양자화 설계
  - 평가 메트릭 및 벤치마크
  - 실험 일정 및 리소스 계획

### Colab 실행 가이드
- [x] **notebooks/00_COLAB_QUICKSTART_GUIDE.md** - Google Colab Pro A100 환경 실행 가이드
  - Step-by-step 실행 가이드 (환경 설정 → 학습 → 양자화 → 평가)
  - 예상 소요 시간: 총 10시간 20분
  - 문제 해결 (Troubleshooting) 가이드
  - 체크리스트 및 검증 방법

---

## 산출물 체크리스트

### Week 5: 모델 준비 및 데이터 수집
- [x] experiment_plan.md - 학습 실험 계획서
- [x] notebooks/00_COLAB_QUICKSTART_GUIDE.md - Colab 실행 가이드
- [x] model_download_log.md - EXAONE 모델 다운로드 로그
- [x] data_collection_log.md - AI Hub 데이터 수집 로그
- [ ] calibration_dataset/ - AWQ 캘리브레이션 데이터셋 (512 샘플)
- [x] peft_config.json - QLoRA 설정 파일

### Week 6: QLoRA 파인튜닝 실험
- [x] training_logs/ - 학습 로그 (WandB, TensorBoard)
  - [x] exp001_baseline_log.json - Baseline 실험 (r=16) 완료 (Eval Loss: 1.0179)
  - [ ] exp002_rank8_log.json - Rank=8 실험
  - [ ] exp003_rank32_log.json - Rank=32 실험
- [x] checkpoints/ - 학습 체크포인트
  - [x] exaone-qlora-baseline/ - [final](https://huggingface.co/umyunsang/civil-complaint-exaone-lora) 저장 완료
  - [x] lora_adapter/ - LoRA 어댑터 (배포용, ~38MB)
- [ ] hyperparameter_tuning.md - 하이퍼파라미터 튜닝 결과 분석
- [x] wandb_run_urls.md - WandB 실험 추적 URL 목록 ([EXP-001-Baseline-EXAONE-7.8B](https://wandb.ai/umyun3/huggingface/runs/kmx8rlvv))

### Week 7: AWQ 양자화 및 평가
- [ ] merged_model/ - LoRA 병합 모델 (bf16)
- [ ] quantized_model/ - AWQ 4-bit 양자화 모델 (~4GB)
- [ ] quantization_log.md - 양자화 과정 로그
- [ ] evaluation_report.md - 성능 평가 리포트
  - 민원 분류 정확도
  - 답변 생성 품질 (BLEU, ROUGE)
  - 추론 속도 벤치마크
  - 메모리 사용량 프로파일
- [ ] benchmark_results.json - 벤치마크 결과 (JSON)

### Week 8: 백엔드 개발 및 통합
- [x] src/training/ - 학습 스크립트
  - [x] train_qlora.py - QLoRA 학습 메인 스크립트
  - [x] trainer_config.py - TrainingArguments 설정
- [ ] src/quantization/ - 양자화 스크립트
  - [ ] quantize_awq.py - AWQ 양자화 스크립트
  - [ ] merge_lora.py - LoRA 병합 스크립트
- [ ] src/evaluation/ - 평가 스크립트
  - [ ] evaluate_model.py - 종합 평가 스크립트
  - [ ] metrics.py - 평가 메트릭 정의
  - [ ] benchmark.py - 추론 속도 벤치마크
- [ ] notebooks/ - Jupyter 노트북
  - [x] 01_setup_environment.ipynb
  - [x] 02_data_preparation.ipynb
  - [x] 03_qlora_training.ipynb
  - [ ] 04_awq_quantization.ipynb
  - [ ] 05_evaluation.ipynb

---

## 완료 기준

### 기술적 완료 기준
- [x] QLoRA 파인튜닝 성공 (1 epoch, validation loss < 1.1)
- [ ] AWQ 양자화 완료 (모델 크기 50% 이상 감소)
- [ ] 민원 분류 정확도 ≥ 85% 달성
- [ ] 답변 생성 BLEU ≥ 30, ROUGE-L ≥ 40 달성
- [ ] 추론 속도 p50 < 2초, p95 < 5초 달성
- [ ] GPU VRAM 사용량 < 8GB (AWQ 모델)

### 문서화 완료 기준
- [x] 실험 계획서 작성 완료
- [x] Colab 실행 가이드 작성 완료
- [x] 실험 결과 기록 및 추적 가이드 작성 완료
- [ ] 평가 리포트 작성 완료
- [ ] 하이퍼파라미터 튜닝 분석 완료
- [x] 최종 모델 배포 가이드 (HuggingFace Model Card) 작성 완료

### 멘토 점검 준비
- [ ] MVP 데모 준비 (추론 테스트 영상)
- [ ] 실험 결과 요약 발표 자료
- [ ] 멘토 중간 점검 통과

---

## 실험 진행 현황

### 현재 상태
**Phase**: Week 6 - QLoRA Baseline 실험 완료 및 모델 배포

**완료된 작업**:
- 실험 계획서 및 결과 기록 문서 작성
- AI Hub 데이터 수집 및 전처리 (71852, 71844)
- EXAONE-Deep-7.8B QLoRA Baseline (r=16) 학습 완료 (Best Eval Loss: 1.0179)
- **Hugging Face Model Adapter**: [umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora)
- WandB 실험 로그 연동 완료 ([EXP-001-Baseline-EXAONE-7.8B](https://wandb.ai/umyun3/huggingface/runs/kmx8rlvv))
- HuggingFace Model Hub 어댑터 배포 완료 ([umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora))

**진행 중인 작업**:
- LoRA Merge & AWQ Quantization 환경 구축
- 캘리브레이션 데이터셋(512 샘플) 생성

**다음 단계** (Week 7):
1. LoRA Merge (bf16)
2. AutoAWQ를 이용한 4-bit 양자화 실행
3. vLLM 기반 추론 속도 및 VRAM 사용량 벤치마킹
4. 민원 분류 및 답변 생성 정량적 평가 (Test Set)

---

## 예상 일정 (Detailed)

### Week 5 (D1-D5): 모델 준비
| Day | 작업 | 담당 | 상태 |
|-----|------|------|------|
| D1 | Colab 환경 설정 | 팀 전체 | ⏳ 진행 중 |
| D1-2 | AI Hub 데이터 다운로드 | Data | ⏳ 진행 중 |
| D2 | 데이터 전처리 실행 | Data | ⬜ 대기 |
| D3 | 캘리브레이션 데이터 생성 | Data | ⬜ 대기 |
| D3-4 | EXAONE 모델 다운로드 | ML | ⬜ 대기 |
| D4-5 | 학습 스크립트 작성 | ML | ⬜ 대기 |

### Week 6 (D6-D10): QLoRA 파인튜닝
| Day | 실험 | 설정 | 상태 |
|-----|------|------|------|
| D6 | EXP-001 Baseline | r=16, lr=2e-4 | ⬜ 대기 |
| D7 | EXP-002 Rank=8 | r=8 | ⬜ 대기 |
| D8 | EXP-003 Rank=32 | r=32 | ⬜ 대기 |
| D9 | EXP-005 LR=1e-4 | lr=1e-4 | ⬜ 대기 |
| D10 | 중간 결과 분석 | - | ⬜ 대기 |

### Week 7 (D11-D14): AWQ 양자화 및 평가
| Day | 작업 | 상태 |
|-----|------|------|
| D11 | LoRA 병합 + AWQ 양자화 | ⬜ 대기 |
| D12 | 양자화 실험 (그룹 크기) | ⬜ 대기 |
| D13 | 성능 평가 | ⬜ 대기 |
| D14 | 벤치마크 및 분석 | ⬜ 대기 |

### Week 8 (D15-D18): 통합 및 문서화
| Day | 작업 | 상태 |
|-----|------|------|
| D15 | 최종 모델 선정 및 저장 | ⬜ 대기 |
| D16-17 | 평가 리포트 작성 | ⬜ 대기 |
| D18 | 멘토 중간 점검 준비 | ⬜ 대기 |

**범례**: ✅ 완료 | ⏳ 진행 중 | ⬜ 대기 | ⚠ 문제 발생

---

## 리소스 및 참고 자료

### 실험 환경
- **GPU**: Google Colab Pro A100 (40GB VRAM)
- **베이스 모델**: [LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)
- **데이터셋**: AI Hub 71852 (공공 민원), 71844 (민간 민원)

### 모니터링 대시보드
- **Weights & Biases**: [프로젝트 대시보드 URL] (실험 추적)
- **TensorBoard**: `/content/logs/` (로컬 로그)

### 관련 문서
- [experiment_plan.md](./experiment_plan.md) - 상세 실험 계획서
- [00_COLAB_QUICKSTART_GUIDE.md](../../notebooks/00_COLAB_QUICKSTART_GUIDE.md) - Colab 실행 가이드
- [M1 Planning](../M1_Planning/README.md) - 데이터 수집 및 시스템 설계
- [PRD](../../prd.md) - 프로젝트 요구사항 정의서

---

**작성일**: 2026-03-05
**최종 수정일**: 2026-03-05
**담당 멘토**: 천세진 교수
