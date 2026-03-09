# M2: 핵심 기능 구현 - MVP (Week 5-8)

**기간**: 2026-03-31 ~ 2026-04-25
**상태**: 핵심 완료 (75%) - AI 모델 파이프라인 완료, 백엔드/프론트엔드(Figma MCP 기반) 미구현
**최종 수정**: 2026-03-09

---

## 진행 현황 요약

M2 단계에서는 EXAONE-Deep-7.8B 모델의 QLoRA 파인튜닝, AWQ 양자화, HuggingFace 배포를 성공적으로 완료했습니다.
모델 크기를 14.56GB에서 4.94GB로 66% 압축하면서 GPU VRAM 4.95GB 수준의 온디바이스 배포 가능성을 검증했습니다.
다만 백엔드 API(FastAPI)와 프론트엔드(Figma MCP 기반 React/Next.js) 구현은 M3 단계로 이월되었습니다.

---

## 주요 성과 (KPI)

| 지표 | 목표 | M2 측정값 | M3 최적화 후 | 상태 |
|------|------|-----------|-------------|------|
| Perplexity | 최저 수렴 | **3.1957** | - | 달성 |
| 분류 정확도 | >= 85% | 2.00% | **90.0%** | M3에서 달성 |
| BLEU | >= 30 | 12.29 | - | 미달 (프롬프트 개선 필요) |
| ROUGE-L | >= 40 | 21.36 | - | 미달 (BERTScore 도입) |
| BERTScore F1 | 베이스라인 | - | **46.05** | M3에서 확보 |
| 추론 속도 (Avg) | < 2s | 9.291s | **2.43s** | M3에서 근접 |
| GPU VRAM | < 8GB | **4.95 GB** | **4.17 GB** | 달성 |
| 모델 크기 | < 5GB | **4.94 GB** | - | 달성 |

---

## 산출물 체크리스트

### Week 5: 모델 준비 및 데이터 수집
- [x] experiment_plan.md - 학습 실험 계획서
- [x] COLAB_FINETUNING_GUIDE.md - Colab 실행 가이드
- [x] model_download_log.md - EXAONE 모델 다운로드 로그
- [x] data_collection_log.md - AI Hub 데이터 수집 로그
- [x] src/training/peft_config.json - QLoRA 설정 파일

### Week 6: QLoRA 파인튜닝 실험
- [x] EXP-001 Baseline (r=16, lr=2e-4) - Eval Loss: 1.0179
- [ ] EXP-002 Rank=8 실험 (미실행)
- [ ] EXP-003 Rank=32 실험 (미실행)
- [x] LoRA 어댑터 HuggingFace 배포 (umyunsang/civil-complaint-exaone-lora)
- [x] WandB 실험 로그 연동 (EXP-001-Baseline)

### Week 7: AWQ 양자화 및 평가
- [x] LoRA 병합 BF16 모델 (umyunsang/civil-complaint-exaone-merged, 14.56GB)
- [x] AWQ 4-bit 양자화 모델 (umyunsang/civil-complaint-exaone-awq, 4.94GB, 2.95x 압축)
- [x] evaluation_report.md - 성능 평가 리포트
- [x] benchmark_results.json - 벤치마크 결과 (JSON)

### Week 8: 백엔드 개발 및 통합
- [x] src/training/train_qlora.py - QLoRA 학습 스크립트
- [x] src/quantization/quantize_awq.py - AWQ 양자화 스크립트
- [x] src/quantization/merge_lora.py - LoRA 병합 스크립트
- [x] src/evaluation/evaluate_model.py - 종합 평가 스크립트
- [ ] FastAPI 백엔드 프로젝트 구축 (M3로 이월)
- [ ] vLLM OpenAI 호환 API 연동 (M3로 이월)
- [ ] Figma MCP 기반 React/Next.js 웹 UI (M3/M4로 이월)
- [ ] MVP 통합 테스트 및 데모 (M3/M4로 이월)

---

## 완료 기준

### 기술적 완료 기준
- [x] QLoRA 파인튜닝 성공 (1 epoch, Eval Loss 1.0179 < 1.1)
- [x] AWQ 양자화 완료 (모델 크기 66.1% 감소, 2.95x 압축)
- [x] GPU VRAM 사용량 < 8GB (실측 4.95 GB)
- [ ] 답변 생성 BLEU >= 30, ROUGE-L >= 40 (현재 BLEU 12.29, ROUGE-L 21.36)
- [ ] 추론 속도 p50 < 2초 (M3에서 2.43초까지 개선)

### 문서화 완료 기준
- [x] 실험 계획서 작성 완료
- [x] Colab 실행 가이드 작성 완료
- [x] 평가 리포트 작성 완료
- [x] 최종 모델 HuggingFace Model Card 배포 완료
- [ ] 하이퍼파라미터 튜닝 분석 (baseline r=16만 실험)

### 멘토 점검 준비
- [ ] MVP 데모 준비
- [ ] 실험 결과 요약 발표 자료
- [ ] 멘토 중간 점검 통과

---

## 모델 배포 정보 (HuggingFace)

| 모델 | 크기 | 링크 |
|------|------|------|
| LoRA Adapter | ~38MB | [umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora) |
| Merged BF16 | 14.56GB | [umyunsang/civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged) |
| AWQ 4-bit | 4.94GB | [umyunsang/civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq) |

---

## 기술적 이슈 및 해결

1. **transformers 5.x 호환성**: EXAONE 아키텍처 `get_interface` 오류 - 소스코드 수동 패치
2. **분류 정확도 0%**: `<thought>` 태그 파싱 문제 - 정규표현식 분리 및 Chat Template 적용
3. **높은 추론 레이턴시**: `max_new_tokens` 과도 설정 - 토큰 제한 및 repetition_penalty 적용

---

## 관련 GitHub 이력

| 항목 | 참조 |
|------|------|
| PR #11 | feat: fine-tuned EXAONE-Deep LoRA adapter |
| PR #10 | feat: 모델 학습 환경 구축 |
| PR #19 | fix(M2-MVP): 평가 스크립트 수정 및 AWQ 모델 배포 |
| PR #24 | M2 MVP Final Report & M3 Optimization Complete |
| Issue #17 | MVP: QLoRA 파인튜닝 및 AWQ 최적화 진행 현황 (Closed) |
| Issue #18 | AWQ 양자화 완료, 평가 스크립트 수정 (Closed) |

---

**작성일**: 2026-03-05
**최종 수정일**: 2026-03-09
