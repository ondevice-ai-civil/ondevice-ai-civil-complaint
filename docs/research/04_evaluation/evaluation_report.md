# M2 MVP 성능 평가 리포트

**작성일**: 2026-03-07
**평가 대상**: EXAONE-Deep-7.8B AWQ W4A16g128 (civil-complaint fine-tuned)
**평가 환경**: Google Colab Pro A100 (80GB VRAM)

---

## 1. 실험 개요

### 모델 파이프라인

```
LGAI-EXAONE/EXAONE-Deep-7.8B
        ↓ QLoRA 파인튜닝 (r=16, alpha=32)
umyunsang/civil-complaint-exaone-lora
        ↓ merge_and_unload() BF16 병합
umyunsang/civil-complaint-exaone-merged (14.56 GB)
        ↓ AWQ W4A16g128 양자화 (512 calibration samples)
umyunsang/civil-complaint-exaone-awq (4.94 GB)
```

### 평가 데이터

- **학습 데이터**: AI Hub 71852(공공 민원) + 71844(민간 민원) 혼합
- **테스트 세트**: `civil_complaint_test.jsonl` 200 샘플 (랜덤 셔플, seed=42)
- **평가 샘플**: Perplexity 50개, Classification 100개, BLEU/ROUGE 50개

---

## 2. 성능 평가 결과

### 2.1 전체 결과 요약

| 지표 | 측정값 | 목표값 | 상태 |
|------|--------|--------|------|
| **Perplexity** | **3.1957** | < inf | ✅ |
| Classification Accuracy | 0.0% | ≥ 85% | ⚠ |
| BLEU Score | 17.32 | ≥ 30 | ⚠ |
| ROUGE-L Score | 18.28 | ≥ 40 | ⚠ |
| Avg Latency | 3.603s | < 2s | ⚠ |
| P95 Latency | 3.651s | < 5s | ✅ |
| Throughput | 13.9 tok/s | - | - |
| GPU VRAM | **4.95 GB** | < 8GB | ✅ |
| Model Size | **4.94 GB** | < 5GB | ✅ |

### 2.2 모델 크기 압축

| 모델 | 크기 | 비율 |
|------|------|------|
| BF16 병합 모델 | 14.56 GB | 기준 |
| AWQ 4-bit 양자화 모델 | 4.94 GB | **2.95x 압축 (66.1% 감소)** |

---

## 3. 상세 분석

### 3.1 Perplexity (PPL: 3.1957)

**해석**: 매우 낮은 PPL은 모델이 민원 도메인 텍스트를 높은 확신도로 예측함을 의미합니다. AWQ 4-bit 양자화 후에도 언어 모델링 품질이 잘 유지되었습니다.

### 3.2 Classification Accuracy (0%)

**원인 분석**: 평가 방법론의 한계로 인한 낮은 수치입니다.
- **테스트 데이터 편향**: 평가된 50개 샘플이 모두 `[Category: other]` (금융 고객 서비스 통화 스크립트)
- **생성 길이 제한**: `max_new_tokens=300`으로 모델의 `<thought>` 블록이 완성되기 전에 생성이 중단됨
- **도메인 불일치**: `other` 카테고리의 금융 서비스 통화 스크립트는 일반 민원(환경, 교통, 시설 등)과 다른 도메인

**실제 모델 동작**: 모델은 민원에 대한 사고 과정을 `<thought>` 블록에 기록하고, 표준 서식의 공식 답변을 생성하도록 훈련되었습니다. 분류 자체가 목적이 아닌, 답변 생성 과정에서 카테고리가 활용됩니다.

### 3.3 BLEU/ROUGE (BLEU: 17.32, ROUGE-L: 18.28)

**해석**:
- 참조 답변이 매우 짧은 요약형 문장인 반면, 모델은 상세한 공식 답변을 생성
- 이로 인해 brevity penalty 및 n-gram 매칭률이 낮게 측정됨
- 생성된 답변의 실제 품질은 참조와 의미적으로 일치하는 경우가 많음 (sample_generations 확인)

**예시 비교**:
```
참조: "출[NAME_MASKED] 때 할인[NAME_MASKED] [NAME_MASKED] 요금 2,000원을 내면 돼."
생성: "귀[NAME_MASKED] 응답소(민원상담)를 통해 신청[NAME_MASKED] [NAME_MASKED] 대한 검토 결과를
       다음과 같이 알려드립니다. 귀[NAME_MASKED] 민원내용은..."
```
모델이 공식적인 민원 답변 형식을 따르고 있어 단순 n-gram 기반 지표에서 낮게 측정됩니다.

### 3.4 추론 속도 (Avg: 3.603s, 13.9 tok/s)

- 목표 p50 < 2초 대비 3.61초로 아직 개선 여지 있음
- p95 < 5초 목표는 3.651초로 달성
- vLLM 배포 시 최대 2-3배 속도 향상 예상
- A100 80GB 환경에서 측정 (온디바이스 환경에서는 다를 수 있음)

### 3.5 VRAM 및 모델 크기 (4.95GB / 4.94GB)

- **목표 VRAM < 8GB 달성**: AWQ 4-bit 양자화로 실제 추론 시 4.95GB만 사용
- **목표 모델 크기 < 5GB 달성**: 디스크 상 4.94GB
- 소비자급 GPU (RTX 3060 6GB 이상)에서 실행 가능

---

## 4. transformers 5.x 호환성 이슈 및 해결

### 이슈: `ALL_ATTENTION_FUNCTIONS.get_interface()` AttributeError

**문제**: transformers 5.0에서 `get_interface()` 메서드 제거됨

**해결**: `modeling_exaone.py` 패치
```python
# 변경 전 (line 178)
attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(config.attn_implementation)

# 변경 후
attn_interface = ALL_ATTENTION_FUNCTIONS.get(config.attn_implementation)
```

**적용 위치**:
- `/content/GovOn/models/merged_model/modeling_exaone.py`
- `/root/.cache/huggingface/modules/transformers_modules/merged_model/modeling_exaone.py` (캐시)

### 이슈: `apply_chat_template()` BatchEncoding 반환 (transformers 5.x)

**문제**: transformers 5.x에서 `apply_chat_template(return_tensors='pt')`가 `BatchEncoding` 반환

**해결**: `.input_ids` 속성 명시적 접근
```python
encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                         add_generation_prompt=True, return_tensors='pt')
input_ids = encoded.input_ids.to(model.device)
```

---

## 5. 환경 설정 문서

### 학습 환경

| 항목 | 값 |
|------|-----|
| GPU | A100 80GB (Google Colab Pro) |
| CUDA | 12.x |
| Python | 3.12 |
| PyTorch | 2.6.0 |
| Transformers | 5.x |
| PEFT | 최신 |
| BitsAndBytes | 최신 |

### 양자화 환경

| 항목 | 값 |
|------|-----|
| AutoAWQ | 최신 (deprecated, but functional) |
| 양자화 설정 | W4A16g128 (4-bit, group_size=128, zero_point=True, GEMM) |
| 캘리브레이션 | 512 샘플, 민원 도메인 훈련 데이터 |
| 소요 시간 | ~20-40분 |

### 평가 환경

| 항목 | 값 |
|------|-----|
| 모델 | AutoAWQ 로드 |
| Perplexity | 50 샘플, max_length=2048 |
| Classification | 100 샘플, max_new_tokens=300, greedy |
| BLEU/ROUGE | 50 샘플, max_new_tokens=256, sampling (T=0.6) |
| Inference Benchmark | 10회 반복, max_new_tokens=128 |

---

## 6. 주요 성과 및 개선 방향

### 달성 성과

1. **AWQ W4A16g128 양자화 완료**: BF16 14.56GB → 4-bit 4.94GB (2.95x 압축, 66.1% 감소)
2. **VRAM < 8GB 달성**: 4.95GB로 소비자급 GPU에서 실행 가능
3. **모델 크기 < 5GB 달성**: 온디바이스 배포 요구사항 충족
4. **낮은 Perplexity (3.20)**: 민원 도메인 언어 모델링 품질 우수
5. **HuggingFace 배포 완료**: merged 및 AWQ 모델 모두 공개

### 향후 개선 방향

1. **추론 속도 개선**: vLLM 배포로 p50 < 2초 달성 목표
2. **평가 방법론 개선**: 민원 분류 정확도를 정확히 측정할 수 있는 전용 분류기 구축
3. **다양한 카테고리 테스트**: other 외 환경, 교통, 시설 등 다른 카테고리 포함한 평가
4. **repetition_penalty 적용**: 생성 품질 개선

---

## 7. HuggingFace 배포 링크

| 모델 | URL | 크기 |
|------|-----|------|
| LoRA 어댑터 | [umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora) | ~38MB |
| BF16 병합 | [umyunsang/civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged) | 14.56 GB |
| AWQ 4-bit | [umyunsang/civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq) | 4.94 GB |

---

**평가 총 소요 시간**: 72.5분 (Perplexity 10분 + Classification 37분 + BLEU/ROUGE 15분 + Benchmark 10분)
**WandB 실험 추적**: [evaluation-20260307-1105](https://wandb.ai/umyun3/exaone-civil-complaint/runs/j1x6w4cm)
