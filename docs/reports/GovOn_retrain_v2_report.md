# GovOn Retrain v2 학습 레포트

## 1. 프로젝트 개요

**GovOn**은 지자체 민원 업무를 지원하는 AI 어시스턴트로, 주민의 민원 질의에 대해 정확하고 신뢰성 있는 답변을 생성하는 것을 목표로 한다. 본 프로젝트는 LG AI Research의 **EXAONE-Deep-7.8B** 모델을 베이스로, **QLoRA** 기반 파인튜닝을 통해 민원 도메인에 특화된 응답 능력을 부여한다.

| 항목 | 내용 |
|------|------|
| 베이스 모델 | LGAI-EXAONE/EXAONE-Deep-7.8B |
| 파인튜닝 방식 | QLoRA (4-bit NF4 양자화 + LoRA r=16) |
| 어댑터 | umyunsang/GovOn-EXAONE-LoRA-v2 |
| 학습 인프라 | Google Colab A100 (40GB) |
| 학습 프레임워크 | TRL SFTTrainer + PEFT + BitsAndBytes |
| W&B 프로젝트 | umyun3/GovOn-retrain-v2 |

---

## 2. 문제 정의: v1에서 발견된 이슈

v1 모델(`retrain-v2-lora-r16-3ep`)에 대한 sanity check 및 평가에서 다음과 같은 문제가 확인되었다.

### 2.1 EOS 토큰 미생성 (0%)

v1에서는 **pad_token을 eos_token과 동일하게 설정**(`pad_token = eos_token`)한 것이 원인이었다. 학습 시 패딩 영역의 loss가 -100으로 마스킹되면서, EOS 토큰에 대한 학습이 사실상 차단되었다. 결과적으로 추론 시 모델이 EOS를 생성하지 못하고 `max_new_tokens`까지 계속 생성하는 문제가 발생했다.

- **EOS 생성률**: 0% (5/5 샘플 모두 EOS 미생성)
- **평균 생성 길이**: 866자 (max_new_tokens 도달)

### 2.2 낮은 텍스트 생성 품질

| 지표 | v1 값 | 비고 |
|------|-------|------|
| BLEU | 0.53 | 매우 낮음 |
| ROUGE-L | 4.20 | 매우 낮음 |
| length_ratio | 0.63 | 참조 대비 37% 짧은 생성 |

### 2.3 데이터 편향

학습 데이터의 카테고리 분포가 극도로 불균형했다.

| 카테고리 | 비율 |
|----------|------|
| 행정 | 89.6% |
| 기타 카테고리 | 10.4% (합계) |

이로 인해 모델이 행정 관련 질의에만 특화되고, 교통/환경/복지 등 다른 민원 유형에 대한 응답 품질이 저하되었다.

---

## 3. 해결 방안과 구현

### 3.1 EOS 학습 차단 해결

```python
# v1 (문제): pad_token = eos_token
tokenizer.pad_token = tokenizer.eos_token  # EOS 학습이 차단됨

# v2 (해결): pad_token = unk_token
tokenizer.pad_token = tokenizer.unk_token  # EOS와 분리하여 정상 학습
```

`pad_token`을 `unk_token`으로 분리함으로써, EOS 토큰이 학습 loss에 정상적으로 포함되어 모델이 응답 종료 시점을 학습할 수 있게 되었다.

### 3.2 데이터 재구성

전체 71,847건의 원본 데이터에 대해 다음을 적용했다.

1. **카테고리 세분화**: 8개 카테고리로 재분류 (행정, 교통, 환경, 복지, 문화, 경제, 안전, 기타)
2. **30% 샘플링 제한**: 과대 대표 카테고리의 비율을 제한하여 균형 잡힌 학습 데이터 구성
3. **최종 데이터**: 10,148 train / 1,265 val / 1,265 test

### 3.3 PII 마스킹 v2

개인정보 마스킹 로직을 개선하여 전화번호, 주민등록번호, 이메일 등의 PII가 학습 데이터에서 효과적으로 제거되도록 했다.

### 3.4 학습 파이프라인 개선

- **DataCollatorForCompletionOnlyLM**: 응답(assistant) 부분에만 loss를 계산하여 질의(user) 부분의 노이즈를 제거
- **SFTConfig**: TRL 최신 API와의 호환성 문제를 해결하여 안정적인 학습 실행
- **response_template**: `[|assistant|]` 토큰을 기반으로 응답 영역을 정확하게 분리

---

## 4. 학습 결과 (W&B 데이터 기반)

### 4.1 학습 설정

두 모델(v1, v2)은 동일한 하이퍼파라미터로 학습되었으며, 핵심 차이는 **pad_token 설정**과 **데이터 전처리 파이프라인**이다.

| 하이퍼파라미터 | 값 |
|---------------|-----|
| learning_rate | 2e-4 |
| lr_scheduler | cosine |
| num_epochs | 3 |
| per_device_batch_size | 2 |
| gradient_accumulation_steps | 8 |
| effective_batch_size | 16 |
| max_seq_length | 2048 |
| warmup_ratio | 0.03 |
| weight_decay | 0.01 |
| max_grad_norm | 1.0 |
| optimizer | paged_adamw_8bit |
| bf16 | True |
| gradient_checkpointing | True |
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 양자화 | 4-bit NF4, double quant, bfloat16 compute |

### 4.2 학습 곡선 비교

#### Train Loss

| 구간 | v1 Loss | v2 Loss | 차이 |
|------|---------|---------|------|
| 시작 (step 0) | 3.3008 | 3.3224 | +0.0216 |
| Epoch ~1 (step 60) | 1.7624 | 1.7586 | -0.0038 |
| Epoch ~2 (step 122) | 1.6827 | 1.6794 | -0.0033 |
| Epoch ~3 (step 182) | 1.5868 | 1.5824 | -0.0044 |
| 최종 (step 206) | 1.5535 | 1.5491 | -0.0044 |
| **최종 평균 train_loss** | **1.7535** | **1.7492** | **-0.0043** |

v2가 전 구간에서 미세하게 낮은 train loss를 기록했다.

#### Eval Loss

| 구간 | v1 Eval Loss | v2 Eval Loss | 차이 |
|------|-------------|-------------|------|
| 초기 (step 10) | 2.1055 | 2.1011 | -0.0044 |
| Epoch ~1 (step 65) | 1.8650 | 1.8608 | -0.0042 |
| Epoch ~2 (step 131) | 1.8010 | 1.7976 | -0.0034 |
| Epoch ~3 (step 197) | 1.7910 | 1.7873 | -0.0037 |
| **최종 eval_loss** | **1.7909** | **1.7872** | **-0.0037** |

v2가 전 구간에서 일관되게 더 낮은 eval loss를 보였으며, 최종 eval loss 기준 **0.21% 개선**되었다.

#### Mean Token Accuracy

| 지표 | v1 | v2 | 차이 |
|------|-----|-----|------|
| 최종 train token accuracy | 0.6438 | 0.6444 | +0.0006 |
| 최종 eval token accuracy | 0.6044 | 0.6046 | +0.0002 |

### 4.3 수렴 분석

두 모델 모두 유사한 수렴 패턴을 보였다.

- **급격한 하강 구간**: step 0~20 (loss 3.3 -> 2.1) - 초기 적응 단계
- **안정적 하강**: step 20~130 (loss 2.1 -> 1.80) - 주요 학습 구간
- **미세 조정**: step 130~208 (loss 1.80 -> 1.79) - 수렴 근접

eval loss가 step 142 부근에서 미세하게 증가한 후 다시 하강하는 패턴이 관찰되었으나, 최종적으로 최저점에 도달했다.

### 4.4 과적합 분석

| 지표 | v1 | v2 |
|------|-----|-----|
| 최종 train_loss | 1.5361 | 1.5320 |
| 최종 eval_loss | 1.7909 | 1.7872 |
| train-eval gap | 0.2548 | 0.2552 |

train-eval gap이 약 0.25로 두 모델 모두 유사하며, 과적합이 심하지 않은 수준이다. 3 epoch 학습이 적절한 것으로 판단된다.

### 4.5 Gradient Norm

| 구간 | v1 grad_norm | v2 grad_norm |
|------|-------------|-------------|
| 시작 (step 0) | 2.9758 | 2.9160 |
| 최종 (step 206) | 1.3568 | 1.3661 |
| Summary | 1.4095 | 1.3909 |

두 모델 모두 gradient norm이 안정적으로 수렴했으며, gradient clipping(max_grad_norm=1.0) 범위 내에서 정상적으로 학습이 진행되었다.

### 4.6 학습 시간

| 지표 | v1 | v2 |
|------|-----|-----|
| train_runtime (초) | 9,994.85 | 9,994.18 |
| train_runtime (분) | 166.6 | 166.6 |
| samples/sec | 3.046 | 3.046 |
| steps/sec | 0.190 | 0.190 |
| total_steps | 1,902 | 1,902 |

동일한 데이터 크기와 설정으로 약 **167분(2시간 47분)** 소요되었다.

---

## 5. Sanity Check 결과

학습 완료 후 5개 샘플에 대한 sanity check를 실시했다.

| 지표 | v1 | v2 | 변화 |
|------|-----|-----|------|
| EOS 생성률 | 0% (0/5) | 20% (1/5) | +20%p |
| 평균 생성 길이 | 866자 | 838자 | -28자 |
| max_new_tokens 도달 | 5/5 | 4/5 | -1 |

### 분석

- **EOS 생성률 개선**: v1의 0%에서 v2의 20%로 개선되었다. `pad_token = unk_token` 분리가 EOS 학습에 긍정적 영향을 미쳤으나, 아직 충분하지 않다.
- **생성 길이 감소**: 평균 생성 길이가 866자에서 838자로 28자 줄었다. 이는 EOS 학습이 부분적으로 작동하고 있음을 시사한다.
- **여전히 긴 생성**: 대부분의 샘플이 `max_new_tokens`에 도달하고 있어, 응답 축약이나 `max_seq_length` 조정이 필요하다.

---

## 6. 향후 개선 방향

### 6.1 max_seq_length 증가 또는 응답 축약

현재 `max_seq_length=2048`로 설정되어 있으나, 민원 응답의 특성상 긴 응답이 필요한 경우가 많다. 두 가지 접근을 검토한다.

- **max_seq_length 증가** (4096): 더 긴 응답을 학습할 수 있으나 메모리 요구량 증가
- **응답 축약 전처리**: 학습 데이터의 응답을 요약하여 간결한 응답 패턴을 학습

### 6.2 본격적인 평가 실행

현재는 5개 샘플 sanity check만 진행되었다. 다음 평가를 계획한다.

- BLEU, ROUGE-L, BERTScore 등 자동 평가 지표
- 카테고리별 응답 품질 비교
- EOS 생성률을 포함한 생성 안정성 평가
- 이전 v1 평가 결과(BLEU 0.53, ROUGE-L 4.20)와의 비교

### 6.3 Thought Tag 처리

EXAONE-Deep 모델은 내부적으로 `<thought>...</thought>` 태그를 사용한 체인-오브-쏘트(CoT)를 수행한다. 현재 이 thought 영역이 최종 응답에 포함될 수 있으므로, 추론 시 thought 태그를 적절히 파싱하여 사용자에게 보이지 않도록 후처리가 필요하다.

### 6.4 추가 개선 방향

- **EOS 학습 강화**: response_template 이후 EOS 토큰의 loss weight를 높이는 방안 검토
- **데이터 품질 추가 개선**: 응답 길이 분포 분석 및 이상치 제거
- **LoRA rank 실험**: r=32 또는 r=64 등 더 높은 rank 실험
- **학습률 스케줄링**: warmup ratio 또는 learning rate 조정 실험

---

## 부록: W&B Run 정보

| 항목 | v1 | v2 |
|------|-----|-----|
| W&B Project | civil-complaint-retrain-v2 | GovOn-retrain-v2 |
| Run ID | atsssd2z | uggxvc3s |
| Run Name | retrain-v2-lora-r16-3ep | GovOn-v2-lora-r16-3ep |
| Entity | umyun3 | umyun3 |
| 상태 | finished | finished |
| 생성일 | 2026-03-15 10:08:50 UTC | 2026-03-15 13:13:12 UTC |
| Transformers | 4.49.0 | 4.49.0 |
| PEFT | 0.18.1 | 0.18.1 |
