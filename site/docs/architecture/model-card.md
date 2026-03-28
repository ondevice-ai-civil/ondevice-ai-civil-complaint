# 모델 카드

GovOn에서 사용하는 AI 모델의 상세 정보를 설명합니다. 베이스 모델 선정부터 QLoRA 파인튜닝, AWQ 양자화까지의 전체 모델 파이프라인을 다룹니다.

---

## 모델 개요

| 항목 | 값 |
|------|-----|
| **모델명** | GovOn-EXAONE-LoRA-v2 |
| **베이스 모델** | [LGAI-EXAONE/EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B) |
| **파인튜닝 방식** | QLoRA (4-bit NF4, double quantization) |
| **양자화** | AWQ W4A16g128 |
| **라이선스** | Apache 2.0 |
| **용도** | 지자체 민원 분류 및 답변 초안 생성 |
| **HuggingFace** | [umyunsang/GovOn-EXAONE-LoRA-v2](https://huggingface.co/umyunsang/GovOn-EXAONE-LoRA-v2) |
| **W&B Run** | [umyun3/GovOn-retrain-v2/uggxvc3s](https://wandb.ai/umyun3/GovOn-retrain-v2/runs/uggxvc3s) |

---

## 베이스 모델: EXAONE-Deep-7.8B

LG AI Research가 개발한 한국어 특화 대규모 언어 모델입니다.

| 항목 | 설명 |
|------|------|
| **파라미터** | 7.8B |
| **개발사** | LG AI Research |
| **라이선스** | Apache 2.0 |
| **특징** | 한국어 사전학습 데이터 대규모 포함, `<thought>` 태그 기반 추론 체인(CoT) 내장 |
| **선정 근거** | 한국어 민원 도메인 최적, 폐쇄망 배포 무제약, 양자화 파이프라인 검증 완료 |

EXAONE 모델을 선정한 상세 근거는 [ADR-001](adr/index.md#adr-001-exaone-deep-78b-모델-선정)을 참고하세요.

---

## QLoRA 파인튜닝

### 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| 파인튜닝 방식 | QLoRA (4-bit NF4) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 양자화 | 4-bit NF4, double quantization, bfloat16 compute |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Epochs | 3 |
| Batch size (per device) | 2 |
| Gradient accumulation | 8 |
| **Effective batch size** | **16** |
| Max sequence length | 2048 |
| Max grad norm | 1.0 |
| 정밀도 | bf16 |
| Gradient checkpointing | True |

### v2 주요 개선사항

v1 대비 다음 사항을 개선했습니다.

| 개선 항목 | 내용 |
|-----------|------|
| EOS 토큰 학습 정상화 | `pad_token`을 `unk_token`으로 분리하여 EOS 학습 차단 문제 해결 (0% -> 20%) |
| 데이터 균형화 | 카테고리별 30% 샘플링 제한으로 편향(행정 89.6%) 해소 |
| PII 마스킹 강화 | 개인정보 마스킹 로직 v2 적용 |
| 학습 파이프라인 안정화 | SFTConfig + DataCollatorForCompletionOnlyLM 적용 |

---

## 학습 데이터

한국 지방자치단체 민원 데이터를 기반으로 구성했습니다. 71,847건의 원본 데이터에서 카테고리 세분화 및 품질 필터링을 거쳤습니다.

| 분할 | 샘플 수 |
|------|---------|
| Train | 10,148 |
| Validation | 1,265 |
| Test | 1,265 |
| **합계** | **12,678** |

### 카테고리 분포

행정, 교통, 환경, 복지, 문화, 경제, 안전, 기타 (총 8개 카테고리)

### 데이터 전처리

1. 71,847건의 원본 데이터에서 카테고리 세분화 및 품질 필터링
2. PII 마스킹 v2 적용 (전화번호, 주민등록번호, 이메일 등)
3. Chat template 형식으로 변환 (system / user / assistant)
4. DataCollatorForCompletionOnlyLM으로 assistant 응답 부분에만 loss 적용

---

## 학습 결과

### 학습 곡선 요약

| 지표 | 값 |
|------|-----|
| 초기 train loss | 3.3224 |
| 최종 train loss | 1.5320 |
| 최종 eval loss | 1.7872 |
| 최종 train token accuracy | 0.6444 |
| 최종 eval token accuracy | 0.6046 |
| Train-Eval gap | 0.2552 |
| Total steps | 1,902 |
| 학습 시간 | 약 167분 |

학습은 3 epoch 동안 안정적으로 수렴했으며, train-eval gap이 0.25 수준으로 과적합이 심하지 않습니다.

### v1 대비 개선

| 지표 | v1 | v2 | 변화 |
|------|-----|-----|------|
| eval_loss | 1.7909 | 1.7872 | -0.0037 (-0.21%) |
| eval token accuracy | 0.6044 | 0.6046 | +0.0002 |
| train_loss (avg) | 1.7535 | 1.7492 | -0.0043 |

---

## AWQ 양자화

파인튜닝된 모델을 프로덕션 서빙에 적합한 크기로 양자화합니다.

### 양자화 과정

```mermaid
graph LR
    A[EXAONE-Deep-7.8B<br/>베이스 모델] --> B[QLoRA 파인튜닝<br/>LoRA 어댑터]
    B --> C[LoRA 병합<br/>BF16 전체 모델]
    C --> D[AWQ 양자화<br/>W4A16g128]
    D --> E[프로덕션 모델<br/>4.94GB]
```

### 양자화 설정

| 설정 | 값 | 설명 |
|------|----|------|
| `w_bit` | 4 | 4비트 가중치 양자화 |
| `q_group_size` | 128 | 128개 가중치를 하나의 양자화 그룹으로 묶음 |
| `zero_point` | True | 비대칭 양자화로 정밀도 향상 |
| `version` | GEMM | vLLM 호환 GEMM 커널 사용 |
| 캘리브레이션 데이터 | 512샘플 | 민원 학습 데이터에서 추출 (도메인 특화) |

### 양자화 결과

| 항목 | 값 |
|------|-----|
| 양자화 전 모델 크기 | 15.6GB (BF16) |
| 양자화 후 모델 크기 | 4.94GB (AWQ INT4) |
| **크기 절감율** | **68.3%** |
| 서빙 GPU VRAM 사용 | 약 4~5GB (16GB GPU에서 KV 캐시 여유 확보) |

양자화 방식을 선정한 상세 근거는 [ADR-002](adr/index.md#adr-002-awq-w4a16g128-양자화-방식-선정)를 참고하세요.

---

## 추론 성능

### 핵심 지표

| 지표 | 값 |
|------|-----|
| 민원 분류 정확도 | 90% |
| BERTScore | 46.05 |
| 추론 응답 시간 | 2.43초 |
| EOS 생성률 | 20% (v1 대비 개선, 추가 개선 진행 중) |

### v1 자동 평가 지표 (참고)

| 지표 | v1 값 |
|------|-------|
| BLEU | 0.53 |
| ROUGE-L | 4.20 |
| length_ratio | 0.63 |

!!! note
    v2에 대한 본격적인 자동 평가(BLEU, ROUGE-L, BERTScore)는 M3 마일스톤에서 진행 예정입니다.

---

## 추론 설정

프로덕션 환경에서의 vLLM 추론 설정입니다.

| 항목 | 값 | 근거 |
|------|----|------|
| `gpu_memory_utilization` | 0.8 | 16GB GPU 기준 KV 캐시 여유 확보 |
| `max_model_len` | 8192 | 민원 텍스트 + RAG 컨텍스트 + 답변 생성 |
| `trust_remote_code` | True | EXAONE 커스텀 모델 코드 로드 |
| `enforce_eager` | True | 패치된 모델 안정성 확보 |
| `dtype` | float16 | AWQ 모델 연산 정밀도 |
| `repetition_penalty` | 1.1 | EXAONE 안정성을 위한 반복 페널티 |

---

## 학습 인프라

| 항목 | 내용 |
|------|------|
| GPU | NVIDIA A100 40GB (Google Colab) |
| 학습 시간 | 약 167분 (2시간 47분) |
| 학습 프레임워크 | TRL 0.18.x + PEFT 0.18.1 + Transformers 4.49.0 |
| 양자화 라이브러리 | BitsAndBytes (학습), AutoAWQ (프로덕션 양자화) |
| 실험 추적 | Weights & Biases |

---

## 제한사항

1. **EOS 생성 불안정**: EOS 생성률이 20%로, 대부분의 응답이 `max_new_tokens`에 도달할 때까지 생성을 계속합니다. `max_new_tokens`를 적절히 설정하고 후처리로 응답을 정리해야 합니다.

2. **Thought 태그 포함**: EXAONE-Deep 모델의 특성상 `<thought>...</thought>` 태그가 응답에 포함될 수 있습니다. 추론 서버(`api_server.py`)에서 `_strip_thought_blocks()`로 자동 제거합니다.

3. **응답 길이**: 응답이 참조 답변 대비 짧은 경향이 있습니다 (v1 기준 length_ratio 0.63). 중요한 정보가 누락될 수 있으므로 답변 품질 검수가 필요합니다.

4. **카테고리 범위**: 8개 카테고리(행정, 교통, 환경, 복지, 문화, 경제, 안전, 기타)에 대해 학습되었으며, 이 범위를 벗어나는 질의에 대해서는 답변 품질이 보장되지 않습니다.

5. **법적/규정 정확성**: AI가 생성한 답변은 참고용이며, 법적 효력이 있는 공식 답변으로 사용할 수 없습니다. 실제 업무에서는 반드시 담당 공무원의 검토가 필요합니다.

6. **최대 시퀀스 길이**: `max_seq_length=2048`로 학습되었으므로, 이를 초과하는 긴 입력은 잘릴 수 있습니다. 추론 시 `max_model_len=8192`로 설정하지만, 학습 시 본 적 없는 긴 시퀀스에 대한 품질은 보장되지 않습니다.

---

## 인용

```bibtex
@misc{govon-exaone-lora-v2,
  title={GovOn-EXAONE-LoRA-v2: QLoRA Fine-tuned EXAONE-Deep-7.8B for Korean Civil Complaint Assistance},
  author={GovOn Team},
  year={2026},
  url={https://huggingface.co/umyunsang/GovOn-EXAONE-LoRA-v2}
}
```

---

## 관련 문서

- [시스템 구성도](overview.md) -- 전체 아키텍처 개요
- [API 명세](api.md) -- 추론 서버 REST API 레퍼런스
- [ADR-001: EXAONE 모델 선정](adr/index.md#adr-001-exaone-deep-78b-모델-선정) -- 모델 선정 근거
- [ADR-002: AWQ 양자화](adr/index.md#adr-002-awq-w4a16g128-양자화-방식-선정) -- 양자화 방식 선정 근거
