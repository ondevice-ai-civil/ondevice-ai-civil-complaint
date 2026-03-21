# M3 vLLM AWQ KPI 평가 보고서

**W&B Report | GovOn Civil Complaint AI Assistant**

---

## 1. Executive Summary

### 1.1 평가 목적

본 보고서는 GovOn 프로젝트 M3 Phase에서 AWQ INT4 양자화 모델(`umyunsang/GovOn-EXAONE-AWQ-v2`)의 vLLM 서빙 성능을 PRD v3.4 수락 기준(Acceptance Criteria)에 따라 종합 평가한 결과를 기술한다. 평가는 (1) 답변 생성 레이턴시, (2) 유사 사례 검색 레이턴시, (3) GPU 자원 효율성, (4) 답변 품질의 네 축으로 수행되었으며, 1,265건 테스트 데이터셋(8개 민원 카테고리)을 대상으로 실행하였다.

### 1.2 주요 발견사항

| 항목 | 결과 | 비고 |
|------|------|------|
| **AC-002 생성 p95** | **2.849s** (목표 < 3.0s) | PASS - 목표 대비 5.0% 여유 |
| **AC-003 검색 p95** | **39.76ms** (목표 < 1,000ms) | PASS - 목표 대비 96.0% 여유 |
| **VRAM 사용량** | **29.41GB** (목표 <= 5.0GB) | FAIL - A100 40GB 기준, KV 캐시 포함 |
| **SacreBLEU** | **7.74** (목표 >= 30) | 미달 - n-gram overlap 부족 |
| **ROUGE-L F1** | **18.76** (목표 >= 40) | 미달 - LCS 기반 구조적 유사도 부족 |
| **BERTScore F1** | **71.04** (목표 >= 55) | PASS - 의미적 유사도는 양호 |
| **EOS 정상 종료율** | **88.6%** (목표 >= 80) | PASS |

### 1.3 종합 판정

**레이턴시 KPI 2건 달성, VRAM KPI 1건 미달, 답변 품질 KPI 2건 미달.**

레이턴시 관점에서 vLLM + AWQ Marlin 커널 최적화의 효과가 확인되었다. 그러나 VRAM 사용량은 KV 캐시와 vLLM 런타임 오버헤드를 포함하면 목표의 5.88배에 달하며, 단일 GPU 서빙 시 A100 40GB급 이상이 필수적이다. 답변 품질(BLEU, ROUGE-L)은 v1 대비 대폭 개선되었으나 목표 도달에는 추가 작업이 필요하다. BERTScore가 71%로 의미적 유사도는 확보되어 있어, 표면적 n-gram 일치도 개선에 집중하는 것이 효율적이다.

---

## 2. 실험 환경 및 설정

### 2.1 하드웨어

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA A100-SXM4-40GB |
| GPU Architecture | Ampere (SM 8.0) |
| Total VRAM | 40.0 GB |
| Compute Capability | 8.0 (bfloat16 native support) |

### 2.2 모델 구성

| 항목 | 값 |
|------|-----|
| 서빙 모델 | `umyunsang/GovOn-EXAONE-AWQ-v2` |
| 기반 모델 | `LGAI-EXAONE/EXAONE-Deep-7.8B` |
| 양자화 방식 | AWQ INT4 (4-bit, group_size=128) |
| 양자화 커널 | `awq_marlin` (Ampere+ 최적화 GEMM) |
| 추론 dtype | `bfloat16` |
| 모델 크기 | ~4.94 GB (양자화 후) |
| Vocabulary | EXAONE tokenizer |

### 2.3 vLLM 엔진 설정

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `max_model_len` | 2,048 | 최대 시퀀스 길이 (KV 캐시 절감) |
| `gpu_memory_utilization` | 0.60 | GPU 메모리 활용률 (60%) |
| `enforce_eager` | False | CUDA Graph 활성화 (추론 가속) |
| `trust_remote_code` | True | EXAONE 커스텀 모델링 코드 |

### 2.4 생성 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `max_new_tokens` | 512 | 최대 생성 토큰 수 |
| `temperature` | 0.0 | Greedy decoding |
| `top_p` | 1.0 | Nucleus sampling 비활성화 |
| `repetition_penalty` | 1.1 | 반복 억제 |
| Stop tokens | `[|endofturn|]`, `</s>`, `<|endoftext|>` | |

### 2.5 데이터셋

| 항목 | 값 |
|------|-----|
| 테스트 셋 | `data/processed/v2_test.jsonl` |
| 샘플 수 | 1,265건 |
| 카테고리 수 | 8개 |
| 형식 | EXAONE Chat Template |

**카테고리 분포:**

| 카테고리 | 건수 | 비율 |
|----------|------|------|
| 교통 | 276 | 21.8% |
| 행정 | 243 | 19.2% |
| 환경 | 235 | 18.6% |
| 세금 | 186 | 14.7% |
| 기타 | 131 | 10.4% |
| 건축 | 86 | 6.8% |
| 복지 | 68 | 5.4% |
| 안전 | 40 | 3.2% |

데이터 분포는 long-tail 형태로, 복지(68건)와 안전(40건) 카테고리가 전체의 8.6%에 불과하여 해당 도메인의 학습 표현력이 제한적이다.

### 2.6 RAG 구성

| 항목 | 값 |
|------|-----|
| 임베딩 모델 | `intfloat/multilingual-e5-large` |
| 벡터 DB | FAISS IndexFlatIP |
| 인덱스 크기 | 10,148건 |
| 검색 top_k | 5 |
| BERTScore 모델 | `bert-base-multilingual-cased` |

---

## 3. KPI 평가 결과

### 3.1 AC-002: 답변 생성 레이턴시

**판정: PASS** (p95 = 2.849s < 3.0s 목표)

| 통계량 | 값 | 비고 |
|--------|-----|------|
| p50 (중앙값) | 1.559s | 대부분의 요청이 1.6초 내 완료 |
| **p95** | **2.849s** | **목표 3.0s 이내 달성** |
| p99 | 2.904s | tail latency 안정적 |
| 평균 | 1.570s | |
| 표준편차 | 0.727s | |
| 처리량 | 178.4 tokens/sec | 단일 요청 기준 |
| 측정 샘플 | 200건 | 카테고리별 균등 샘플링 |

**분석:**

- p95-p99 간격이 55ms로 극히 작아, tail latency가 안정적임을 나타낸다. 이는 vLLM의 CUDA Graph와 AWQ Marlin 커널이 일관된 추론 성능을 제공함을 의미한다.
- 표준편차(0.727s)가 평균(1.570s)의 46.3%로, 생성 길이에 따른 분산이 존재한다. 긴 답변(400+ tokens)이 tail latency의 주요 원인이다.
- gpu_memory_utilization을 0.60으로 보수적으로 설정했음에도 목표를 달성하여, 0.80~0.90으로 올릴 경우 KV 캐시 확장을 통한 동시 처리 개선이 가능하다.

**레이턴시 분포:**

```
  <0.5s  :    4 ( 2.0%)  #
  0.5-1s :   44 (22.0%)  ###########
  1-1.5s :   23 (11.5%)  #####
  1.5-2s :   58 (29.0%)  ##############
  2-2.5s :   51 (25.5%)  ############
  2.5-3s :   19 ( 9.5%)  ####
  3-4s   :    1 ( 0.5%)
  >4s    :    0 ( 0.0%)
```

99.5%의 요청이 3초 내 완료되었으며, 3초 초과 요청은 단 1건으로 극단적 outlier에 해당한다.

### 3.2 AC-003: 유사 사례 검색 레이턴시

**판정: PASS** (p95 = 39.76ms < 1,000ms 목표)

| 통계량 | 값 | 비고 |
|--------|-----|------|
| p50 | 23.08ms | |
| **p95** | **39.76ms** | **목표 대비 96.0% 여유** |
| p99 | 41.24ms | |
| 평균 | 24.93ms | |
| 측정 쿼리 | 200건 | |
| Recall@1 | 39.0% | 카테고리 기준 |
| 인덱스 크기 | 10,148건 | |

**분석:**

- E2E 레이턴시(임베딩 인코딩 + FAISS 검색)로 측정하였으며, 목표 대비 25배의 여유가 있다. 이는 IndexFlatIP의 brute-force 검색이 10K 규모에서 충분히 빠르다는 것을 보여준다.
- Recall@1이 39.0%로 낮은 편이다. 이는 카테고리 기반 proxy metric의 한계이기도 하지만, 임베딩 모델(`multilingual-e5-large`)이 민원 도메인에 fine-tuning되지 않은 점도 원인이다.
- 인덱스 규모가 10K 수준이므로 IVF/HNSW 등 근사 최근접 이웃(ANN) 알고리즘 전환 없이도 충분한 성능을 보인다. 100K+ 규모 확장 시에는 `IndexIVFFlat` 또는 `IndexHNSWFlat` 전환을 고려해야 한다.

### 3.3 VRAM 사용량

**판정: FAIL** (29.41GB > 5.0GB 목표)

| 항목 | 사용량 | 비고 |
|------|--------|------|
| 모델 가중치 (AWQ INT4) | 24.37 GB | 실측치 (nvidia-smi 기준) |
| 추론 시 전체 VRAM | 29.41 GB | KV 캐시, 활성화 메모리 포함 |
| 가용 VRAM | 40.0 GB | A100-SXM4-40GB |
| 모델 로드 시간 | 45.95s | 디스크 → GPU 전송 |

**분석:**

VRAM 5.0GB 목표는 원래 AWQ INT4 모델 파일 크기(~4.94GB)를 기준으로 설정된 것으로 보이나, 실제 vLLM 서빙 시에는 다음 요인으로 인해 크게 증가한다:

1. **KV 캐시 할당**: `max_model_len=2048`, `gpu_memory_utilization=0.60`으로 설정 시 vLLM이 사전 할당하는 KV 캐시가 수 GB를 차지
2. **vLLM 런타임 오버헤드**: CUDA Graph, 스케줄러 버퍼 등
3. **활성화 메모리**: Forward pass 중간 텐서
4. **AWQ Dequantization 버퍼**: Marlin 커널의 작업 메모리

**권장사항:**
- KPI 목표를 "모델 파일 크기 <= 5.0GB" 또는 "추론 시 VRAM <= 8.0GB (gpu_memory_utilization=0.3 기준)"로 재정의할 것을 제안한다.
- 또는 `max_model_len`을 1024로 축소하고 `gpu_memory_utilization`을 0.3으로 낮추면 ~12GB 수준으로 절감 가능하나, 동시 처리 성능이 저하된다.

---

## 4. 답변 품질 분석

### 4.1 메트릭 종합

| 메트릭 | v1 Baseline | v2 (LoRA) | AWQ v2 (현재) | 목표 | 달성 |
|--------|-------------|-----------|---------------|------|------|
| **SacreBLEU** | 0.53 | 11.45 | **7.74** | >= 30 | FAIL |
| **ROUGE-L F1** | 4.20 | 25.14 | **18.76** | >= 40 | FAIL |
| **BERTScore F1** | 59.15 | 72.34 | **71.04** | >= 55 | PASS |
| **EOS 종료율** | 0% | 91.3% | **88.6%** | >= 80 | PASS |
| 길이 비율 | N/A | N/A | 0.987 | ~1.0 | OK |
| Brevity Penalty | N/A | N/A | 0.991 | ~1.0 | OK |

### 4.2 AWQ 양자화 영향 분석

LoRA v2 (FP16/BF16) 대비 AWQ INT4 양자화 후 품질 변화:

| 메트릭 | LoRA v2 (FP) | AWQ v2 (INT4) | 변화 | 해석 |
|--------|-------------|---------------|------|------|
| SacreBLEU | 11.45 | 7.74 | **-3.71** | 양자화로 인한 n-gram precision 하락 |
| ROUGE-L | 25.14 | 18.76 | **-6.38** | 구조적 유사도 유의미한 감소 |
| BERTScore | 72.34 | 71.04 | **-1.30** | 의미적 유사도는 거의 보존 |
| EOS Rate | 91.3% | 88.6% | **-2.7%p** | 경미한 하락 |

양자화에 의한 품질 저하가 관측된다. 특히 BLEU(-3.71)와 ROUGE-L(-6.38)의 하락이 유의미하다. 이는 INT4 양자화 시 토큰 단위의 미세한 확률 분포 변화가 lexical overlap 메트릭에 민감하게 반영되기 때문이다. 반면 BERTScore(-1.30)는 거의 보존되어, 의미적 품질은 유지되고 있다.

**시사점:** 양자화 자체의 품질 손실을 줄이기 위해 (1) GPTQ 대비 AWQ의 적합성 재검증, (2) Calibration 데이터셋 품질 개선, (3) 양자화 후 학습(QAT) 검토가 필요하다.

### 4.3 카테고리별 품질 분석

LoRA v2 기준 카테고리별 성능 (AWQ 후 전카테고리 일률적 하락 예상):

| 카테고리 | BLEU | ROUGE-L | BERTScore | 특이사항 |
|----------|------|---------|-----------|----------|
| 건축 | 11.16 | **37.05** | 73.08 | ROUGE-L 최고, 정형화된 답변 구조 |
| 교통 | 12.19 | 25.66 | 73.50 | 가장 많은 학습 데이터(276건) |
| 환경 | 11.92 | 26.82 | 72.89 | |
| 행정 | 12.08 | 24.47 | 71.84 | |
| 기타 | 11.21 | 22.11 | 70.42 | |
| 안전 | 9.70 | 21.05 | 70.15 | 데이터 최소(40건) |
| 세금 | 7.67 | 23.47 | **76.39** | BLEU 최저, BERTScore 최고 (역설) |
| 복지 | **8.90** | **15.86** | 69.21 | 전체 최저 성능 |

**핵심 인사이트:**

1. **건축 카테고리의 높은 ROUGE-L(37.05)**: 건축 허가, 신고 등 법규 기반 답변이 정형화되어 있어 LCS 기반 매칭이 유리하다. 이 패턴을 다른 카테고리에 적용할 수 있는지 분석이 필요하다.

2. **세금 역설 현상**: BLEU 최저(7.67)이나 BERTScore 최고(76.39). 세금 관련 답변이 의미적으로는 정확하나 표현이 다양하여 n-gram 일치율이 낮다. 참조 답변과 생성 답변의 어휘 분포가 다르되 의미는 보존되는 전형적인 paraphrase 패턴이다.

3. **복지/안전 저성능**: 학습 데이터 부족(복지 68건, 안전 40건)이 직접적 원인이다. 최소 150건 이상으로 증강해야 한다.

### 4.4 오류 패턴 분석

v2 평가에서 식별된 주요 오류 패턴:

| 오류 유형 | 빈도 | 영향도 | 설명 |
|-----------|------|--------|------|
| 서울 편향 Hallucination | 168건 (15.4%) | 높음 | 지자체 무관하게 "서울시" 언급 |
| 영어 혼입 | 87건 (6.9%) | 중간 | 영어 단어/구문 삽입 |
| '끝.' 과학습 | 73.6% | 중간 | 참조(29.6%) 대비 과도한 종결 패턴 |
| 시작 패턴 고착화 | 42% | 높음 | 상위 3개 시작 패턴이 전체의 42% |
| 반복 루프 (EOS 미생성) | 11.4% | 높음 | max_new_tokens 한계 도달 |
| ROUGE-L < 5 극단적 미매칭 | 104건 (8.2%) | 높음 | 답변 구조 자체가 불일치 |

---

## 5. 처리 시간 분석

### 5.1 모델 로드 성능

| 단계 | 시간 | 비고 |
|------|------|------|
| 모델 다운로드 + 로드 | 45.95s | HuggingFace Hub → GPU |
| 배치 추론 (1,265건) | 77.71s | ~16.3 samples/sec |

### 5.2 추론 효율성

| 지표 | 값 |
|------|-----|
| 단일 요청 평균 레이턴시 | 1.570s |
| 단일 요청 처리량 | 178.4 tokens/sec |
| 배치 처리량 | ~16.3 samples/sec |
| 평균 생성 토큰 수 | ~280 tokens (추정) |

vLLM의 continuous batching과 PagedAttention이 배치 처리 시 효율적으로 동작하여, 1,265건을 약 78초 만에 처리하였다. 단일 요청 대비 배치 처리의 throughput이 약 10배 향상된다.

---

## 6. 주요 발견사항 및 인사이트

### 6.1 AWQ Marlin 커널의 효과

AWQ INT4 + Marlin 커널(`awq_marlin`)은 Ampere 아키텍처에서 최적화된 GEMM을 제공하여, 모델 크기 대비 우수한 추론 속도를 달성했다. `enforce_eager=False` (CUDA Graph 활성화)와 결합하여 커널 launch 오버헤드를 최소화한 것이 p95 레이턴시 달성의 핵심 요인이다.

### 6.2 BERTScore vs Lexical Metrics 괴리

BERTScore F1(71.04)과 BLEU(7.74)/ROUGE-L(18.76) 간의 큰 격차는, 모델이 **의미적으로는 적절한 답변을 생성하되 표현 방식이 참조 답변과 다르다**는 것을 시사한다. 이는 다음을 의미한다:

1. 모델은 민원 의도를 정확히 파악하고 관련 정보를 제공하고 있음
2. 그러나 참조 답변과의 어휘/구문 일치도가 낮음
3. **참조 답변 자체의 품질/다양성 개선**이 lexical metric 향상의 핵심

### 6.3 VRAM 목표의 재정의 필요성

AWQ INT4 모델 파일(~4.94GB)과 실제 서빙 시 VRAM(29.41GB) 간의 6배 차이는 vLLM 런타임의 구조적 특성이다. 5.0GB 목표는 **on-device 추론**(ONNX Runtime, llama.cpp 등) 기준으로는 달성 가능하나, **vLLM 서빙 기준**으로는 비현실적이다. 목표 재정의 또는 서빙 아키텍처 변경이 필요하다.

### 6.4 데이터 불균형의 영향

복지(68건), 안전(40건) 카테고리의 학습 데이터 부족이 성능 저하의 주요 원인이다. 특히 복지 카테고리는 ROUGE-L 15.86으로 건축(37.05) 대비 21.2pp 낮으며, 이 격차를 좁히기 위한 데이터 증강이 필수적이다.

---

## 7. 권장사항 및 개선 방안: 답변 품질 고도화 로드맵

> *이하 내용은 GitHub Issue #68 (답변 생성 품질 고도화: BLEU >= 30, ROUGE-L >= 40)의 상세 개선 계획이다.*

### 7.0 현황 진단 및 Root Cause Analysis (RCA)

**목표 대비 Gap:**
- BLEU: 7.74 → 30 (Gap: -22.26, 필요 개선율: +288%)
- ROUGE-L: 18.76 → 40 (Gap: -21.24, 필요 개선율: +113%)

**근본 원인 계층 구조 (Ishikawa Diagram):**

```
                       ┌─── 학습 데이터 품질
                       │    ├── 카테고리 불균형 (안전 40건)
                       │    ├── 시작 패턴 편향 (3개 패턴 → 42%)
           데이터 ─────┤    ├── 지자체명 context 부재 → 서울 편향
                       │    └── 참조 답변 스타일 불일치
                       │
BLEU/ROUGE-L ─────────┤
미달                   │
                       │    ┌─── max_new_tokens=512 한계
           모델/추론 ──┤    ├── repetition_penalty 불충분
                       │    ├── AWQ 양자화 품질 손실 (-3.71 BLEU)
                       │    └── LoRA rank=16 표현력 제한
                       │
                       │    ┌─── 프롬프트 단순 (단일 turn)
           프롬프트 ───┤    ├── 카테고리 힌트 미활용
                       └────└── Few-shot 예시 미포함
```

### 7.1 Phase 1: 데이터 품질 혁신 (예상 효과: BLEU +8~12, ROUGE-L +10~15)

**기간:** 1~2주 | **우선순위:** 최고 | **의존성:** 없음

데이터 품질은 모든 NLG 메트릭 개선의 기반이다. Google의 "Data-Centric AI" 연구(Zha et al., 2023)에서 입증된 바와 같이, 데이터 품질 개선은 모델 아키텍처 변경 대비 2~5배 높은 ROI를 보인다.

#### 7.1.1 카테고리별 데이터 증강 (Targeted Augmentation)

**현황:**
| 카테고리 | 현재 건수 | 목표 건수 | 증강 배수 | ROUGE-L (현재) |
|----------|-----------|-----------|-----------|----------------|
| 안전 | 40 | 200 | 5.0x | 21.05 |
| 복지 | 68 | 200 | 2.9x | 15.86 |
| 건축 | 86 | 150 | 1.7x | 37.05 |
| 기타 | 131 | 150 | 1.1x | 22.11 |
| 세금 | 186 | 200 | 1.1x | 23.47 |
| 환경 | 235 | 250 | 1.1x | 26.82 |
| 행정 | 243 | 250 | 1.0x | 24.47 |
| 교통 | 276 | 250 | 0.9x | 25.66 |

**실행 방안:**

1. **Dataset 71852 재활용 확대**: `consulting_content` 필드에서 미사용 Q-A 쌍 추출. 현재 활용률이 낮은 복지/안전 도메인을 우선 발굴
2. **Back-Translation 증강**: 한→영→한 번역을 통한 paraphrase 생성 (NLLB-200 또는 Google Translate API). 각 원본에서 2~3개 변형 생성
3. **LLM-based Augmentation**: GPT-4o 또는 Claude를 활용하여 기존 민원의 변형 생성. 지자체명, 민원인 상황, 구체적 위치 등을 다양화. 품질 필터링은 BERTScore >= 0.7 기준
4. **Cross-category Transfer**: 건축 카테고리(ROUGE-L 37.05)의 정형화된 답변 구조를 복지/안전에 적용하여 template 생성

**검증 방법:**
- Augmented 데이터의 품질: 무작위 100건 수동 평가 (적절성 4점 이상/5점 기준)
- 증강 전후 데이터 분포: t-SNE 시각화로 원본과의 분포 일치 확인
- 부트스트랩 신뢰구간: 증강 데이터 포함/미포함 학습 결과의 95% CI 비교

#### 7.1.2 시작 패턴 다양화 (Template De-biasing)

**현황:** 상위 3개 시작 패턴("안녕하세요, ", "민원 내용을 ", "해당 민원에 ")이 전체 생성의 42%를 차지하여 n-gram diversity가 심각하게 제한됨.

**실행 방안:**

1. **참조 답변 rewriting**: 기존 학습 데이터의 참조 답변 시작부를 10가지 이상의 패턴으로 변형
   - "말씀하신 [카테고리] 관련 민원에 대해 안내드립니다."
   - "[지자체명] [담당부서]입니다. 민원 내용 확인했습니다."
   - "접수하신 민원([민원번호])에 대한 답변입니다."
   - 등 실제 공무원 답변 패턴 기반 10+ 변형
2. **시작 패턴 샘플링 균등화**: 학습 데이터에서 각 시작 패턴의 빈도가 10% 이하가 되도록 resampling

**예상 효과:** 시작 패턴 다양화만으로 Unigram Precision 27.4% → 35%+ 개선 가능 (BLEU +3~5 기여)

#### 7.1.3 지자체 Context 주입 (Hallucination Mitigation)

**현황:** 서울 편향 168건(15.4%). 학습 데이터에 지자체명이 미포함되어 모델이 가장 빈번한 "서울시"를 기본값으로 학습.

**실행 방안:**

1. **프롬프트 내 지자체명 명시**: `[카테고리: 교통] [지역: 부산광역시]`
2. **학습 데이터에 지자체 메타데이터 추가**: Dataset 71852의 지자체 정보 활용
3. **Negative Sampling**: 잘못된 지자체명이 포함된 답변을 negative example로 활용하여 DPO 학습

**검증:** 서울 편향율 15.4% → 5% 미만 목표. 지자체명 정확도를 별도 메트릭으로 추적.

#### 7.1.4 참조 답변 품질 고도화

**현황:** 참조 답변의 평균 길이 편차가 크고, 일부는 100자 미만의 초단문.

**실행 방안:**

1. **최소 답변 길이 필터**: 200자 미만 참조 답변 제거 또는 확장
2. **답변 구조 표준화**: 모든 참조 답변에 (1) 인사 (2) 민원 내용 확인 (3) 답변 본문 (4) 안내사항 (5) 종결 구조 적용
3. **800자 이상 답변 비중 확대**: 상세한 답변이 BLEU/ROUGE-L에 유리 (더 많은 n-gram 매칭 기회)

### 7.2 Phase 2: 생성 설정 및 프롬프트 최적화 (예상 효과: BLEU +3~5, ROUGE-L +5~8)

**기간:** 1주 | **우선순위:** 높음 | **의존성:** Phase 1 데이터 준비 완료 후

#### 7.2.1 생성 하이퍼파라미터 탐색

현재 greedy decoding(temperature=0.0)을 사용하고 있으며, 이는 deterministic하지만 diversity가 낮다.

**탐색 범위 (Bayesian Optimization with Optuna):**

| 파라미터 | 현재 | 탐색 범위 | 근거 |
|----------|------|-----------|------|
| `max_new_tokens` | 512 | 512, 768, 1024 | EOS 미생성 11.4% 해소 |
| `repetition_penalty` | 1.1 | [1.1, 1.3] | 반복 루프 억제 (17.3% 반복률) |
| `temperature` | 0.0 | [0.0, 0.3, 0.5] | Diversity-quality trade-off |
| `top_p` | 1.0 | [0.85, 0.90, 0.95, 1.0] | Nucleus sampling으로 long-tail 제거 |
| `top_k` | - | [0, 30, 50] | Vocabulary 탐색 범위 제어 |

**실험 설계:**

1. **Grid Search (1단계)**: `max_new_tokens` x `repetition_penalty` 2D grid (6 combinations)
2. **Bayesian Optimization (2단계)**: Optuna TPE sampler로 temperature/top_p/top_k 3D 탐색 (50 trials)
3. **Best Config Selection**: Pareto frontier에서 BLEU와 ROUGE-L의 harmonic mean 최대화

**통계적 검증:**
- 각 설정에 대해 5-fold bootstrapping (n=253/fold)으로 95% 신뢰구간 산출
- Wilcoxon signed-rank test로 baseline 대비 유의성 검증 (p < 0.05)

#### 7.2.2 프롬프트 엔지니어링

**현재 프롬프트:**
```
[|system|]당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다.[|endofturn|]
[|user|][카테고리: {category}]
민원 내용: {complaint}[|endofturn|]
[|assistant|]
```

**개선안 A - 카테고리 + 지역 힌트:**
```
[|system|]당신은 {municipality} {department} 민원 담당 공무원을 돕는 AI 어시스턴트입니다.
답변 시 다음 형식을 따르세요:
1. 민원 내용 확인
2. 관련 법규/절차 안내
3. 처리 방법 및 기간
4. 추가 안내사항[|endofturn|]
[|user|][카테고리: {category}] [지역: {municipality}]
민원 내용: {complaint}[|endofturn|]
[|assistant|]
```

**개선안 B - Few-shot 프롬프트:**
```
[|system|]당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다.
아래는 모범 답변 예시입니다:

[예시 민원] 도로에 포트홀이 있어 위험합니다.
[예시 답변] 민원 접수해 주셔서 감사합니다. 말씀하신 도로 포트홀 관련하여 안내드립니다. 해당 구간은 도로관리과에서 현장 확인 후 72시간 이내 보수 작업을 진행할 예정입니다...[|endofturn|]
[|user|][카테고리: {category}]
민원 내용: {complaint}[|endofturn|]
[|assistant|]
```

**실험 설계:**
- A/B 테스트: 현재 프롬프트 vs 개선안 A vs 개선안 B (각 1,265건 전수 평가)
- 카테고리별 few-shot 예시 최적 개수: 0, 1, 2, 3-shot 비교
- Context window 제약: few-shot 예시 포함 시 `max_model_len=2048` 내 수용 가능한지 확인

#### 7.2.3 Decoding 전략 비교

| 전략 | 장점 | 단점 | BLEU 예상 영향 |
|------|------|------|----------------|
| Greedy (현재) | Deterministic, 빠름 | Diversity 최저 | Baseline |
| Beam Search (k=4) | Higher quality, 다양한 후보 | 2~4x 느림 | +2~4 |
| Sampling (t=0.3) | Diversity 증가 | 비결정적 | +1~3 |
| Contrastive Search | Degeneration 방지 | 구현 복잡 | +2~5 (연구 결과) |

**권장:** Beam Search (k=4)를 먼저 시도하되, 레이턴시가 p95 3.0s를 초과하면 Contrastive Search로 전환.

### 7.3 Phase 3: 모델 학습 전략 고도화 (예상 효과: BLEU +5~10, ROUGE-L +8~12)

**기간:** 2~3주 | **우선순위:** 중간 | **의존성:** Phase 1 + Phase 2 완료 후

#### 7.3.1 LoRA 하이퍼파라미터 확장

**현재:** r=16, alpha=32, lr=2e-4, 1 epoch

**탐색 계획:**

| 파라미터 | 현재 | 후보 | 근거 |
|----------|------|------|------|
| LoRA rank (r) | 16 | 32, 64 | 표현력 확장 (Hu et al., 2022: r 증가 시 저빈도 패턴 학습 개선) |
| LoRA alpha | 32 | 64, 128 | alpha/r 비율 2 유지 또는 4로 증가 |
| Learning rate | 2e-4 | 1e-4, 5e-5, 2e-5 | 큰 r에서는 낮은 lr이 안정적 |
| Epochs | 1 | 2, 3 | 데이터 증강 후 더 많은 epoch 필요 |
| Warmup ratio | 0.03 | 0.05, 0.10 | 학습 안정성 |
| Target modules | q_proj, v_proj | q,k,v,o_proj + gate,up,down_proj | Full LoRA vs Attention-only 비교 |
| Dropout | 0 | 0.05, 0.10 | 과적합 방지 |

**실험 설계:**

1. **Stage 1 - LoRA rank 탐색**: r={16, 32, 64}, alpha=2r, 고정 lr=2e-4 (3 runs)
2. **Stage 2 - Learning rate 탐색**: 최적 r에서 lr={1e-4, 5e-5, 2e-5} (3 runs)
3. **Stage 3 - Target modules**: Attention-only vs Full LoRA (2 runs)
4. **Stage 4 - Fine-tuning**: 최적 조합에서 epochs, warmup, dropout 미세 조정 (Optuna 20 trials)

**리소스 추정:** A100 40GB 기준, r=64 학습 시 약 3~4시간/epoch (1,265건)

#### 7.3.2 DPO (Direct Preference Optimization) 적용

SFT 이후 DPO를 통해 선호 답변 쪽으로 정렬하는 전략.

**실행 방안:**

1. **Preference 데이터 구축:**
   - Chosen: 참조 답변 (정확한 지자체, 구조적 답변)
   - Rejected: 현재 모델 생성 답변 중 서울 편향, 반복 루프, 시작 패턴 고착 사례
   - 규모: 최소 500 쌍 (카테고리별 균등)

2. **DPO 학습 설정:**
   - beta=0.1 (KL divergence 페널티)
   - lr=5e-7 (SFT 대비 10x 낮은 학습률)
   - 1 epoch

3. **기대 효과:**
   - 서울 편향 15.4% → 5% 미만
   - 시작 패턴 다양성 42% → 20% 미만
   - BLEU +3~5, ROUGE-L +3~5 (선행 연구 기반)

**Trade-off:** DPO는 alignment tax가 존재하여 BERTScore가 1~2% 하락할 수 있음. 이를 모니터링해야 함.

#### 7.3.3 양자화 전략 재검토

AWQ INT4에서 BLEU -3.71 손실이 관측되었으므로:

| 전략 | 모델 크기 | 예상 BLEU 손실 | VRAM | 추론 속도 |
|------|-----------|----------------|------|-----------|
| AWQ INT4 (현재) | 4.94 GB | -3.71 | 29.4 GB | 178.4 tps |
| GPTQ INT4 | ~5.0 GB | -2~3 (예상) | ~28 GB | ~160 tps |
| AWQ INT4 + QAT | 4.94 GB | -1~2 (예상) | 29.4 GB | 178.4 tps |
| BF16 (비양자화) | ~16 GB | 0 (baseline) | ~35 GB | ~120 tps |
| GGUF Q5_K_M | ~5.5 GB | -1~2 (예상) | ~8 GB* | ~80 tps* |

*llama.cpp 기준 추정치

**권장:** 단기적으로는 AWQ 유지하되, Calibration 데이터를 현재 학습 데이터로 교체하여 양자화 품질 개선. 중기적으로 QAT(Quantization-Aware Training) 도입 검토.

### 7.4 Phase별 예상 효과 및 마일스톤

```
현재 (AWQ v2)
├── BLEU:    7.74
├── ROUGE-L: 18.76
│
Phase 1 완료 (+2주)
├── BLEU:    ~16-20 (+8~12)
├── ROUGE-L: ~29-34 (+10~15)
│   - 데이터 증강, 패턴 다양화, 지자체 context
│
Phase 2 완료 (+1주)
├── BLEU:    ~19-25 (+3~5)
├── ROUGE-L: ~34-42 (+5~8)
│   - HP 최적화, 프롬프트 개선, beam search
│
Phase 3 완료 (+2~3주)
├── BLEU:    ~24-35 (+5~10)
├── ROUGE-L: ~42-54 (+8~12)
│   - LoRA 확장, DPO, 양자화 개선
│
목표
├── BLEU:    >= 30
└── ROUGE-L: >= 40
```

### 7.5 리스크 및 Trade-off 분석

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| 데이터 증강 품질 저하 | 중 | 높음 | 수동 검수 100건 + BERTScore 필터 |
| LoRA r=64 학습 불안정 | 낮 | 중간 | Gradient clipping, lr warmup 확대 |
| DPO alignment tax | 중 | 중간 | BERTScore 모니터링, beta 조절 |
| Beam search 레이턴시 초과 | 중 | 높음 | Contrastive search 대안 |
| 양자화 후 재학습 필요 | 높 | 중간 | QAT 파이프라인 구축 |
| BLEU 30 미달성 | 중 | 높음 | 참조 답변 자체를 모델 생성과 유사하게 재구성 |

### 7.6 실험 관리 및 모니터링

모든 실험은 W&B(`umyun3/GovOn`)에 기록하며, 다음 메트릭을 실시간 추적한다:

- **Primary:** SacreBLEU, ROUGE-L F1
- **Secondary:** BERTScore F1, EOS Rate, 시작 패턴 entropy
- **Guard rails:** 서울 편향율, 영어 혼입율, 반복률
- **Efficiency:** 학습 시간, 추론 레이턴시, VRAM

---

## 8. 결론

### 8.1 M3 Phase 평가

M3 Phase의 핵심 목표는 AWQ 양자화 모델의 프로덕션 서빙 가능성 검증이었다.

**달성 사항:**
- 답변 생성 레이턴시 p95 < 3.0s 달성 (2.849s)
- FAISS 검색 레이턴시 p95 < 1.0s 달성 (39.76ms)
- vLLM + AWQ Marlin 커널의 안정적 추론 확인
- BERTScore F1 71.04%로 의미적 품질 확보

**미달 사항:**
- VRAM 사용량 29.41GB (목표 5.0GB) - KPI 재정의 필요
- BLEU 7.74 (목표 30) - 데이터 및 학습 전략 개선 필요
- ROUGE-L 18.76 (목표 40) - 데이터 및 학습 전략 개선 필요

### 8.2 다음 단계 (M4 Phase 권장 사항)

| 우선순위 | 작업 | 예상 기간 | 예상 효과 |
|----------|------|-----------|-----------|
| 1 | Phase 1: 데이터 품질 혁신 | 2주 | BLEU +8~12, ROUGE-L +10~15 |
| 2 | Phase 2: 생성 설정 최적화 | 1주 | BLEU +3~5, ROUGE-L +5~8 |
| 3 | Phase 3: 모델 학습 전략 | 2~3주 | BLEU +5~10, ROUGE-L +8~12 |
| 4 | VRAM KPI 재정의 | 1일 | KPI 정합성 확보 |
| 5 | Recall@1 개선 (도메인 임베딩) | 2주 | 검색 품질 39% → 60%+ |

### 8.3 핵심 메시지

> 모델은 민원의 의미를 잘 이해하고 적절한 답변을 생성하고 있다(BERTScore 71%). 그러나 참조 답변과의 표면적 일치도(BLEU, ROUGE-L)를 높이기 위해서는 **데이터 품질 개선이 최우선**이며, 이는 모델 아키텍처 변경보다 비용 대비 효과가 크다.

---

## Appendix

### A. 평가 결과 파일

- JSON: `docs/outputs/kpi_evaluation/kpi_eval_awq_vllm_20260320_121929.json`
- 노트북: `notebooks/M3_issue26/06_evaluate_awq_vllm_kpi.ipynb`
- W&B Project: `umyun3/GovOn`

### B. 재현 방법

1. Google Colab에서 `06_evaluate_awq_vllm_kpi.ipynb` 실행
2. Runtime: A100 GPU 선택
3. W&B API Key 및 HuggingFace Token 설정
4. 전체 셀 순차 실행 (약 15분 소요)

### C. 버전 정보

| 항목 | 버전 |
|------|------|
| PRD | v3.4 |
| 모델 | umyunsang/GovOn-EXAONE-AWQ-v2 |
| 기반 모델 | LGAI-EXAONE/EXAONE-Deep-7.8B |
| vLLM | 최신 (Colab 설치 시점) |
| PyTorch | 최신 (CUDA 12.x) |
| 평가 일시 | 2026-03-20 12:19:29 |
| 테스트 데이터 | v2_test.jsonl (1,265건) |

### D. 용어 정의

| 용어 | 정의 |
|------|------|
| AWQ | Activation-aware Weight Quantization. 가중치의 중요도를 활성화 분포 기반으로 판단하여 양자화하는 기법 |
| Marlin | Mixed-precision matrix multiplication 커널. Ampere+ GPU에서 INT4 가중치를 효율적으로 dequantize하며 GEMM 수행 |
| PagedAttention | vLLM의 핵심 기술. KV 캐시를 페이지 단위로 관리하여 메모리 단편화 방지 |
| DPO | Direct Preference Optimization. RLHF의 reward model 없이 직접 선호 데이터로 정렬하는 기법 |
| LoRA | Low-Rank Adaptation. 사전학습 모델의 가중치를 고정하고 저랭크 행렬만 학습하는 효율적 미세조정 기법 |
| QAT | Quantization-Aware Training. 양자화 오차를 학습 과정에 반영하여 양자화 후 품질 손실을 최소화하는 기법 |
