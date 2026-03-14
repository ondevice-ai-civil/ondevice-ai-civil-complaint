# AWQ 양자화 분석 및 실험 기록

**작성일**: 2026-03-14
**프로젝트**: On-Device AI 민원 분석 및 처리 시스템

---

## 1. AWQ 양자화 개요

### 1.1 기법 소개

AWQ (Activation-aware Weight Quantization)는 MIT Han Lab에서 개발한 4비트 가중치 전용 양자화 기법이다.
MLSys 2024 Best Paper Award를 수상하였으며, 활성화 분포를 기반으로 중요 가중치 채널을 식별하여
양자화 오류를 최소화하는 것이 핵심 원리이다.

- **논문**: [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- **구현체**: [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

### 1.2 양자화 설정 (W4A16g128)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| bits | 4 | 4비트 가중치 양자화 |
| group_size | 128 | 128개 가중치마다 스케일 팩터 적용 |
| quant_method | awq | AWQ 양자화 방식 |
| version | gemm | GEMM 최적화 커널 사용 |
| zero_point | true | Zero-point 양자화 활성화 |
| modules_to_not_convert | lm_head | 출력 헤드는 전체 정밀도 유지 |

---

## 2. 양자화 파이프라인

### 2.1 전체 흐름

```
LGAI-EXAONE/EXAONE-Deep-7.8B (BF16, ~15.6GB)
    ↓ QLoRA 파인튜닝 (r=16, alpha=32)
umyunsang/civil-complaint-exaone-lora (~38MB 어댑터)
    ↓ merge_and_unload() BF16 병합
umyunsang/civil-complaint-exaone-merged (14.56 GB)
    ↓ AWQ W4A16g128 양자화 (512 calibration samples)
umyunsang/civil-complaint-exaone-awq (4.94 GB)
```

### 2.2 캘리브레이션 설정

- **캘리브레이션 데이터**: 민원 도메인 훈련 데이터 512 샘플
- **소요 시간**: ~20-40분 (A100 80GB 환경)

---

## 3. 양자화 결과

### 3.1 모델 크기 비교

| 모델 | 크기 | 대비 |
|------|------|------|
| BF16 원본 | ~15.6 GB | 기준 |
| BF16 병합 (LoRA merged) | 14.56 GB | -6.7% |
| AWQ 4-bit | 4.94 GB | **-68.3%** |

### 3.2 VRAM 사용량

| 환경 | VRAM | 비고 |
|------|------|------|
| BF16 추론 | ~20 GB | A100 필요 |
| AWQ 4-bit 추론 | 4.95 GB | RTX 3060 이상에서 실행 가능 |
| AWQ + vLLM | 4.17 GB | vLLM 최적화 적용 |

### 3.3 품질 영향

| 지표 | BF16 | AWQ 4-bit | 차이 |
|------|------|-----------|------|
| Perplexity | - | 3.1957 | 양호 (절대값 기준) |
| 언어 모델링 품질 | 기준 | 유지 | <1% 손실 추정 |

---

## 4. 발견된 문제점

### 4.1 Merged 모델 손상 (Critical)

M2 단계에서 생성한 `civil-complaint-exaone-merged` 모델에 손상이 발견되었다.

- **증상**: AWQ 모델에서 의미 없는 출력 생성
- **원인 분석**: transformers v5 환경에서 merge_and_unload() 수행 시 EXAONE 모델 코드 불일치
- **조치**: HuggingFace에서 merged 모델 삭제, AWQ 모델도 무효화 판정
- **재양자화 계획**: M3 평가 결과 확인 후 올바른 버전 조합으로 재수행 예정

### 4.2 올바른 양자화를 위한 필수 조건

AWQ 재양자화 시 반드시 지켜야 할 조건:

1. **transformers 4.44~4.49** 사용 (LoRA 학습 시점 버전)
2. **EXAONE revision `17b70148e344`** 고정 (학습 호환 코드)
3. merge_and_unload() 후 sanity check 필수 (정상 출력 확인)
4. 캘리브레이션 데이터는 학습 도메인과 동일한 민원 데이터 사용

---

## 5. vLLM 서빙 성능

### 5.1 AWQ + vLLM 조합 결과 (M3)

| 지표 | 측정값 | 목표 |
|------|--------|------|
| 평균 추론 속도 | 2.43s | < 2s |
| GPU VRAM | 4.17 GB | < 8GB |
| 분류 정확도 | 90.0% | ≥ 85% |

### 5.2 Marlin 커널 지원

vLLM은 AWQ 모델에 대해 Marlin 커널을 활용한 고성능 추론을 지원한다:
- GPTQ 대비 2.6배 속도 향상
- 표준 AWQ 대비 10.9배 속도 향상

---

## 6. 향후 계획

1. M3 표준 평가(920 샘플) 결과 확인
2. 올바른 버전 조합(transformers 4.44~4.49 + revision 17b70148e344)으로 모델 재병합
3. 재병합 모델 sanity check 후 AWQ 재양자화
4. vLLM 배포 및 속도 벤치마크 재측정

---

## 참고 자료

- [AWQ 논문](https://arxiv.org/abs/2306.00978)
- [AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ)
- [EXAONE-Deep-7.8B-AWQ (공식)](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ)
- [프로젝트 AWQ 모델](https://huggingface.co/umyunsang/civil-complaint-exaone-awq) — 현재 무효화
