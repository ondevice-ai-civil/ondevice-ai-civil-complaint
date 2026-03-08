# M2 MVP 최종 결과 보고서 (Final Report)

## 1. 프로젝트 개요 및 MVP 목표
본 프로젝트는 **On-Device AI 민원 분석 및 처리 시스템** 구축을 목표로 하며, M2 MVP 단계에서는 **EXAONE-Deep-7.8B** 모델을 기반으로 민원 도메인 특화(파인튜닝) 및 경량화(AWQ 양자화) 가능성을 검증하였습니다.

## 2. 최종 성과 지표 (KPI) 달성 현황

| 지표 | 목표 | 실제값 (M2 MVP) | 상태 | 비고 |
|------|------|----------------|------|------|
| **Perplexity** | 최저 수렴 | **3.1957** | ✅ 달성 | 도메인 적응 성공 |
| **분류 정확도** | ≥ 85% | **2.00%** | ❌ 미달 | 파서 및 프롬프트 개선 필요 |
| **답변 생성 품질 (BLEU)** | ≥ 30 | **12.29** | ❌ 미달 | 참조 데이터 요약형 vs 모델 상세형 차이 |
| **답변 생성 품질 (ROUGE-L)** | ≥ 40 | **21.36** | ❌ 미달 | 의미론적 유사성 측정 도구(BERTScore) 도입 필요 |
| **추론 속도 (p50)** | < 2s | **9.291s** | ❌ 미달 | vLLM 도입 및 토큰 제한 필요 |
| **GPU VRAM** | < 8GB | **4.95 GB** | ✅ 달성 | AWQ 4-bit 양자화 효과 |
| **모델 크기** | < 5GB | **4.94 GB** | ✅ 달성 | 온디바이스 배포 가능 수준 확보 |

## 3. 진행 중 발생한 기술적 이슈 및 해결 방법

### 3.1 모델 로딩 및 라이브러리 호환성 (Critical)
- **이슈**: `transformers` 버전 업데이트(5.x)에 따른 EXAONE 아키텍처 호환성 파손 (`AttributeError: get_interface`).
- **해결**: `modeling_exaone.py` 및 `configuration_exaone.py` 내의 소스코드를 수동 패치(Monkey-patching)하여 `ALL_ATTENTION_FUNCTIONS` 및 `RopeParameters` 누락 문제를 해결.
- **교훈**: 특정 모델(특히 Dynamic Module을 쓰는 경우)은 라이브러리 버전 고정이 필수적임.

### 3.2 분류 정확도 0% 현상
- **이슈**: 모델이 답변을 생성할 때 `<thought>` 태그를 사용하여 추론 과정을 먼저 출력함에 따라 기존 단순 매칭 파서가 작동하지 않음.
- **해결**: 정규표현식을 이용해 `<thought>` 블록을 분리하고, `add_generation_prompt=True`를 포함한 Chat Template을 적용하여 출력을 정형화함.

### 3.3 높은 추론 레이턴시
- **이슈**: `max_new_tokens` 설정이 과도하게 높고, 모델의 '생각하는 과정'이 길어 실시간 응답 지연 발생.
- **해결**: 분류 작업 시 `max_tokens`를 10 토큰 미만으로 제한하고, `repetition_penalty=1.1`을 적용하여 무한 루프 방지.

## 4. 환경 설정 및 모델 정보

### 4.1 실행 환경
- **GPU**: NVIDIA L4 (24GB) / A100 (40GB/80GB)
- **Python**: 3.12
- **주요 패키지**: `transformers==4.46.1`, `peft`, `bitsandbytes`, `autoawq`, `bert-score`, `vllm`

### 4.2 모델 배포 정보 (HuggingFace)
- **LoRA Adapter**: [umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora)
- **Merged BF16**: [umyunsang/civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged)
- **AWQ 4-bit**: [umyunsang/civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq)

### 4.3 WandB 로그 기록
- **M2 파인튜닝 런**: [EXP-001-Baseline](https://wandb.ai/umyun3/huggingface/runs/kmx8rlvv)
- **M2 AWQ 평가 런**: [evaluation-20260307-0637](https://wandb.ai/umyun3/exaone-civil-complaint/runs/706jqzmk)
- **M3 최적화 시도 (Qwen Baseline)**: [m3-qwen-final-20260308-0342](https://wandb.ai/umyun3/exaone-civil-complaint/runs/4sms92k1)

## 5. 결론 및 M3 개선 로드맵
M2 MVP를 통해 모델의 도메인 적응 능력과 경량화 가능성을 확인했습니다. 미달된 지표(분류 정확도, 레이턴시)는 M3 단계에서 **vLLM 서버 통합**, **BERTScore 도입**, **전용 분류기(KR-ELECTRA) 추가**를 통해 개선할 예정입니다.
