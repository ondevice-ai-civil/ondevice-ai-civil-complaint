# Models Directory

이 디렉토리는 GovOn 서비스에서 사용하는 모델 가중치 및 LoRA 어댑터 관련 정보를 담고 있습니다.

모든 모델 가중치는 파일 크기 제한으로 인해 [Hugging Face Model Hub](https://huggingface.co/umyunsang)에 호스팅됩니다.

---

## 현재 아키텍처: Multi-LoRA 서빙

GovOn은 **단일 베이스 모델 + 다중 LoRA 어댑터** 구조로 운영됩니다.  
vLLM의 Multi-LoRA 기능을 활용하여 하나의 베이스 모델 위에 여러 어댑터를 동적으로 로드합니다.

### 베이스 모델

| 항목 | 값 |
|------|-----|
| **모델** | [LGAI-EXAONE/EXAONE-4.0-32B-AWQ](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-AWQ) |
| **타입** | AWQ 4-bit 양자화 |
| **VRAM 요구량** | ~20GB |
| **상태** | ✅ 운영 중 |

환경변수 설정:

```bash
MODEL_PATH=LGAI-EXAONE/EXAONE-4.0-32B-AWQ
# 오프라인(에어갭) 환경에서는 컨테이너 내부 경로 사용
# MODEL_PATH=/app/models/EXAONE-4.0-32B-AWQ
```

### LoRA 어댑터

| 어댑터 | 용도 | 상태 |
|--------|------|------|
| `civil-adapter` | 민원 초안 생성 | ✅ 운영 중 |
| `legal-adapter` | 법률 근거 조회 | 🔜 개발 중 |

어댑터 경로는 `ADAPTER_PATHS` 환경변수로 설정합니다:

```bash
# 콤마로 구분하여 여러 어댑터 지정
ADAPTER_PATHS=/app/models/adapters/civil-adapter,/app/models/adapters/legal-adapter
```

---

## 오프라인(에어갭) 환경 설정

에어갭 환경에서는 베이스 모델과 어댑터를 컨테이너 내부에 미리 배치합니다:

```
/app/models/
  EXAONE-4.0-32B-AWQ/      ← 베이스 모델 가중치
  adapters/
    civil-adapter/           ← 민원 LoRA 어댑터
    legal-adapter/           ← 법률 LoRA 어댑터 (예정)
```

`.env.airgap.example`을 참고하여 환경변수를 구성하세요.

---

## 폐기된 모델

아래 모델들은 더 이상 사용하지 않습니다. 코드베이스에서 참조를 제거하였으므로 사용하지 마세요.

| 모델 | 폐기 사유 |
|------|----------|
| ~~[umyunsang/civil-complaint-exaone-lora](https://huggingface.co/umyunsang/civil-complaint-exaone-lora)~~ | 잘못된 학습 데이터/설정 (v1) |
| ~~[umyunsang/civil-complaint-exaone-merged](https://huggingface.co/umyunsang/civil-complaint-exaone-merged)~~ | 폐기된 LoRA v1 기반 병합 |
| ~~[umyunsang/civil-complaint-exaone-awq](https://huggingface.co/umyunsang/civil-complaint-exaone-awq)~~ | 폐기된 병합 모델 기반 양자화 |
| ~~[umyunsang/GovOn-EXAONE-AWQ-v2](https://huggingface.co/umyunsang/GovOn-EXAONE-AWQ-v2)~~ | fine-tuned 풀 모델 방식 폐기 → Multi-LoRA 전환 |
| ~~[umyunsang/GovOn-EXAONE-LoRA-v2](https://huggingface.co/umyunsang/GovOn-EXAONE-LoRA-v2)~~ | EXAONE-Deep-7.8B 기반 → EXAONE-4.0-32B 전환으로 대체 |
