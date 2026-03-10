# M2 MVP 세션 인계 문서

**작성일**: 2026-03-07
**작성 목적**: 다음 세션에서 이어서 진행할 수 있도록 현재 상태 기록
**WandB 프로젝트**: https://wandb.ai/umyun3/exaone-civil-complaint

---

## 1. 완료된 작업 (이번 세션)

### Stage 1: QLoRA 어댑터 병합 ✅
- 베이스 모델: `LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ`
- 어댑터: `umyunsang/exaone-civil-complaint-qlora` (HuggingFace)
- 병합 모델 저장: `/content/GovOn/models/merged_model`
- 병합 모델 크기: **14.56 GB** (BF16)

### Stage 2: AWQ 양자화 ✅
- 방식: W4A16g128 (4-bit weights, 16-bit activations, group_size=128)
- 양자화 시간: **10.5분** (632.9초)
- 양자화 모델 저장: `/content/GovOn/models/awq_quantized_model`
- AWQ 모델 크기: **4.94 GB** (압축률 2.95x, 66.1% 감소)
- WandB 런: https://wandb.ai/umyun3/exaone-civil-complaint/runs/4jxb31v4

### Stage 3: 평가 ⚠️ 완료되었으나 성능 미달
- 최종 사용 방법: AutoAWQ 로딩 (`AwqConfig` 사용)
- 평가 결과 저장: `/content/GovOn/docs/outputs/M2_MVP/benchmark_results.json`
- WandB 런: https://wandb.ai/umyun3/exaone-civil-complaint/runs/706jqzmk

---

## 2. 최종 평가 결과

| 지표 | 실제 값 | 목표 | 상태 |
|------|---------|------|------|
| Perplexity | **3.1957** | - | ✅ 양호 |
| 분류 정확도 | **2.00%** (1/50) | ≥85% | ❌ 미달 |
| BLEU Score | **12.29** | ≥30 | ❌ 미달 |
| ROUGE-L | **21.36** | ≥40 | ❌ 미달 |
| 평균 레이턴시 | **9.293s** | <2s | ❌ 미달 |
| P50 레이턴시 | **9.291s** | <2s | ❌ 미달 |
| P95 레이턴시 | **9.384s** | <5s | ❌ 미달 |
| Throughput | **13.8 tok/s** | - | 참고 |
| GPU VRAM | **4.95 GB** | <8GB | ✅ 통과 |
| 모델 크기 | **4.94 GB** | <5GB | ✅ 통과 |

---

## 3. 발생한 이슈 및 근본 원인

### Issue 1: AWQ 모델 로딩 방법 혼선
**증상**: `ImportError: Loading an AWQ quantized model requires gptqmodel`
**원인**: 최신 transformers가 AWQ 모델 로딩에 gptqmodel을 요구
**시도 1**: `pip install gptqmodel` → `ModuleNotFoundError: gptqmodel_exllamav2_awq_kernels` (ExLlama 커널 미설치)
**해결책**: AutoAWQ 직접 사용 (`from awq import AutoAWQForCausalLM`)

### Issue 2: WandB 버전 충돌 (일시적)
**증상**: `ImportError: cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'`
**원인**: wandb 패키지 내부 proto 파일 불일치 (설치 후 재시작으로 해소)

### Issue 3: 분류 정확도 2% (핵심 이슈)
**증상**: 50개 중 1개만 정답 (모두 "unknown" 예측)
**근본 원인**:
1. **평가 스크립트 프롬프트 형식 오류**: EXAONE 모델은 `apply_chat_template(add_generation_prompt=True)` 필수인데 미적용
2. **출력 파서 부재**: 모델이 자유 형식으로 답변하는데 카테고리 파싱 로직 없음
3. **`<thought>` 태그 처리**: EXAONE-Deep은 추론 과정을 `<thought>...</thought>`로 감싸서 출력, 파서가 이를 처리 못함
4. **max_new_tokens=128**: 분류 작업에 비해 너무 많아 레이턴시 증가 (실제 분류에는 5-10 토큰이면 충분)

### Issue 4: 높은 레이턴시 (9.3s)
**원인**: 분류 작업에 max_new_tokens=128 설정. EXAONE-Deep의 추론 토큰(`<thought>`) 생성으로 과도한 토큰 생성
**해결책**: 분류 평가 시 max_new_tokens=30~50으로 제한

---

## 4. 다음 세션에서 할 일 (우선순위 순)

### 4-1. 평가 스크립트 수정 (최우선) 🔴
파일: `/content/GovOn/src/evaluation/evaluate_model.py`

```python
# 현재 (잘못된 방식)
inputs = tokenizer(prompt, return_tensors="pt")

# 수정해야 할 방식 (EXAONE 권장)
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # <thought>\n 자동 추가
    return_tensors="pt"
)
```

### 4-2. 분류 프롬프트 수정 🔴
```python
# 구조화된 출력 강제
prompt = f"""다음 민원을 아래 카테고리 중 하나로 분류하세요.
반드시 카테고리 이름만 답하세요.

카테고리: ["환경오염", "교통/주차", "시설물관리", "민원서비스", "other"]

민원: {complaint_text}

카테고리:"""
```

### 4-3. 출력 파서 수정 (thought 태그 처리) 🔴
```python
import re

def parse_category(output_text):
    # <thought>...</thought> 제거
    clean = re.sub(r'<thought>.*?</thought>', '', output_text, flags=re.DOTALL)
    # </thought> 이후 텍스트 추출
    if '</thought>' in output_text:
        clean = output_text.split('</thought>')[-1]
    # 카테고리 키워드 찾기
    categories = ["환경오염", "교통/주차", "시설물관리", "민원서비스", "other"]
    for cat in categories:
        if cat in clean:
            return cat
    return "unknown"
```

### 4-4. max_new_tokens 조정 🟡
- 분류 평가: `max_new_tokens=50` (현재 128)
- 생성 평가: `max_new_tokens=256` (레이턴시 목표 충족을 위해)

### 4-5. AWQ 모델 HuggingFace 업로드 (선택) 🟢
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/content/GovOn/models/awq_quantized_model",
    repo_id="umyunsang/exaone-civil-complaint-awq",
    repo_type="model"
)
```

---

## 5. 환경 정보

### 현재 설치된 주요 패키지
- `autoawq` (최신) - AWQ 모델 로딩에 사용
- `gptqmodel` - transformers의 AWQ 로딩에 필요하지만 ExLlama 커널 미설치로 직접 사용 불가
- `wandb 0.25.0` - 로깅 정상 동작
- CUDA GPU 환경 (A100 추정)

### AWQ 모델 로딩 방법 (검증된 방법)
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/content/GovOn/models/awq_quantized_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=True,
    trust_remote_code=True,
    safetensors=True
)
model = model.to("cuda")
```

### 파일 경로 참조
- 병합 모델: `/content/GovOn/models/merged_model`
- AWQ 양자화 모델: `/content/GovOn/models/awq_quantized_model`
- 평가 스크립트: `/content/GovOn/src/evaluation/evaluate_model.py`
- 벤치마크 결과: `/content/GovOn/docs/outputs/M2_MVP/benchmark_results.json`
- 양자화 로그: `/content/GovOn/models/awq_quantized_model/quantization_log.json`

---

## 6. 관련 WandB 런 목록

| 단계 | 런 이름 | 런 ID | 링크 |
|------|---------|-------|------|
| Stage 2: AWQ 양자화 | awq-quantize-20260307-0542 | 4jxb31v4 | https://wandb.ai/umyun3/exaone-civil-complaint/runs/4jxb31v4 |
| Stage 3: 평가 (실패) | evaluation-20260307-0556 | dofv5ppu | https://wandb.ai/umyun3/exaone-civil-complaint/runs/dofv5ppu |
| Stage 3: 평가 (실패) | evaluation-20260307-0606 | pghfke59 | https://wandb.ai/umyun3/exaone-civil-complaint/runs/pghfke59 |
| Stage 3: 평가 (성공) | evaluation-20260307-0637 | 706jqzmk | https://wandb.ai/umyun3/exaone-civil-complaint/runs/706jqzmk |

---

## 7. 깃허브 이슈 참조

- M2 MVP 진행: #17
- 평가 스크립트 수정 필요: (이번 세션에서 새로 생성 - issue 번호 확인 필요)
