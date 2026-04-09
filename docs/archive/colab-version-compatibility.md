# Colab 환경 버전 호환성 문제 및 해결 방법

> **최종 업데이트**: 2026-03-14
> **관련 이슈**: #82 (평가 파이프라인), #67 (QLoRA 하이퍼파라미터), #47 (AWQ 양자화)

## 1. 핵심 원칙

**LoRA 어댑터를 로드할 때는 반드시 학습 시점의 transformers 버전과 모델 코드(revision)를 사용해야 한다.**

LoRA 어댑터는 베이스 모델의 특정 레이어에 저랭크 행렬을 주입하는 방식이므로,
모델의 `forward()` 구현이 달라지면 학습된 가중치가 엉뚱한 위치에 적용되어 **출력이 무의미해진다**.

## 2. 문제 발생 배경

### EXAONE-Deep-7.8B 모델 코드 변경 이력

| 날짜 | HF Revision | 변경 내용 |
|------|-------------|-----------|
| 2025-03-19 | `17b70148e344` | 학습 당시 사용된 원본 `modeling_exaone.py` |
| 2026-02-06 | `8a120673fc73` | **transformers v5 대응 전면 재작성** — forward pass 구조 변경 |

- LoRA 학습 시점 (2025년): `transformers ~4.44`, 원본 `modeling_exaone.py`
- Colab 기본 환경 (2026년): `transformers 4.57+`, 재작성된 `modeling_exaone.py` 자동 다운로드

### trust_remote_code=True의 동작 방식

EXAONE 모델은 `trust_remote_code=True`가 필수이며, HuggingFace 리포지토리에서
`modeling_exaone.py`를 **동적으로 다운로드**하여 실행한다.
revision을 지정하지 않으면 **최신 버전**이 자동으로 적용되므로,
학습 시점과 다른 코드가 로드될 수 있다.

## 3. 발생한 오류들 (transformers 4.57 + 최신 EXAONE 코드)

### 오류 1: rope_parameters TypeError
```
TypeError: 'NoneType' object is not subscriptable
→ self.config.rope_parameters["rope_type"]
```
- **원인**: 새 모델 코드가 `rope_parameters` 설정을 요구하지만, 이전 config에는 없음
- **임시 해결**: `AutoConfig`에서 `rope_parameters` dict 수동 구성
- **근본 해결**: 학습 시점 revision 사용 시 발생하지 않음

### 오류 2: get_input_embeddings NotImplementedError
```
NotImplementedError: `get_input_embeddings` not auto-handled for ExaoneModel
```
- **원인**: 새 transformers에서 `PreTrainedModel`의 임베딩 접근 방식 변경
- **임시 해결**: `PreTrainedModel.get_input_embeddings` 몽키패치
- **근본 해결**: transformers 4.44~4.49 사용 시 발생하지 않음

### 오류 3: AttentionInterface.get_interface
```
AttributeError: 'AttentionInterface' object has no attribute 'get_interface'
```
- **원인**: transformers v5에서 어텐션 인터페이스 API 변경
- **근본 해결**: transformers 4.44~4.49 사용 시 발생하지 않음

### 오류 4: 쓰레기 출력 (가장 위험)
```
모델 출력: "큰일 큰일 작전작전 loci loci..."
```
- **원인**: 위 오류들을 모두 패치로 우회해도, forward pass 구조 자체가 달라서
  LoRA 가중치가 엉뚱한 위치에 적용됨
- **증상**: 문법적으로 한국어처럼 보이지만 내용이 전혀 무의미한 텍스트 생성
- **근본 해결**: 학습 시점 버전 조합으로 전환

## 4. 해결 방법

### 패키지 설치 (Colab 셀)

```python
# 기존 버전 제거 후 학습 호환 버전 설치
!pip uninstall -y transformers accelerate -q
!pip install -q "transformers>=4.44,<4.50" "accelerate>=1.3.0,<2.0" peft
!pip install -q "numpy>=1.26,<3" "pandas>=2.2.2" sacrebleu rouge_score \
    bert_score wandb tqdm bitsandbytes tabulate "protobuf<5.0.0"
```

### 모델 로딩 시 revision 고정

```python
EXAONE_REVISION = "17b70148e344"  # 2025-03-19: 학습 호환 마지막 revision

tokenizer = AutoTokenizer.from_pretrained(
    "LGAI-EXAONE/EXAONE-Deep-7.8B",
    trust_remote_code=True,
    revision=EXAONE_REVISION,      # ← 반드시 지정
)

base_model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-Deep-7.8B",
    revision=EXAONE_REVISION,       # ← 반드시 지정
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
```

### EOS 토큰 처리

EXAONE의 EOS 토큰은 `[|endofturn|]` (id=361)이다.
생성 시 명시적으로 지정해야 한다:

```python
def get_eos_ids():
    ids = [tokenizer.eos_token_id]
    for tok in ['[|endofturn|]', '<|endofturn|>']:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids

output = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=False,
    repetition_penalty=1.1,
    eos_token_id=get_eos_ids(),
)
```

## 5. 체크리스트

Colab에서 EXAONE + LoRA 실험 시 반드시 확인할 항목:

- [ ] `transformers` 버전이 4.44~4.49 범위인가?
- [ ] `AutoTokenizer`, `AutoModelForCausalLM` 호출 시 `revision=EXAONE_REVISION`을 지정했는가?
- [ ] `trust_remote_code=True`를 사용하는가?
- [ ] 몽키패치 없이 모델이 정상 로드되는가? (패치가 필요하면 버전이 잘못된 것)
- [ ] sanity check에서 한국어 민원 답변이 정상 생성되는가?
- [ ] `eos_token_id`에 `[|endofturn|]`(361)을 포함시켰는가?

## 6. 관련 파일

| 파일 | 설명 |
|------|------|
| `notebooks/M3_issue82_evaluation/02_evaluate_standard.ipynb` | 표준 평가 파이프라인 (해결 적용됨) |
| `notebooks/M3_issue67_hparam/03_qlora_hparam_optimization.ipynb` | QLoRA 학습 노트북 (학습 시점 환경 참조) |

## 7. 향후 주의사항

- **AWQ 재양자화 시에도** 동일한 버전 조합 사용 필요 (merged 모델 생성 → AWQ 변환)
- **새로운 LoRA 학습 시** transformers 버전을 업그레이드하면, 이후 추론/평가도 해당 버전 사용 필요
- HuggingFace 리포지토리의 `modeling_exaone.py`가 다시 변경될 수 있으므로, **항상 revision 고정**을 권장
