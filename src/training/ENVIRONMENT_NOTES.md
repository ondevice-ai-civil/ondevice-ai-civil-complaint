# EXAONE-Deep-7.8B QLoRA 학습 환경 설정 가이드

본 문서는 Google Colab L4 환경에서 `EXAONE-Deep-7.8B` 모델을 QLoRA로 파인튜닝하기 위해 구축된 현재 환경을 기록합니다.

## 1. 하드웨어 환경
- **GPU**: NVIDIA L4 (24GB VRAM)
- **CUDA**: 12.1+ (vLLM 및 QLoRA 학습에 최적화)

## 2. 주요 라이브러리 버전
- **transformers**: 5.3.0 (최신 dev/stable 버전, `RopeParameters` 포함)
- **trl**: 0.12.0 (`DataCollatorForCompletionOnlyLM` 포함 버전으로 다운그레이드)
- **peft**: 0.14.0+
- **bitsandbytes**: 0.45.0+ (4-bit 양자화용)
- **accelerate**: 1.3.0+

## 3. 핵심 수정 사항 (Compatibility Patches)

### 3.1 `transformers` 라이브러리 패치
`transformers` 5.3.0 버전에서 `check_model_inputs`가 삭제되어 `EXAONE` 모델의 `modeling_exaone.py`에서 임포트 에러가 발생합니다. 이를 해결하기 위해 아래와 같이 수동 패치를 적용했습니다:

- **파일 경로**: `/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py`
- **추가 내용**:
  ```python
  def check_model_inputs(func):
      return func
  ```

### 3.2 `train_qlora.py` 스크립트 수정 (Monkey-patching)
`EXAONE` 모델이 PEFT와 연동될 때 `get_input_embeddings` 메서드가 구현되어 있지 않아 발생하는 `NotImplementedError`를 해결하기 위해 스크립트 내에서 동적으로 패치했습니다:

```python
# Exaone 모델을 위한 몽키 패치
try:
    model.get_input_embeddings()
except (NotImplementedError, AttributeError):
    model.get_input_embeddings = lambda: model.transformer.wte
```

### 3.3 TrainingArguments 파라미터 변경
`transformers` 5.3.0 이상 버전에서 `evaluation_strategy`가 `eval_strategy`로 변경되었습니다:
- `evaluation_strategy="steps"` -> `eval_strategy="steps"`

## 4. 실행 명령어 (권장)
```bash
python src/training/train_qlora.py \
    --train_path data/processed/civil_complaint_train.jsonl \
    --val_path data/processed/civil_complaint_val.jsonl \
    --output_dir ./models/checkpoints/exaone-civil-qlora \
    --epochs 1 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4
```
