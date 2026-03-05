# Colab Quickstart Guide
## EXAONE-3.5-7.8B QLoRA 파인튜닝 및 AWQ 양자화

**환경**: Google Colab Pro A100
**예상 소요 시간**: 데이터 준비 2시간 + 학습 6시간 + 양자화 1시간 = 총 9시간

---

## 전체 워크플로우

```
1. 환경 설정 (30분)
   └─> 라이브러리 설치, GPU 확인, 프로젝트 클론

2. 데이터 준비 (2시간)
   └─> AI Hub 다운로드, 전처리, 캘리브레이션 데이터 생성

3. QLoRA 학습 (6시간)
   └─> 모델 로드, 학습 실행, 체크포인트 저장

4. AWQ 양자화 (1시간)
   └─> LoRA 병합, AWQ 양자화, 성능 비교

5. 평가 및 배포 (30분)
   └─> 평가 실행, 모델 다운로드
```

---

## Step 0: Colab 런타임 설정

### GPU 확인 및 변경
1. Colab 메뉴: `런타임` > `런타임 유형 변경`
2. 하드웨어 가속기: **A100 GPU** 선택
3. 런타임 모양: **High-RAM** 선택 (권장)
4. 저장 후 연결

### GPU 확인
```python
!nvidia-smi
```

**예상 출력**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.1   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   34C    P0    46W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**중요**: A100이 아닌 경우 런타임을 다시 설정하세요.

---

## Step 1: 환경 설정 (30분)

### 1.1 필수 라이브러리 설치

```bash
%%bash

# 시스템 패키지 업데이트
apt-get update -qq
apt-get install -y git-lfs
git lfs install

# Python 라이브러리 설치
pip install -q --upgrade pip

# Core ML libraries
pip install -q torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face ecosystem
pip install -q transformers==4.40.0
pip install -q datasets==2.18.0
pip install -q accelerate==0.28.0
pip install -q peft==0.10.0

# Quantization
pip install -q bitsandbytes==0.43.0
pip install -q autoawq==0.2.0
pip install -q optimum==1.17.0

# Training utilities
pip install -q trl==0.8.1
pip install -q einops==0.7.0
pip install -q sentencepiece==0.2.0

# Evaluation
pip install -q evaluate==0.4.1
pip install -q rouge-score==0.1.2
pip install -q sacrebleu==2.4.0

# Monitoring
pip install -q wandb==0.16.4

# Utilities
pip install -q python-dotenv==1.0.1
pip install -q tqdm==4.66.2

echo "✓ Installation complete!"
```

### 1.2 프로젝트 클론

```bash
%%bash

cd /content
git clone https://github.com/YOUR_USERNAME/ondevice-ai-civil-complaint.git
cd ondevice-ai-civil-complaint
echo "✓ Project cloned!"
```

### 1.3 환경 변수 설정

```python
import os
from google.colab import userdata

# Colab Secrets에서 API 키 로드
# (Colab 좌측 '🔑' 아이콘에서 등록)
try:
    os.environ["AIHUB_API_KEY"] = userdata.get('AIHUB_API_KEY')
    os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')
    print("✓ API keys loaded from Colab Secrets")
except:
    print("⚠ API keys not found. Please add them to Colab Secrets.")
    print("  Go to: 🔑 icon > Add new secret")
    print("  - AIHUB_API_KEY: Your AI Hub API key")
    print("  - WANDB_API_KEY: Your Weights & Biases API key")
```

### 1.4 디렉토리 생성

```bash
%%bash

cd /content/ondevice-ai-civil-complaint
mkdir -p data/raw/aihub data/raw/seoul_api
mkdir -p data/processed data/calibration
mkdir -p models/base models/checkpoints models/merged models/quantized
mkdir -p logs

tree -L 2 -d
echo "✓ Directories created!"
```

---

## Step 2: 데이터 준비 (2시간)

### 2.1 AI Hub 데이터 다운로드

```bash
%%bash

# aihubshell 다운로드 및 설정
cd /content
wget -q https://api.aihub.or.kr/down/aihubshell_linux.tar.gz
tar -xzf aihubshell_linux.tar.gz
chmod +x aihubshell

# 데이터셋 다운로드 (71852: 공공 민원 상담 LLM 데이터)
# 주의: 실제 다운로드는 약 1-2시간 소요
export AIHUB_API_KEY="${AIHUB_API_KEY}"
./aihubshell -mode d -datasetkey 71852

echo "✓ Dataset download started. This may take 1-2 hours."
```

**다운로드 진행 중 확인**:
```bash
# 다른 셀에서 실행 (다운로드 진행 상황 확인)
!ls -lh ~/aihub/
```

### 2.2 Mock 데이터로 테스트 (빠른 검증용)

다운로드를 기다리는 동안 Mock 데이터로 파이프라인을 먼저 테스트할 수 있습니다.

```python
import sys
sys.path.append('/content/ondevice-ai-civil-complaint')

from src.data_collection_preprocessing.pipeline import DataPipeline
from src.data_collection_preprocessing.config import get_config

# Mock 데이터로 테스트 실행
print("Testing pipeline with mock data...")
pipeline = DataPipeline()

result = pipeline.run_full_pipeline(
    use_mock=True,
    mock_samples=1000,
    output_prefix="test_civil_complaint"
)

print(f"\n{'='*60}")
print(f"Pipeline Success: {result.success}")
print(f"Raw records: {result.total_raw_records}")
print(f"Processed records: {result.total_processed_records}")
print(f"Duration: {result.duration_seconds:.2f}s")
print(f"{'='*60}")

# 품질 리포트 출력
print(pipeline.get_quality_report())
```

### 2.3 실제 데이터 전처리 실행

AI Hub 다운로드가 완료되면 실제 데이터로 전처리를 실행합니다.

```python
import sys
sys.path.append('/content/ondevice-ai-civil-complaint')

from src.data_collection_preprocessing.pipeline import DataPipeline
from src.data_collection_preprocessing.config import get_config

# 실제 데이터로 파이프라인 실행
print("Running pipeline with real data...")
pipeline = DataPipeline()

result = pipeline.run_full_pipeline(
    use_mock=False,
    output_prefix="civil_complaint"
)

# 결과 확인
print(f"\n{'='*60}")
print(f"Pipeline Success: {result.success}")
print(f"Raw records: {result.total_raw_records}")
print(f"Processed records: {result.total_processed_records}")
print(f"Duration: {result.duration_seconds:.2f}s")
print(f"\nOutput files:")
for key, path in result.output_files.items():
    print(f"  {key}: {path}")
print(f"{'='*60}")

# 품질 리포트
print("\n" + pipeline.get_quality_report())
```

### 2.4 데이터 확인

```python
import json

# Train 데이터 샘플 확인
with open('/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_train.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print("Sample training data:")
print(json.dumps(sample, indent=2, ensure_ascii=False))

# 데이터 통계
!wc -l /content/ondevice-ai-civil-complaint/data/processed/civil_complaint_*.jsonl
```

---

## Step 3: QLoRA 학습 (6시간)

### 3.1 Weights & Biases 초기화

```python
import wandb

wandb.login(key=os.environ.get("WANDB_API_KEY"))

wandb.init(
    project="exaone-civil-complaint",
    name="qlora-baseline-colab",
    config={
        "model": "EXAONE-3.5-7.8B-Instruct",
        "method": "QLoRA",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 3
    },
    tags=["colab", "qlora", "baseline"]
)

print("✓ WandB initialized")
print(f"Run URL: {wandb.run.get_url()}")
```

### 3.2 베이스 모델 로드 및 QLoRA 설정

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 재현성을 위한 시드 고정
def set_seed(seed=42):
    import random
    import numpy as np
    from transformers import set_seed as hf_set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)
    print(f"✓ Random seed set to {seed}")

set_seed(42)

# 1. 토크나이저 로드
print("Loading tokenizer...")
model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. 4-bit 양자화 설정
print("Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 3. 모델 로드 (약 5분 소요)
print("Loading EXAONE model (this may take 5 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# 4. LoRA 준비
print("Preparing model for LoRA training...")
model = prepare_model_for_kbit_training(model)

# 5. LoRA 설정
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✓ Model loaded and configured!")
```

**예상 출력**:
```
trainable params: 41,943,040 || all params: 7,841,943,040 || trainable%: 0.5348
```

### 3.3 데이터셋 준비

```python
from datasets import load_dataset

# 데이터셋 로드
print("Loading datasets...")
data_files = {
    "train": "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_train.jsonl",
    "validation": "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_val.jsonl"
}

dataset = load_dataset("json", data_files=data_files)

print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")

# Chat Template 포맷팅
def format_chat_template(example):
    """EXAONE Chat Template 적용"""
    messages = [
        {
            "role": "system",
            "content": "당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다."
        },
        {
            "role": "user",
            "content": f"{example['instruction']}\n\n{example['input']}"
        },
        {
            "role": "assistant",
            "content": example['output']
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# 포맷팅 적용
print("Formatting datasets with EXAONE chat template...")
formatted_train = dataset["train"].map(
    format_chat_template,
    num_proc=4,
    desc="Formatting train"
)
formatted_val = dataset["validation"].map(
    format_chat_template,
    num_proc=4,
    desc="Formatting validation"
)

print("✓ Datasets formatted!")
```

### 3.4 학습 실행

```python
# Training Arguments
print("Configuring training arguments...")
training_args = TrainingArguments(
    output_dir="/content/models/checkpoints/exaone-qlora-baseline",

    # 학습 설정
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16

    # 최적화
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # 정밀도
    bf16=True,
    tf32=True,

    # 메모리 최적화
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",

    # 로깅
    logging_steps=10,
    logging_dir="/content/logs",
    report_to="wandb",

    # 평가 및 저장
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # 재현성
    seed=42,
    data_seed=42,
)

# Trainer 초기화
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train,
    eval_dataset=formatted_val,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
)

# 학습 시작
print("=" * 60)
print("Starting training (this will take ~6 hours)...")
print("=" * 60)

trainer.train()

print("✓ Training complete!")
```

**학습 모니터링**:
- WandB 대시보드: 실시간 Loss, Learning Rate 확인
- Colab 출력: Step별 진행 상황 확인

### 3.5 모델 저장

```python
# 최종 모델 저장
print("Saving final model...")
trainer.save_model("/content/models/checkpoints/exaone-qlora-baseline/final")

# LoRA 어댑터만 저장 (경량, 약 150MB)
print("Saving LoRA adapter...")
model.save_pretrained("/content/models/checkpoints/exaone-qlora-baseline/lora_adapter")
tokenizer.save_pretrained("/content/models/checkpoints/exaone-qlora-baseline/lora_adapter")

print("✓ Model saved!")

# Google Drive에 백업 (선택사항)
from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/models/checkpoints/exaone-qlora-baseline/lora_adapter \
   /content/drive/MyDrive/exaone-civil-complaint-backup/

print("✓ Backup to Google Drive complete!")
```

---

## Step 4: AWQ 양자화 (1시간)

### 4.1 LoRA 어댑터 병합

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Merging LoRA adapter with base model...")

# 베이스 모델 로드 (bf16)
base_model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# LoRA 어댑터 로드 및 병합
adapter_path = "/content/models/checkpoints/exaone-qlora-baseline/lora_adapter"
merged_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = merged_model.merge_and_unload()

# 병합된 모델 저장
merged_path = "/content/models/merged/exaone-qlora-baseline-merged"
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

print(f"✓ Merged model saved to: {merged_path}")

# 메모리 정리
del base_model, merged_model
torch.cuda.empty_cache()
```

### 4.2 AWQ 양자화 실행

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

print("=" * 60)
print("AWQ Quantization")
print("=" * 60)

# 1. 병합된 모델 로드
model_path = "/content/models/merged/exaone-qlora-baseline-merged"
quant_path = "/content/models/quantized/exaone-qlora-awq-4bit"

print(f"\n[1/5] Loading merged model...")
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 캘리브레이션 데이터 로드
print("\n[2/5] Loading calibration data...")
calib_file = "/content/ondevice-ai-civil-complaint/data/calibration/exaone_civil_calibration.txt"

with open(calib_file, "r", encoding="utf-8") as f:
    calib_data = [line.strip() for line in f if line.strip()][:512]

print(f"Loaded {len(calib_data)} calibration samples")

# 3. AWQ 양자화 설정
print("\n[3/5] Configuring AWQ quantization...")
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

print(f"Configuration: {quant_config}")

# 4. 양자화 실행 (약 30분 소요)
print("\n[4/5] Performing quantization (this may take 30 minutes)...")
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_data
)

# 5. 양자화 모델 저장
print(f"\n[5/5] Saving quantized model to: {quant_path}")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print("\n" + "=" * 60)
print("AWQ Quantization Complete!")
print("=" * 60)

# 크기 비교
import os
original_size = sum(
    os.path.getsize(os.path.join(model_path, f))
    for f in os.listdir(model_path) if f.endswith('.safetensors')
) / (1024**3)

quantized_size = os.path.getsize(
    os.path.join(quant_path, "model.safetensors")
) / (1024**3)

print(f"Original model: {original_size:.2f} GB")
print(f"Quantized model: {quantized_size:.2f} GB")
print(f"Compression: {original_size/quantized_size:.2f}x")
print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
```

---

## Step 5: 평가 (30분)

### 5.1 모델 추론 테스트

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 양자화 모델 로드
print("Loading quantized model for inference...")
model_path = "/content/models/quantized/exaone-qlora-awq-4bit"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 테스트 민원
test_complaint = """[카테고리: 도로/교통]
민원 내용: 우리 아파트 앞 도로에 가로등이 고장나서 밤에 너무 어둡습니다.
주민들이 불안해하고 있으니 빠른 수리 부탁드립니다."""

# 프롬프트 생성
prompt = f"""[|system|]
당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다.
[|user|]
다음 민원에 대해 단계적으로 분석하고, 표준 서식에 맞춰 공손하고 명확한 답변을 작성하세요.

{test_complaint}
[|assistant|]
"""

# 추론
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

import time
start = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.95,
        do_sample=True
    )

end = time.time()

# 결과 출력
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = generated_text.split("[|assistant|]")[-1].strip()

print("=" * 60)
print("Generated Answer:")
print("=" * 60)
print(answer)
print("\n" + "=" * 60)
print(f"Inference time: {(end-start)*1000:.0f} ms")
print("=" * 60)
```

### 5.2 Test Set 평가

```python
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

# 평가 메트릭 초기화
bleu = load("sacrebleu")
rouge = load("rouge")

# Test 데이터 로드
test_dataset = load_dataset(
    "json",
    data_files="/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"
)["train"]

print(f"Evaluating on {len(test_dataset)} test samples...")

# 추론 및 평가
predictions = []
references = []

for i, example in enumerate(tqdm(test_dataset.select(range(100)))):  # 샘플 100개로 빠른 평가
    prompt = f"[|user|]\n{example['instruction']}\n\n{example['input']}\n[|assistant|]\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.95,
            do_sample=False  # Greedy for evaluation
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated.split("[|assistant|]")[-1].strip()

    predictions.append(answer)
    references.append(example['output'])

# 메트릭 계산
bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
rouge_scores = rouge.compute(predictions=predictions, references=references)

# 결과 출력
results = {
    "bleu": bleu_score["score"],
    "rouge_1": rouge_scores["rouge1"] * 100,
    "rouge_2": rouge_scores["rouge2"] * 100,
    "rouge_l": rouge_scores["rougeL"] * 100,
}

print("\n" + "=" * 60)
print("Evaluation Results (100 samples)")
print("=" * 60)
for metric, value in results.items():
    print(f"{metric:12s}: {value:6.2f}")
print("=" * 60)

# WandB에 로깅
wandb.log(results)
```

---

## Step 6: 모델 다운로드 및 배포 준비

### 6.1 Google Drive에 백업

```python
from google.colab import drive
drive.mount('/content/drive')

# 최종 모델 복사
!mkdir -p /content/drive/MyDrive/exaone-civil-complaint-models

# AWQ 양자화 모델 (배포용, 약 4GB)
!cp -r /content/models/quantized/exaone-qlora-awq-4bit \
   /content/drive/MyDrive/exaone-civil-complaint-models/

# LoRA 어댑터 (백업용, 약 150MB)
!cp -r /content/models/checkpoints/exaone-qlora-baseline/lora_adapter \
   /content/drive/MyDrive/exaone-civil-complaint-models/

print("✓ Models backed up to Google Drive!")
```

### 6.2 Hugging Face Hub에 업로드 (선택사항)

```python
from huggingface_hub import HfApi, login

# Hugging Face 로그인
hf_token = userdata.get('HF_TOKEN')
login(token=hf_token)

# 모델 업로드
api = HfApi()

repo_id = "YOUR_USERNAME/exaone-civil-complaint-awq"

api.upload_folder(
    folder_path="/content/models/quantized/exaone-qlora-awq-4bit",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload AWQ quantized EXAONE model for civil complaint processing"
)

print(f"✓ Model uploaded to: https://huggingface.co/{repo_id}")
```

### 6.3 로컬 다운로드 (ZIP)

```python
# 모델을 ZIP으로 압축하여 다운로드
!zip -r exaone-civil-awq-4bit.zip /content/models/quantized/exaone-qlora-awq-4bit

from google.colab import files
files.download('exaone-civil-awq-4bit.zip')

print("✓ Model download started!")
```

---

## 문제 해결 (Troubleshooting)

### 1. OOM (Out of Memory) 에러

**증상**: `CUDA out of memory` 에러 발생

**해결책**:
```python
# Batch size 감소
per_device_train_batch_size=2  # 4 → 2
gradient_accumulation_steps=8  # 4 → 8

# 또는 Gradient checkpointing 강화
gradient_checkpointing=True
gradient_checkpointing_kwargs={"use_reentrant": False}
```

### 2. 세션 타임아웃

**증상**: 학습 중 Colab 세션이 끊김

**해결책**:
```python
# Google Drive 마운트 및 자동 저장
from google.colab import drive
drive.mount('/content/drive')

# Training arguments에 추가
output_dir="/content/drive/MyDrive/checkpoints/exaone-qlora"
save_steps=200  # 더 자주 저장
```

### 3. GPU 할당 실패

**증상**: A100을 받지 못함

**해결책**:
- Colab Pro+ 업그레이드
- 시간대를 변경하여 재시도 (밤 시간대 권장)
- 런타임 연결 해제 후 재연결

### 4. 데이터 다운로드 실패

**증상**: AI Hub 다운로드가 진행되지 않음

**해결책**:
```bash
# API 키 확인
echo $AIHUB_API_KEY

# 수동 다운로드 후 업로드
# 1. 로컬에서 다운로드
# 2. Google Drive에 업로드
# 3. Colab에서 Drive 마운트하여 사용
```

---

## 체크리스트

### 환경 설정
- [ ] A100 GPU 할당 확인
- [ ] 필수 라이브러리 설치 완료
- [ ] 프로젝트 클론 완료
- [ ] API 키 설정 완료 (AIHUB, WANDB)

### 데이터 준비
- [ ] AI Hub 데이터 다운로드 완료
- [ ] 전처리 파이프라인 실행 완료
- [ ] Train/Val/Test 분할 확인
- [ ] 캘리브레이션 데이터 생성 완료

### 학습
- [ ] 베이스 모델 로드 성공
- [ ] QLoRA 설정 완료
- [ ] 학습 시작 (WandB 로그 확인)
- [ ] 체크포인트 저장 확인
- [ ] 최종 모델 저장 완료

### 양자화
- [ ] LoRA 병합 완료
- [ ] AWQ 양자화 실행 완료
- [ ] 양자화 모델 저장 확인
- [ ] 크기 감소 확인 (50% 이상)

### 평가
- [ ] 추론 테스트 성공
- [ ] Test set 평가 완료
- [ ] BLEU/ROUGE 점수 확인
- [ ] 성능 목표 달성 확인

### 백업
- [ ] Google Drive 백업 완료
- [ ] Hugging Face Hub 업로드 (선택)
- [ ] 로컬 다운로드 완료

---

## 예상 소요 시간 요약

| 단계 | 작업 | 예상 시간 |
|------|------|----------|
| 0 | Colab 설정 | 5분 |
| 1 | 환경 설정 | 30분 |
| 2 | 데이터 준비 | 2시간 |
| 3 | QLoRA 학습 | 6시간 |
| 4 | AWQ 양자화 | 1시간 |
| 5 | 평가 | 30분 |
| 6 | 백업/배포 | 15분 |
| **총계** | | **10시간 20분** |

**권장 스케줄**:
- Day 1: Step 0-2 (환경 설정 + 데이터 준비)
- Day 2: Step 3 (QLoRA 학습 - 밤새 실행)
- Day 3: Step 4-6 (양자화 + 평가 + 배포)

---

## 다음 단계

1. **vLLM 서빙 구축**: 양자화된 모델을 vLLM으로 서빙
2. **FastAPI 백엔드 개발**: REST API 엔드포인트 구현
3. **Streamlit UI 개발**: 민원 분석 웹 인터페이스 구축
4. **Docker 배포**: 온프레미스 환경 배포 준비

---

**문서 버전**: 1.0
**작성일**: 2026-03-05
**문의**: 프로젝트 GitHub Issues
