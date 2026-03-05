# Google Colab 파인튜닝 실행 가이드
## EXAONE-Deep-7.8B QLoRA 학습

이 문서는 `src/training/train_qlora.py` 스크립트를 Google Colab 환경에서 실행하기 위한 절차를 설명합니다.

### 1. 코랩 설정
- **런타임 유형**: GPU (A100 권장)
- **권장 사양**: High-RAM 설정 활성화 (40GB VRAM 이상 권장)

### 2. 필수 라이브러리 설치
코랩 첫 번째 셀에서 아래 명령어를 실행하여 필요한 패키지를 설치합니다.

```bash
!pip install -q -U transformers datasets accelerate peft bitsandbytes trl wandb python-dotenv
```

### 3. 프로젝트 클론 및 환경 설정
```python
# 1. 저장소 클론 (본인의 레포지토리 주소로 변경)
!git clone https://github.com/umyunsang/ondevice-ai-civil-complaint.git
%cd ondevice-ai-civil-complaint

# 2. PYTHONPATH 설정
import sys
import os
sys.path.append(os.getcwd())

# 3. Weights & Biases 로그인 (선택 사항)
import wandb
wandb.login()
```

### 4. 파인튜닝 실행
전처리된 데이터셋 경로를 지정하여 학습을 시작합니다.

```bash
!python src/training/train_qlora.py \
    --train_path data/processed/civil_complaint_train.jsonl \
    --val_path data/processed/civil_complaint_val.jsonl \
    --output_dir ./models/checkpoints/exaone-civil-qlora \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4
```

### 5. 주요 파라미터 설명
- `--model_id`: 베이스 모델 (`LGAI-EXAONE/EXAONE-Deep-7.8B`).
- `--batch_size` & `--grad_accum`: 합산 시 Effective Batch Size가 결정됩니다. (예: 4x4=16). VRAM 부족 시 `batch_size`를 줄이고 `grad_accum`을 늘리세요.
- `--max_seq_length`: EXAONE은 최대 32k를 지원하지만, 학습 효율을 위해 2048 정도로 제한하는 것을 권장합니다.

### 6. 학습 결과 확인
학습이 완료되면 `./models/checkpoints/exaone-civil-qlora/final` 폴더에 LoRA 어댑터 가중치와 토크나이저 설정이 저장됩니다. 이 파일을 Google Drive로 복사하여 백업하는 것을 권장합니다.

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r ./models/checkpoints/exaone-civil-qlora /content/drive/MyDrive/
```
