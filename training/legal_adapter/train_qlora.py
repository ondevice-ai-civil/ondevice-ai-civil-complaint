"""
GovOn Legal Adapter Training Script
=====================================
EXAONE 4.0-32B + Unsloth QLoRA (r16) for legal interpretation & citation.

Usage:
    HF_TOKEN=<token> python train_qlora.py

Environment:
    HF_TOKEN         : HuggingFace token (required)
    OUTPUT_REPO      : HF Hub repo to push adapter (default: siwo/govon-legal-adapter)
    TRAIN_SAMPLE_SIZE: 학습 샘플 수 (default: 100000, None=전체)

1차 학습 결과:
    - 환경: HuggingFace Spaces A100 80GB
    - 학습 스텝: 365 / Final loss: 0.889
    - HF Hub: https://huggingface.co/siwo/govon-legal-adapter
"""

from __future__ import annotations

import os
import logging

import torch
from datasets import load_dataset
from huggingface_hub import login

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (1차 학습 기준)
# ---------------------------------------------------------------------------
BASE_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"
DATASET_REPO = "umyunsang/govon-legal-response-data"
OUTPUT_REPO = os.getenv("OUTPUT_REPO", "siwo/govon-legal-adapter")
OUTPUT_DIR = "./outputs"

MAX_SEQ_LENGTH = 1024
_sample_env = os.getenv("TRAIN_SAMPLE_SIZE", "100000")
TRAIN_SAMPLE_SIZE = None if _sample_env.lower() == "none" else int(_sample_env)

LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.03

SYSTEM_PROMPT = (
    "당신은 대한민국 법률 전문가입니다. "
    "법령 조항과 판례를 정확하게 인용하여 법률 질문에 답변해 주세요."
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. A100 80GB recommended.")
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN required")
    login(token=hf_token, add_to_git_credential=False)

    logger.info("=" * 60)
    logger.info("GovOn Legal Adapter Training")
    logger.info("Base: %s | LoRA r=%d alpha=%d", BASE_MODEL, LORA_RANK, LORA_ALPHA)
    logger.info("Dataset: %s | Sample: %s", DATASET_REPO, TRAIN_SAMPLE_SIZE)
    logger.info("=" * 60)

    # 1. Load model
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    logger.info("LoRA applied")

    # 3. Dataset
    ds = load_dataset(DATASET_REPO)
    train_ds = ds["train"]

    if TRAIN_SAMPLE_SIZE is not None and len(train_ds) > TRAIN_SAMPLE_SIZE:
        train_ds = train_ds.shuffle(seed=42).select(range(TRAIN_SAMPLE_SIZE))
        logger.info("Sampled %d from dataset", TRAIN_SAMPLE_SIZE)
    else:
        train_ds = train_ds.shuffle(seed=42)
        logger.info("Using full dataset (%d samples)", len(train_ds))

    val_ds = ds.get("validation") or ds.get("val")
    if val_ds is None:
        split = train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    def format_chat(example):
        user_content = example.get("instruction", "")
        if example.get("input"):
            user_content = f"{user_content}\n\n{example['input']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example.get("output", "")},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            text = (
                f"[|system|]{SYSTEM_PROMPT}[|endofturn|]\n"
                f"[|user|]{user_content}[|endofturn|]\n"
                f"[|assistant|]{example.get('output', '')}[|endofturn|]"
            )
        return {"text": text}

    train_ds = train_ds.map(format_chat, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_chat, remove_columns=val_ds.column_names)

    # 4. Train
    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        tf32=True,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=False,
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    result = trainer.train()
    logger.info("Training complete: %s", result.metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 5. Push to Hub
    logger.info("Pushing to %s", OUTPUT_REPO)
    model.push_to_hub(
        OUTPUT_REPO, token=hf_token, private=True,
        commit_message=f"feat: EXAONE 4.0-32B Unsloth QLoRA legal adapter (r{LORA_RANK})",
    )
    tokenizer.push_to_hub(OUTPUT_REPO, token=hf_token, private=True)
    logger.info("Done: https://huggingface.co/%s", OUTPUT_REPO)


if __name__ == "__main__":
    main()
