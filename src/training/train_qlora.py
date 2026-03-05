"""
EXAONE-Deep-7.8B QLoRA Fine-tuning Script
Optimized for Civil Complaint Dataset & Colab A100/L4 Environment
"""

import os
import torch
import argparse
import json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune EXAONE-Deep-7.8B with QLoRA")
    parser.add_argument("--model_id", type=str, default="LGAI-EXAONE/EXAONE-Deep-7.8B", help="Base model ID")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation JSONL")
    parser.add_argument("--output_dir", type=str, default="./models/checkpoints/exaone-civil-qlora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--peft_config_path", type=str, default="src/training/peft_config.json", help="Path to PEFT config JSON")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--wandb_project", type=str, default="exaone-civil-complaint")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    # 1. 토크나이저 로드 (EXAONE 표준 설정)
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # SFT 학습 시 권장

    # 2. 4-bit 양자화 설정 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. 모델 로드 (QLoRA)
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Monkey-patching for Exaone model (missing in transformers 5.3.0 dev implementation)
    # Check if they are actually implemented or if they raise NotImplementedError
    try:
        model.get_input_embeddings()
    except (NotImplementedError, AttributeError):
        model.get_input_embeddings = lambda: model.transformer.wte
        print("Monkey-patched get_input_embeddings")
    
    try:
        model.get_output_embeddings()
    except (NotImplementedError, AttributeError):
        model.get_output_embeddings = lambda: model.lm_head
        print("Monkey-patched get_output_embeddings")

    # 4. LoRA 설정 (Config 파일 또는 인자 반영)
    model = prepare_model_for_kbit_training(model)

    
    if os.path.exists(args.peft_config_path):
        print(f"Loading PEFT config from {args.peft_config_path}")
        # LoraConfig.from_json_file은 dict를 반환하므로 **를 사용하여 초기화
        with open(args.peft_config_path, "r") as f:
            config_dict = json.load(f)
        lora_config = LoraConfig(**config_dict)
    else:
        print(f"Config file not found at {args.peft_config_path}. Using command line arguments.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 데이터셋 포맷팅 (EXAONE Chat Template + <thought> 태그)
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            # PRD: EXAONE 표준 포맷 적용
            # [|system|]\n{system}\n[|user|]\n{input}\n[|assistant|]\n{output}
            messages = [
                {"role": "system", "content": "당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다."},
                {"role": "user", "content": f"{example['instruction'][i]}\n\n{example['input'][i]}"},
                {"role": "assistant", "content": example['output'][i]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    # 데이터 로드
    dataset = load_dataset("json", data_files={"train": args.train_path, "validation": args.val_path})

    # 6. 학습 인자 설정 (PRD 명세 반영)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"exaone-qlora-{datetime.now().strftime('%Y%m%d-%H%M')}",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )

    # 7. Trainer 초기화 및 학습
    # Note: EXAONE의 추론 과정을 학습하기 위해 completion_only 방식을 사용할 수 있으나, 
    # 일반적인 SFTTrainer 구성을 우선 적용합니다.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # 8. 최종 저장
    print(f"Saving model to {args.output_dir}/final")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Training Complete!")

if __name__ == "__main__":
    main()
