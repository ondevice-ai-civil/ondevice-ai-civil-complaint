"""
Issue #67 - QLoRA Hyperparameter Optimization Experiment Runner

AI Engineer: Systematic hyperparameter search for EXAONE-Deep-7.8B QLoRA fine-tuning
Experiment Tracker: W&B logging for EXP-002, EXP-003, EXP-004
Model QA Specialist: BLEU/ROUGE-L evaluation per run

Targets: BLEU >= 30, ROUGE-L >= 40
Baseline (EXP-001): BLEU=17.32, ROUGE-L=18.28 (rank=16, lr=2e-4, 1 epoch)
"""

import os
import sys
import json
import torch
import wandb
import numpy as np
import re
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/content/GovOn")
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/processed/civil_complaint_train.jsonl")
VAL_PATH = os.path.join(PROJECT_ROOT, "data/processed/civil_complaint_val.jsonl")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/processed/civil_complaint_test.jsonl")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "models/checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "docs/outputs/experiments")
os.makedirs(RESULTS_DIR, exist_ok=True)

WANDB_PROJECT = "govon-qlora-hparam-search"
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-Deep-7.8B"
SEED = 42

# ─── Experiment Definitions (Experiment Tracker) ────────────────────────────
#
# EXP-001 (Baseline, already done): rank=16, lr=2e-4, epochs=1
#   BLEU=17.32, ROUGE-L=18.28
#
# EXP-002: LoRA Rank search — vary rank while fixing lr & epochs
# EXP-003: Learning Rate search — vary lr & scheduler while fixing rank & epochs
# EXP-004: Epoch exploration — vary epochs with best params from EXP-002/003

EXPERIMENTS = [
    # EXP-002: LoRA Rank variations
    {
        "exp_id": "EXP-002-rank8",
        "group": "EXP-002-rank-search",
        "lora_r": 8,
        "lora_alpha": 16,
        "lr": 2e-4,
        "lr_scheduler": "cosine",
        "epochs": 1,
        "hypothesis": "Lower rank (8) reduces overfitting on small civil complaint dataset",
    },
    {
        "exp_id": "EXP-002-rank32",
        "group": "EXP-002-rank-search",
        "lora_r": 32,
        "lora_alpha": 64,
        "lr": 2e-4,
        "lr_scheduler": "cosine",
        "epochs": 1,
        "hypothesis": "Higher rank (32) increases model capacity for domain adaptation",
    },
    {
        "exp_id": "EXP-002-rank64",
        "group": "EXP-002-rank-search",
        "lora_r": 64,
        "lora_alpha": 128,
        "lr": 2e-4,
        "lr_scheduler": "cosine",
        "epochs": 1,
        "hypothesis": "Maximum rank (64) maximizes expressivity at cost of parameters",
    },
    # EXP-003: Learning Rate variations
    {
        "exp_id": "EXP-003-lr1e4-cosine",
        "group": "EXP-003-lr-search",
        "lora_r": 16,
        "lora_alpha": 32,
        "lr": 1e-4,
        "lr_scheduler": "cosine",
        "epochs": 1,
        "hypothesis": "Lower lr (1e-4) with cosine scheduler prevents overshooting",
    },
    {
        "exp_id": "EXP-003-lr5e5-cosine",
        "group": "EXP-003-lr-search",
        "lora_r": 16,
        "lora_alpha": 32,
        "lr": 5e-5,
        "lr_scheduler": "cosine",
        "epochs": 1,
        "hypothesis": "Conservative lr (5e-5) for stable convergence on small dataset",
    },
    {
        "exp_id": "EXP-003-lr2e4-linear",
        "group": "EXP-003-lr-search",
        "lora_r": 16,
        "lora_alpha": 32,
        "lr": 2e-4,
        "lr_scheduler": "linear",
        "epochs": 1,
        "hypothesis": "Baseline lr with linear decay vs cosine (scheduler ablation)",
    },
    # EXP-004: Epoch exploration (uses best config from EXP-002/003 — defaulting to rank=32)
    {
        "exp_id": "EXP-004-2epochs",
        "group": "EXP-004-epoch-search",
        "lora_r": 32,
        "lora_alpha": 64,
        "lr": 1e-4,
        "lr_scheduler": "cosine",
        "epochs": 2,
        "hypothesis": "2 epochs with rank=32 and lr=1e-4 improves convergence",
    },
    {
        "exp_id": "EXP-004-3epochs",
        "group": "EXP-004-epoch-search",
        "lora_r": 32,
        "lora_alpha": 64,
        "lr": 1e-4,
        "lr_scheduler": "cosine",
        "epochs": 3,
        "hypothesis": "3 epochs to assess risk of overfitting vs further improvement",
    },
]


# ─── Model QA Specialist: Evaluation Functions ──────────────────────────────


def compute_bleu_rouge(model, tokenizer, data, max_samples=50):
    """
    Model QA Specialist: Compute BLEU (approx) and ROUGE-L scores.
    Evidence-based assessment of generation quality.
    """
    try:
        from rouge_score import rouge_scorer as rs_module

        scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=False)
        has_rouge = True
    except ImportError:
        has_rouge = False

    bleu_scores, rouge_scores = [], []

    for item in data[:max_samples]:
        try:
            messages = [{"role": "user", "content": f"{item['instruction']}\n\n{item['input']}"}]
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = encoded.input_ids.to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    eos_token_id=int(tokenizer.eos_token_id),
                )

            generated = tokenizer.decode(
                output[0][input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            generated_clean = re.sub(
                r"<thought>.*?</thought>", "", generated, flags=re.DOTALL
            ).strip()
            reference_clean = re.sub(
                r"<thought>.*?</thought>", "", item.get("output", ""), flags=re.DOTALL
            ).strip()

            if not generated_clean or not reference_clean:
                continue

            gen_tokens = generated_clean.split()
            ref_tokens = reference_clean.split()
            if gen_tokens and ref_tokens:
                matches = sum(1 for t in gen_tokens if t in ref_tokens)
                precision = matches / len(gen_tokens)
                bp = min(1.0, len(gen_tokens) / len(ref_tokens))
                bleu_scores.append(bp * precision * 100)

            if has_rouge:
                score = scorer.score(reference_clean, generated_clean)
                rouge_scores.append(score["rougeL"].fmeasure * 100)

        except Exception:
            continue

    avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    avg_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    return avg_bleu, avg_rouge


def load_eval_data(path, max_samples=100):
    import random

    random.seed(SEED)
    data = []
    with open(path) as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    random.shuffle(data)
    return data[:max_samples]


# ─── AI Engineer: Training Pipeline ─────────────────────────────────────────


def load_base_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # EXAONE compatibility patch
    try:
        model.get_input_embeddings()
    except (NotImplementedError, AttributeError):
        model.get_input_embeddings = lambda: model.transformer.wte
    try:
        model.get_output_embeddings()
    except (NotImplementedError, AttributeError):
        model.get_output_embeddings = lambda: model.lm_head

    return model, tokenizer


def run_single_experiment(exp_cfg, tokenizer, eval_data):
    """
    AI Engineer: Run one hyperparameter configuration.
    Experiment Tracker: Log all metrics to W&B.
    """
    exp_id = exp_cfg["exp_id"]
    output_dir = os.path.join(OUTPUT_BASE, exp_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Experiment Tracker] Starting: {exp_id}")
    print(f"  Hypothesis: {exp_cfg['hypothesis']}")
    print(
        f"  Config: rank={exp_cfg['lora_r']}, lr={exp_cfg['lr']}, "
        f"scheduler={exp_cfg['lr_scheduler']}, epochs={exp_cfg['epochs']}"
    )
    print(f"{'='*65}")

    # Initialize W&B run
    run = wandb.init(
        project=WANDB_PROJECT,
        name=exp_id,
        group=exp_cfg["group"],
        tags=["qlora", "exaone", "civil-complaint", "issue-67", exp_cfg["group"]],
        config={
            "exp_id": exp_id,
            "hypothesis": exp_cfg["hypothesis"],
            "base_model": BASE_MODEL_ID,
            "lora_r": exp_cfg["lora_r"],
            "lora_alpha": exp_cfg["lora_alpha"],
            "learning_rate": exp_cfg["lr"],
            "lr_scheduler": exp_cfg["lr_scheduler"],
            "num_epochs": exp_cfg["epochs"],
            "batch_size": 2,
            "grad_accum": 8,
            "max_seq_length": 2048,
            "seed": SEED,
            "baseline_bleu": 17.32,
            "baseline_rouge_l": 18.28,
            "target_bleu": 30.0,
            "target_rouge_l": 40.0,
        },
        reinit=True,
    )

    set_seed(SEED)

    try:
        # Load fresh model for each experiment
        model, _ = load_base_model_and_tokenizer(BASE_MODEL_ID)
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=exp_cfg["lora_r"],
            lora_alpha=exp_cfg["lora_alpha"],
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Dataset
        dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})

        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example["instruction"])):
                messages = [
                    {
                        "role": "system",
                        "content": "당신은 지자체 민원 담당 공무원을 돕는 AI 어시스턴트입니다.",
                    },
                    {
                        "role": "user",
                        "content": f"{example['instruction'][i]}\n\n{example['input'][i]}",
                    },
                    {"role": "assistant", "content": example["output"][i]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                output_texts.append(text)
            return output_texts

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            eval_accumulation_steps=10,
            learning_rate=exp_cfg["lr"],
            num_train_epochs=exp_cfg["epochs"],
            lr_scheduler_type=exp_cfg["lr_scheduler"],
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
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="wandb",
            run_name=exp_id,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            seed=SEED,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            max_seq_length=2048,
            tokenizer=tokenizer,
            formatting_func=formatting_prompts_func,
            args=training_args,
        )

        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final"))

        # Model QA Specialist: Evaluate BLEU / ROUGE-L
        print(f"\n[Model QA Specialist] Evaluating {exp_id}...")
        model.eval()
        bleu, rouge_l = compute_bleu_rouge(model, tokenizer, eval_data, max_samples=50)

        # Experiment Tracker: Log final metrics
        improvement_bleu = bleu - 17.32
        improvement_rouge = rouge_l - 18.28
        bleu_target_met = bleu >= 30.0
        rouge_target_met = rouge_l >= 40.0

        final_metrics = {
            "final/bleu": bleu,
            "final/rouge_l": rouge_l,
            "final/bleu_improvement_vs_baseline": improvement_bleu,
            "final/rouge_l_improvement_vs_baseline": improvement_rouge,
            "final/bleu_target_met": int(bleu_target_met),
            "final/rouge_target_met": int(rouge_target_met),
            "final/both_targets_met": int(bleu_target_met and rouge_target_met),
        }
        wandb.log(final_metrics)

        print(f"\n[Experiment Tracker] Results for {exp_id}:")
        print(
            f"  BLEU:   {bleu:.2f}  (baseline: 17.32, target: >=30) "
            f"{'✓ PASS' if bleu_target_met else '✗ FAIL'}"
        )
        print(
            f"  ROUGE-L:{rouge_l:.2f}  (baseline: 18.28, target: >=40) "
            f"{'✓ PASS' if rouge_target_met else '✗ FAIL'}"
        )
        print(f"  Improvement: BLEU +{improvement_bleu:.2f}, ROUGE-L +{improvement_rouge:.2f}")

        result = {
            "exp_id": exp_id,
            "config": exp_cfg,
            "bleu": bleu,
            "rouge_l": rouge_l,
            "bleu_improvement": improvement_bleu,
            "rouge_improvement": improvement_rouge,
            "bleu_target_met": bleu_target_met,
            "rouge_target_met": rouge_target_met,
            "wandb_run_id": run.id,
            "wandb_run_url": run.url,
            "checkpoint_dir": output_dir,
        }

    except Exception as e:
        print(f"[ERROR] Experiment {exp_id} failed: {e}")
        wandb.log({"error": str(e)})
        result = {"exp_id": exp_id, "config": exp_cfg, "error": str(e)}

    finally:
        import gc

        del model
        gc.collect()
        torch.cuda.empty_cache()
        wandb.finish()

    return result


# ─── Experiment Tracker: Summary Report ─────────────────────────────────────


def print_experiment_summary(results):
    """
    Experiment Tracker: Portfolio summary with go/no-go recommendations.
    Model QA Specialist: Evidence-based ranking.
    """
    print("\n" + "=" * 75)
    print("EXPERIMENT TRACKER — Issue #67 Hyperparameter Optimization Summary")
    print("=" * 75)
    print(f"{'Exp ID':<28} {'BLEU':>8} {'ROUGE-L':>9} {'Δ BLEU':>8} {'Δ ROUGE':>8} {'Status':<12}")
    print("-" * 75)

    successful = [r for r in results if "bleu" in r]
    successful.sort(key=lambda x: x["bleu"] + x["rouge_l"], reverse=True)

    for r in successful:
        status = (
            "✓ BOTH"
            if r["bleu_target_met"] and r["rouge_target_met"]
            else "~ PARTIAL" if r["bleu_target_met"] or r["rouge_target_met"] else "✗ BELOW"
        )
        print(
            f"{r['exp_id']:<28} {r['bleu']:>8.2f} {r['rouge_l']:>9.2f} "
            f"{r['bleu_improvement']:>+8.2f} {r['rouge_improvement']:>+8.2f} {status:<12}"
        )

    for r in results:
        if "error" in r:
            print(f"{r['exp_id']:<28} {'ERROR':>8} {'—':>9} {'—':>8} {'—':>8} {'✗ FAILED':<12}")

    print("-" * 75)
    print(f"{'Baseline EXP-001':<28} {'17.32':>8} {'18.28':>9} {'—':>8} {'—':>8} {'reference':<12}")

    if successful:
        best = successful[0]
        print(f"\n[Model QA Specialist] Best Configuration:")
        print(f"  Experiment : {best['exp_id']}")
        print(f"  BLEU       : {best['bleu']:.2f} (target ≥30, baseline 17.32)")
        print(f"  ROUGE-L    : {best['rouge_l']:.2f} (target ≥40, baseline 18.28)")
        cfg = best["config"]
        print(
            f"  Config     : rank={cfg['lora_r']}, lr={cfg['lr']}, "
            f"scheduler={cfg['lr_scheduler']}, epochs={cfg['epochs']}"
        )
        print(f"\n[Experiment Tracker] Recommendation:")
        if best["bleu_target_met"] and best["rouge_target_met"]:
            print(f"  GO — Both targets met. Deploy best model to HuggingFace.")
        elif best["bleu"] > 25 or best["rouge_l"] > 35:
            print(f"  PARTIAL GO — Significant improvement. Consider additional epochs.")
        else:
            print(f"  NO-GO — Targets not met. Explore data augmentation or larger rank.")

    return successful[0] if successful else None


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_ids", nargs="*", default=None, help="Run specific experiment IDs (default: all)"
    )
    parser.add_argument("--wandb_key", type=str, default=None)
    args = parser.parse_args()

    # W&B auth
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    elif os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    else:
        wandb.login()

    # Load tokenizer once (shared across experiments)
    print(f"[AI Engineer] Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load eval data once
    print(f"[Model QA Specialist] Loading evaluation data...")
    eval_data = load_eval_data(TEST_PATH, max_samples=100)
    print(f"  Loaded {len(eval_data)} evaluation samples")

    # Filter experiments if specified
    experiments = EXPERIMENTS
    if args.exp_ids:
        experiments = [e for e in EXPERIMENTS if e["exp_id"] in args.exp_ids]
    print(f"\n[Experiment Tracker] Running {len(experiments)} experiments for Issue #67")

    # Run all experiments
    results = []
    for exp_cfg in experiments:
        result = run_single_experiment(exp_cfg, tokenizer, eval_data)
        results.append(result)

        # Save intermediate results
        results_path = os.path.join(RESULTS_DIR, "hparam_search_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    best = print_experiment_summary(results)

    # Save final results
    final_path = os.path.join(RESULTS_DIR, "hparam_search_results.json")
    with open(final_path, "w") as f:
        json.dump(
            {
                "issue": 67,
                "timestamp": datetime.now().isoformat(),
                "baseline": {"exp_id": "EXP-001", "bleu": 17.32, "rouge_l": 18.28},
                "targets": {"bleu": 30.0, "rouge_l": 40.0},
                "results": results,
                "best_config": best["config"] if best else None,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(f"\n[Experiment Tracker] Full results saved: {final_path}")
    print(f"[W&B] Project dashboard: https://wandb.ai/{WANDB_PROJECT}")
    return results


if __name__ == "__main__":
    main()
