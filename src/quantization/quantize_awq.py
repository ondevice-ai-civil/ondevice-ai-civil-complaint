"""
AWQ Quantization Script for EXAONE-Deep-7.8B (Merged Model)
Applies W4A16g128 quantization using AutoAWQ.
"""

import os
import sys
import time
import json
import torch
import wandb
import gc
from datetime import datetime

MERGED_MODEL_DIR = "/content/ondevice-ai-civil-complaint/models/merged_model"
AWQ_OUTPUT_DIR = "/content/ondevice-ai-civil-complaint/models/awq_quantized_model"
CALIB_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_train.jsonl"


def prepare_calibration_data(tokenizer, data_path, n_samples=512, max_length=2048):
    """Prepare domain-specific calibration data from training set."""
    import random

    random.seed(42)

    samples = []
    with open(data_path, "r") as f:
        lines = f.readlines()

    random.shuffle(lines)

    for line in lines[: n_samples * 2]:  # oversample to filter short ones
        try:
            item = json.loads(line.strip())
            # Format as chat template
            messages = [
                {"role": "user", "content": f"{item['instruction']}\n\n{item['input']}"},
                {"role": "assistant", "content": item["output"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if len(text) > 100:  # Filter very short entries
                samples.append(text)
        except (json.JSONDecodeError, KeyError):
            continue

        if len(samples) >= n_samples:
            break

    print(f"  Prepared {len(samples)} calibration samples from domain data")
    return samples


def main():
    start_time = time.time()

    # Initialize WandB
    run = wandb.init(
        project="exaone-civil-complaint",
        name=f"awq-quantize-{datetime.now().strftime('%Y%m%d-%H%M')}",
        tags=["quantization", "awq", "exaone-7.8b", "W4A16g128"],
        config={
            "merged_model_dir": MERGED_MODEL_DIR,
            "output_dir": AWQ_OUTPUT_DIR,
            "quant_config": {
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM",
            },
            "calibration_samples": 512,
            "stage": "2_awq_quantization",
        },
    )

    print("=" * 60)
    print("Stage 2: AWQ Quantization (W4A16g128)")
    print("=" * 60)

    # Step 1: Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Step 2: Prepare calibration data
    print("\n[2/4] Preparing calibration data...")
    calib_data = prepare_calibration_data(tokenizer, CALIB_DATA_PATH, n_samples=512)
    wandb.log({"calibration_samples": len(calib_data)})

    # Step 3: Load model with AutoAWQ and quantize
    print("\n[3/4] Loading model and applying AWQ quantization...")
    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        safetensors=True,
        trust_remote_code=True,
    )
    print(f"  Model loaded successfully")

    mem_before = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory before quantization: {mem_before:.2f} GB")

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print(f"  Quantization config: {quant_config}")
    print(f"  Starting quantization (this may take 20-40 minutes)...")

    quant_start = time.time()
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
    )
    quant_elapsed = time.time() - quant_start
    print(f"  Quantization completed in {quant_elapsed:.1f}s ({quant_elapsed/60:.1f}min)")

    wandb.log(
        {
            "quantization_time_seconds": quant_elapsed,
            "gpu_mem_before_quant_gb": mem_before,
        }
    )

    # Step 4: Save quantized model
    print(f"\n[4/4] Saving quantized model to {AWQ_OUTPUT_DIR}...")
    os.makedirs(AWQ_OUTPUT_DIR, exist_ok=True)
    model.save_quantized(AWQ_OUTPUT_DIR, safetensors=True)
    tokenizer.save_pretrained(AWQ_OUTPUT_DIR)

    # Also copy the modeling file with our patch
    import shutil

    src_modeling = os.path.join(MERGED_MODEL_DIR, "modeling_exaone.py")
    src_config_py = os.path.join(MERGED_MODEL_DIR, "configuration_exaone.py")
    if os.path.exists(src_modeling):
        shutil.copy2(src_modeling, os.path.join(AWQ_OUTPUT_DIR, "modeling_exaone.py"))
    if os.path.exists(src_config_py):
        shutil.copy2(src_config_py, os.path.join(AWQ_OUTPUT_DIR, "configuration_exaone.py"))

    # Calculate model size
    total_size = sum(
        os.path.getsize(os.path.join(AWQ_OUTPUT_DIR, f))
        for f in os.listdir(AWQ_OUTPUT_DIR)
        if f.endswith((".safetensors", ".bin"))
    )
    awq_size_gb = total_size / 1024**3
    print(f"  AWQ model size on disk: {awq_size_gb:.2f} GB")

    # Compare sizes
    merged_size = sum(
        os.path.getsize(os.path.join(MERGED_MODEL_DIR, f))
        for f in os.listdir(MERGED_MODEL_DIR)
        if f.endswith((".safetensors", ".bin"))
    )
    merged_size_gb = merged_size / 1024**3
    compression_ratio = merged_size_gb / awq_size_gb if awq_size_gb > 0 else 0
    size_reduction_pct = (1 - awq_size_gb / merged_size_gb) * 100 if merged_size_gb > 0 else 0

    print(f"  Merged model size: {merged_size_gb:.2f} GB")
    print(f"  AWQ model size: {awq_size_gb:.2f} GB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Size reduction: {size_reduction_pct:.1f}%")

    wandb.log(
        {
            "awq_model_size_gb": awq_size_gb,
            "merged_model_size_gb": merged_size_gb,
            "compression_ratio": compression_ratio,
            "size_reduction_pct": size_reduction_pct,
        }
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Stage 2 Complete! Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"AWQ model saved to: {AWQ_OUTPUT_DIR}")
    print(f"{'=' * 60}")

    # Save quantization log
    quant_log = {
        "stage": "2_awq_quantization",
        "timestamp": datetime.now().isoformat(),
        "merged_model_dir": MERGED_MODEL_DIR,
        "output_dir": AWQ_OUTPUT_DIR,
        "quant_config": quant_config,
        "calibration_samples": len(calib_data),
        "awq_model_size_gb": awq_size_gb,
        "merged_model_size_gb": merged_size_gb,
        "compression_ratio": compression_ratio,
        "size_reduction_pct": size_reduction_pct,
        "quantization_time_seconds": quant_elapsed,
        "total_time_seconds": elapsed,
    }
    log_path = os.path.join(AWQ_OUTPUT_DIR, "quantization_log.json")
    with open(log_path, "w") as f:
        json.dump(quant_log, f, indent=2, ensure_ascii=False)
    print(f"Quantization log saved to: {log_path}")

    wandb.log({"total_stage2_time_seconds": elapsed})
    wandb.finish()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return quant_log


if __name__ == "__main__":
    main()
