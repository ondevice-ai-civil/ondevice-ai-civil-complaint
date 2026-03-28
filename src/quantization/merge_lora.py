"""
LoRA Adapter Merge Script

Merges QLoRA adapter into the base EXAONE-Deep-7.8B model.
Outputs a full-precision (BF16) merged model.

Note: As of transformers 4.53.0+, auto_docstring, check_model_inputs,
maybe_autocast, RopeParameters, use_kernel_forward_from_hub,
use_kernel_func_from_hub, use_kernelized_func, and create_causal_mask
all exist natively — no monkey-patches needed.
"""

import gc
import json
import os
import time
from datetime import datetime

import torch
import wandb
from loguru import logger

# Paths
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-Deep-7.8B"
ADAPTER_ID = "umyunsang/civil-complaint-exaone-lora"
MERGED_OUTPUT_DIR = "/content/ondevice-ai-civil-complaint/models/merged_model"


def main():
    start_time = time.time()

    # Initialize WandB
    run = wandb.init(
        project="exaone-civil-complaint",
        name=f"lora-merge-{datetime.now().strftime('%Y%m%d-%H%M')}",
        tags=["merge", "lora", "exaone-7.8b"],
        config={
            "base_model": BASE_MODEL_ID,
            "adapter": ADAPTER_ID,
            "output_dir": MERGED_OUTPUT_DIR,
            "torch_dtype": "bfloat16",
            "stage": "1_lora_merge",
        },
    )

    logger.info("=" * 60)
    logger.info("Stage 1: LoRA Adapter Merge")
    logger.info("=" * 60)

    # Step 1: Load adapter config for validation
    logger.info("[1/5] Validating adapter config...")
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(ADAPTER_ID, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)

    logger.info(f"  Base model: {adapter_config['base_model_name_or_path']}")
    logger.info(f"  PEFT type: {adapter_config['peft_type']}")
    logger.info(f"  Rank (r): {adapter_config['r']}")
    logger.info(f"  Alpha: {adapter_config['lora_alpha']}")
    logger.info(f"  Target modules: {adapter_config['target_modules']}")
    logger.info(f"  Dropout: {adapter_config['lora_dropout']}")

    assert (
        adapter_config["base_model_name_or_path"] == BASE_MODEL_ID
    ), f"Base model mismatch: {adapter_config['base_model_name_or_path']} != {BASE_MODEL_ID}"
    assert adapter_config["peft_type"] == "LORA", "Expected LORA adapter"

    wandb.log({"adapter_config": adapter_config})
    logger.info("  [OK] Adapter config validated")

    # Step 2: Load tokenizer
    logger.info("[2/5] Loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    logger.info(f"  Vocab size: {tokenizer.vocab_size}")
    logger.info(f"  Model max length: {tokenizer.model_max_length}")

    # Step 3: Load base model in BF16 for clean merging
    logger.info("[3/5] Loading base model in BF16...")
    from transformers import AutoModelForCausalLM

    mem_before = torch.cuda.memory_allocated() / 1024**3
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Verify EXAONE embedding accessors work natively; patch only if broken
    try:
        base_model.get_input_embeddings()
    except (NotImplementedError, AttributeError):
        base_model.get_input_embeddings = lambda: base_model.transformer.wte
        logger.warning("Monkey-patched get_input_embeddings (model lacks native implementation)")
    try:
        base_model.get_output_embeddings()
    except (NotImplementedError, AttributeError):
        base_model.get_output_embeddings = lambda: base_model.lm_head
        logger.warning("Monkey-patched get_output_embeddings (model lacks native implementation)")

    mem_after_base = torch.cuda.memory_allocated() / 1024**3

    base_param_count = sum(p.numel() for p in base_model.parameters())
    logger.info(f"  Base model parameters: {base_param_count:,}")
    logger.info(f"  GPU memory used: {mem_after_base:.2f} GB")

    wandb.log(
        {
            "base_model_params": base_param_count,
            "gpu_mem_base_model_gb": mem_after_base,
        }
    )

    # Step 4: Load adapter and merge
    logger.info("[4/5] Loading LoRA adapter and merging...")
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    mem_after_adapter = torch.cuda.memory_allocated() / 1024**3

    # Count trainable (adapter) parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total params (with adapter): {total_params:,}")
    logger.info(f"  Trainable params (adapter): {trainable_params:,}")
    logger.info(f"  Trainable %: {100 * trainable_params / total_params:.4f}%")
    logger.info(f"  GPU memory (with adapter): {mem_after_adapter:.2f} GB")

    # Merge and unload
    logger.info("  Merging adapter weights into base model...")
    model = model.merge_and_unload()
    mem_after_merge = torch.cuda.memory_allocated() / 1024**3

    merged_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  Merged model parameters: {merged_param_count:,}")
    logger.info(f"  GPU memory after merge: {mem_after_merge:.2f} GB")

    # Validate: parameter count should match base model
    assert (
        merged_param_count == base_param_count
    ), f"Parameter count mismatch after merge: {merged_param_count} != {base_param_count}"
    logger.info("  [OK] Parameter count matches base model")

    # Check no adapter modules remain
    has_lora = any("lora" in name.lower() for name, _ in model.named_parameters())
    assert not has_lora, "LoRA modules still present after merge!"
    logger.info("  [OK] No LoRA modules remaining")

    wandb.log(
        {
            "adapter_trainable_params": trainable_params,
            "adapter_trainable_pct": 100 * trainable_params / total_params,
            "merged_param_count": merged_param_count,
            "gpu_mem_after_merge_gb": mem_after_merge,
            "merge_param_count_match": merged_param_count == base_param_count,
        }
    )

    # Step 5: Save merged model
    logger.info(f"[5/5] Saving merged model to {MERGED_OUTPUT_DIR}...")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    # Calculate saved model size
    total_size = sum(
        os.path.getsize(os.path.join(MERGED_OUTPUT_DIR, f))
        for f in os.listdir(MERGED_OUTPUT_DIR)
        if f.endswith((".safetensors", ".bin"))
    )
    model_size_gb = total_size / 1024**3
    logger.info(f"  Model size on disk: {model_size_gb:.2f} GB")

    # Sanity check: inference test
    logger.info("[Sanity Check] Running inference test...")
    test_prompt = "다음 민원을 분류해주세요: 우리 동네 도로에 포트홀이 생겨서 위험합니다."
    messages = [{"role": "user", "content": test_prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=False)
    logger.info(f"  Input: {test_prompt}")
    logger.info(f"  Output (first 300 chars): {response[:300]}")
    logger.info("  [OK] Inference test passed")

    wandb.log(
        {
            "merged_model_size_gb": model_size_gb,
            "sanity_check_input": test_prompt,
            "sanity_check_output": response[:500],
        }
    )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Stage 1 Complete! Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Merged model saved to: {MERGED_OUTPUT_DIR}")
    logger.info("=" * 60)

    wandb.log({"merge_time_seconds": elapsed})

    # Save merge log
    merge_log = {
        "stage": "1_lora_merge",
        "timestamp": datetime.now().isoformat(),
        "base_model": BASE_MODEL_ID,
        "adapter": ADAPTER_ID,
        "output_dir": MERGED_OUTPUT_DIR,
        "base_param_count": base_param_count,
        "adapter_trainable_params": trainable_params,
        "merged_param_count": merged_param_count,
        "model_size_gb": model_size_gb,
        "elapsed_seconds": elapsed,
        "adapter_config": adapter_config,
        "sanity_check_passed": True,
    }
    log_path = os.path.join(MERGED_OUTPUT_DIR, "merge_log.json")
    with open(log_path, "w") as f:
        json.dump(merge_log, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Merge log saved to: {log_path}")

    wandb.finish()

    # Free memory
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return merge_log


if __name__ == "__main__":
    main()
