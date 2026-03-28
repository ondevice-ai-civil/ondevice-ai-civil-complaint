"""
EXAONE M3 Evaluation Script (v3 Runtime)

Evaluates the fine-tuned EXAONE model on civil complaint test data
using BERTScore and latency metrics.

Note: As of transformers 4.53.0+, all previously monkey-patched APIs
(check_model_inputs, maybe_autocast, RopeParameters, ALL_ATTENTION_FUNCTIONS,
auto_docstring, can_return_tuple, etc.) exist natively. The broken
prepare_inputs_for_generation and embedding accessor overrides have been
removed in favor of the native implementations.
"""

import json
import time
from datetime import datetime

import bert_score
import numpy as np
import torch
import wandb
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Paths
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-Deep-7.8B"
ADAPTER_ID = "umyunsang/civil-complaint-exaone-lora"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"


def main():
    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-peft-final-verified-{datetime.now().strftime('%Y%m%d-%H%M')}",
    )

    logger.info("Loading 4-bit Model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    logger.info("Loading Adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model.eval()

    logger.info(f"Model input embeddings: {model.get_input_embeddings()}")

    # Data
    test_data = []
    with open(TEST_DATA_PATH, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    test_data = test_data[:5]  # Very small batch for final verification

    logger.info(f"Running Eval on {len(test_data)} samples...")
    latencies = []
    clean_gens = []
    clean_refs = []

    for item in test_data:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["input"][:500]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, repetition_penalty=1.1)
        latencies.append(time.perf_counter() - start)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        logger.info(f"Gen sample: {gen[:50]}...")
        clean_gens.append(gen.strip())
        clean_refs.append(item["output"].strip())

    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")
    metrics = {"avg_latency": np.mean(latencies), "bert_score_f1": F1.mean().item() * 100}
    wandb.log(metrics)
    logger.info(f"Final Result: {metrics}")
    wandb.finish()


if __name__ == "__main__":
    main()
