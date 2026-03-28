"""
EXAONE M3 Evaluation Script

Evaluates the fine-tuned EXAONE model on civil complaint test data
using BERTScore and latency metrics.

Note: As of transformers 4.53.0+, all previously monkey-patched APIs
(check_model_inputs, maybe_autocast, RopeParameters, ALL_ATTENTION_FUNCTIONS,
auto_docstring, can_return_tuple, etc.) exist natively. The broken
prepare_inputs_for_generation override (list-of-tuples format) has been
removed in favor of the native Cache-based implementation.
"""

import json
import re
import time
from datetime import datetime

import bert_score
import numpy as np
import torch
import wandb
from loguru import logger
from peft import PeftModel
from rouge_score import rouge_scorer  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Paths
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-Deep-7.8B"
ADAPTER_ID = "umyunsang/civil-complaint-exaone-lora"
TEST_DATA_PATH = "/content/ondevice-ai-civil-complaint/data/processed/civil_complaint_test.jsonl"


def main():
    wandb.init(
        project="exaone-civil-complaint",
        name=f"m3-exaone-final-verified-{datetime.now().strftime('%m%d-%H%M')}",
    )

    logger.info("Loading model...")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()

    # Data
    test_data = []
    with open(TEST_DATA_PATH, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    test_data = test_data[:5]

    logger.info(f"Evaluating {len(test_data)} samples...")
    latencies, gens, refs = [], [], []
    for item in test_data:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["input"][:500]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        latencies.append(time.perf_counter() - start)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        logger.info(f"Gen: {gen[:50]}...")
        gens.append(re.sub(r"<thought>.*?</thought>", "", gen, flags=re.DOTALL).strip())
        refs.append(re.sub(r"<thought>.*?</thought>", "", item["output"], flags=re.DOTALL).strip())

    P, R, F1 = bert_score.score(gens, refs, lang="ko")
    m = {"avg_latency": np.mean(latencies), "bert_score_f1": F1.mean().item() * 100}
    wandb.log(m)
    logger.info(f"Final M3 Result: {m}")
    wandb.finish()


if __name__ == "__main__":
    main()
