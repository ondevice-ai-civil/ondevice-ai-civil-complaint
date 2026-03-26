import os
import sys
import time
import json
import argparse
import torch
import numpy as np
import wandb
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import bert_score

# 2024-12-11: transformers v5 업데이트(2026-02-06) 이전 마지막 호환 revision
EXAONE_REVISION = "0ff6b5ec7c13b049b253a16a889aa269e6b79a94"

# Paths (defaults)
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
ADAPTER_ID = "umyunsang/civil-complaint-exaone-lora"
DEFAULT_TEST_DATA_PATH = "data/processed/v2_test.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="GovOn v2 Model Evaluation")
    parser.add_argument("--model_path", type=str, default=ADAPTER_ID,
                        help="LoRA adapter ID or path")
    parser.add_argument("--data_path", type=str, default=DEFAULT_TEST_DATA_PATH,
                        help="Path to test data JSONL file")
    parser.add_argument("--output_report", type=str, default="evaluation_report.md",
                        help="Path to output evaluation report")
    return parser.parse_args()


def main():
    args = parse_args()

    # Force online mode if API key is present
    wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
    wandb.init(
        project="GovOn",
        name=f"m3-model-eval-{datetime.now().strftime('%m%d-%H%M')}",
        mode=wandb_mode,
    )

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True, revision=EXAONE_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=EXAONE_REVISION,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading Adapter...")
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()

    # Load test data
    actual_test_path = args.data_path
    if not os.path.exists(actual_test_path):
        print(f"Error: Test data not found at {actual_test_path}")
        return

    test_data = []
    with open(actual_test_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    test_data = test_data[:3]

    print(f"Evaluating {len(test_data)} samples...")
    latencies = []
    clean_gens = []
    clean_refs = []

    for item in test_data:
        # v2 데이터 포맷 대응 (text 필드에서 input 추출)
        input_text = item.get("input") or item.get("text", "")[:500]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        latencies.append(time.perf_counter() - start)
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        print(f"Gen: {gen[:30]}...")
        clean_gens.append(gen.strip())

        ref_text = item.get("output") or item.get("answer", "N/A")
        clean_refs.append(ref_text.strip())

    P, R, F1 = bert_score.score(clean_gens, clean_refs, lang="ko")
    metrics = {"avg_latency": np.mean(latencies), "bert_score_f1": F1.mean().item() * 100}
    wandb.log(metrics)
    print(f"Metrics: {metrics}")

    # 평가 리포트 파일 생성
    with open(args.output_report, "w") as f:
        f.write("| Metric | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| Avg Latency | {metrics['avg_latency']:.4f}s |\n")
        f.write(f"| BERTScore F1 | {metrics['bert_score_f1']:.2f} |\n")
        f.write(f"\nEvaluation completed at {datetime.now().isoformat()}")

    wandb.finish()


if __name__ == "__main__":
    main()
